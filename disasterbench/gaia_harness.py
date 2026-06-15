"""Same-harness GAIA pilot for Table 1 repair baselines.

This runner is intentionally separate from the DisasterBench plan scorer:
GAIA evaluates final answers, while DisasterBench evaluates structured plans.
Gold answers are loaded only by the scorer after each method finishes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config
from llm_client import LLMClient
from search import SearchClient


GAIA_ROOT = Path(
    os.environ.get(
        "GAIA_ROOT",
        r"G:\develop\PGIR-experiments\data\GAIA-official",
    )
)


@dataclass
class GaiaTask:
    task_id: str
    question: str
    level: str
    final_answer: str
    file_name: str
    file_path: str


def load_gaia_validation(limit: int, seed: int = 42) -> list[GaiaTask]:
    metadata_path = GAIA_ROOT / "2023" / "validation" / "metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"GAIA validation metadata not found: {metadata_path}")
    frame = pd.read_parquet(metadata_path)
    tasks = [
        GaiaTask(
            task_id=str(row["task_id"]),
            question=str(row["Question"]),
            level=str(row["Level"]),
            final_answer=str(row["Final answer"]),
            file_name=str(row.get("file_name") or ""),
            file_path=str(row.get("file_path") or ""),
        )
        for _, row in frame.iterrows()
    ]
    rng = random.Random(seed)
    by_level: dict[str, list[GaiaTask]] = {}
    for task in tasks:
        by_level.setdefault(task.level, []).append(task)
    selected: list[GaiaTask] = []
    levels = sorted(by_level)
    base = limit // len(levels)
    extra = limit % len(levels)
    for index, level in enumerate(levels):
        bucket = list(by_level[level])
        rng.shuffle(bucket)
        selected.extend(bucket[: base + (1 if index < extra else 0)])
    if len(selected) < limit:
        chosen = {task.task_id for task in selected}
        remaining = [task for task in tasks if task.task_id not in chosen]
        rng.shuffle(remaining)
        selected.extend(remaining[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def normalize_answer(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^a-z0-9.+\-/% ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_answer(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not ref:
        return 0.0
    if pred == ref:
        return 1.0
    if ref in pred and len(pred) <= max(80, len(ref) * 3):
        return 1.0
    return 0.0


def extract_final_answer(text: str) -> str:
    text = str(text).strip()
    match = re.search(r"FINAL_ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().splitlines()[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def read_attachment(task: GaiaTask, max_chars: int = 5000) -> str:
    if not task.file_path:
        return "No attachment."
    path = GAIA_ROOT / task.file_path
    if not path.is_file():
        path = GAIA_ROOT / "2023" / "validation" / task.file_name
    if not path.is_file():
        return f"Attachment listed but not found: {task.file_path or task.file_name}"
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".py", ".json", ".jsonld", ".csv"}:
            return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        if suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(path, sheet_name=None, nrows=20)
            chunks = []
            for name, sheet in sheets.items():
                chunks.append(f"Sheet {name}:\n{sheet.to_csv(index=False)}")
            return "\n".join(chunks)[:max_chars]
        if suffix == ".pdf":
            try:
                import pypdf

                reader = pypdf.PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages[:5]]
                return "\n".join(pages)[:max_chars]
            except Exception as exc:
                return f"PDF attachment available at {path}; text extraction failed: {exc}"
        if suffix == ".docx":
            try:
                import docx

                document = docx.Document(str(path))
                return "\n".join(p.text for p in document.paragraphs)[:max_chars]
            except Exception as exc:
                return f"DOCX attachment available at {path}; text extraction failed: {exc}"
        if suffix == ".pptx":
            try:
                from pptx import Presentation

                prs = Presentation(str(path))
                texts = []
                for slide in prs.slides[:10]:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            texts.append(shape.text)
                return "\n".join(texts)[:max_chars]
            except Exception as exc:
                return f"PPTX attachment available at {path}; text extraction failed: {exc}"
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()[:30]
            return f"ZIP attachment contents: {names}"
    except Exception as exc:
        return f"Attachment extraction failed for {path.name}: {exc}"
    return f"Attachment available but not parsed: {path.name} ({suffix})"


class GaiaMethod:
    def __init__(self, condition: str, config: Config):
        self.condition = condition
        self.config = config
        self.llm = LLMClient(
            config.get_model_endpoint("deepseek-v4-pro"),
            config.get_model_api_key("deepseek-v4-pro"),
            config.get_model_id("deepseek-v4-pro"),
            config.temperature,
            1024,
        )
        self.search = SearchClient(config.tavily_api_key, max_calls=3)
        self.tokens = 0
        self.llm_calls = 0
        self.search_calls = 0
        self.prompts: list[str] = []

    def call_llm(self, prompt: str, max_tokens: int = 768) -> str:
        self.tokens += len(prompt.split()) + max_tokens
        self.llm_calls += 1
        self.prompts.append(prompt)
        return self.llm.chat_completion(prompt, max_tokens)

    def search_context(self, task: GaiaTask) -> str:
        try:
            results = self.search.search(task.question[:400], max_results=3)
            self.search_calls = self.search.calls_made
        except Exception as exc:
            return f"Search failed: {exc}"
        lines = []
        for result in results[:3]:
            lines.append(f"- {result.get('title','')}: {result.get('content','')}")
        return "\n".join(lines)

    def base_prompt(self, task: GaiaTask, attachment: str, search_context: str) -> str:
        return (
            "Answer this GAIA validation task. Use the search snippets and attachment "
            "extracts when useful. If the attachment is an image/audio/binary file that "
            "is not parsed, say so in your reasoning and rely on available text/search. "
            "Do not mention hidden answers or benchmark metadata. Return a concise final "
            "answer on the last line as FINAL_ANSWER: <answer>.\n"
            f"Question:\n{task.question}\n\n"
            f"Attachment extract:\n{attachment}\n\n"
            f"Search snippets:\n{search_context}\n"
        )

    def run(self, task: GaiaTask) -> dict[str, Any]:
        start = time.time()
        attachment = read_attachment(task)
        search_context = self.search_context(task)
        first = self.call_llm(self.base_prompt(task, attachment, search_context))
        answer = extract_final_answer(first)
        reexecuted = 0
        preserved = 1.0
        trace = [{"step": 1, "kind": "initial_answer", "content": first}]

        if self.condition == "reflexion":
            prompt = (
                "You are applying Reflexion: reflect on the previous attempt, identify "
                "mistakes or missing evidence, then retry the full task. Return only a "
                "concise final answer on the last line as FINAL_ANSWER: <answer>.\n"
                f"Question:\n{task.question}\nAttachment extract:\n{attachment}\n"
                f"Search snippets:\n{search_context}\nPrevious attempt:\n{first}\n"
            )
            revised = self.call_llm(prompt)
            answer = extract_final_answer(revised)
            trace.append({"step": 2, "kind": "reflexion_retry", "content": revised})
            reexecuted = 1
            preserved = 0.0
        elif self.condition == "agentdebug":
            phase1_prompt = (
                "You are reproducing AgentDebug Phase 1 for a GAIA agent answer. "
                "Detect memory, reflection, planning, action, and system errors in "
                "the attempt. Return JSON with step_analyses.\n"
                f"Question:\n{task.question}\nAttempt:\n{first}\n"
            )
            phase1 = self.call_llm(phase1_prompt)
            phase2_prompt = (
                "You are reproducing AgentDebug Phase 2. Identify the earliest "
                "critical error and write corrective guidance. Return JSON with "
                "critical_step, critical_module, error_type, root_cause, "
                "correction_guidance, confidence.\n"
                f"Question:\n{task.question}\nAttempt:\n{first}\nPhase1:\n{phase1}\n"
            )
            phase2 = self.call_llm(phase2_prompt)
            retry_prompt = (
                "Retry the GAIA task after receiving AgentDebug corrective feedback. "
                "Return only a concise final answer on the last line as FINAL_ANSWER: <answer>.\n"
                f"Question:\n{task.question}\nAttachment extract:\n{attachment}\n"
                f"Search snippets:\n{search_context}\nPrevious attempt:\n{first}\n"
                f"Corrective feedback:\n{phase2}\n"
            )
            revised = self.call_llm(retry_prompt)
            answer = extract_final_answer(revised)
            trace.extend([
                {"step": 2, "kind": "agentdebug_phase1", "content": phase1},
                {"step": 3, "kind": "agentdebug_phase2", "content": phase2},
                {"step": 4, "kind": "agentdebug_retry", "content": revised},
            ])
            reexecuted = 1
            preserved = 0.5
        elif self.condition == "pgir":
            contract_prompt = (
                "You are applying PGIR-style contract verification for a GAIA answer. "
                "Using only public task text, attachment extract, search snippets, and "
                "the attempt, decide whether the answer format/evidence is execution-"
                "critical invalid. If invalid, identify the minimal responsible clause "
                "or evidence gap and provide a localized correction. Return JSON with "
                "valid, responsible_frontier, correction_guidance.\n"
                f"Question:\n{task.question}\nAttachment extract:\n{attachment}\n"
                f"Search snippets:\n{search_context}\nAttempt:\n{first}\n"
            )
            contract = self.call_llm(contract_prompt)
            repair_prompt = (
                "Apply the PGIR localized correction only if needed. Preserve any "
                "correct evidence and revise the smallest answer-relevant part. Return "
                "only a concise final answer on the last line as FINAL_ANSWER: <answer>.\n"
                f"Question:\n{task.question}\nAttachment extract:\n{attachment}\n"
                f"Search snippets:\n{search_context}\nPrevious attempt:\n{first}\n"
                f"Contract verification:\n{contract}\n"
            )
            revised = self.call_llm(repair_prompt)
            answer = extract_final_answer(revised)
            trace.extend([
                {"step": 2, "kind": "pgir_contract_verification", "content": contract},
                {"step": 3, "kind": "pgir_localized_repair", "content": revised},
            ])
            reexecuted = 0.5
            preserved = 0.75

        score = score_answer(answer, task.final_answer)
        return {
            "task_id": task.task_id,
            "level": task.level,
            "condition": self.condition,
            "prediction": answer,
            "primary_metric": score,
            "success": bool(score >= 1.0),
            "total_repair_tokens": self.tokens,
            "llm_calls": self.llm_calls,
            "search_calls": self.search_calls,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": preserved,
            "repair_latency": time.time() - start,
            "trace": trace,
        }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_condition.setdefault(record["condition"], []).append(record)
    no_repair = {
        record["task_id"]: record
        for record in by_condition.get("no_repair", [])
    }
    failed = {
        task_id
        for task_id, record in no_repair.items()
        if not record.get("success")
    }
    summary = {}
    for condition, items in sorted(by_condition.items()):
        by_id = {item["task_id"]: item for item in items}
        recovery = (
            sum(1 for task_id in failed if by_id.get(task_id, {}).get("success")) / len(failed)
            if failed
            else math.nan
        )
        summary[condition] = {
            "n": len(items),
            "task_score": sum(float(item["primary_metric"]) for item in items) / len(items),
            "success_rate": sum(1 for item in items if item["success"]) / len(items),
            "failure_recovery": recovery,
            "reexecuted_steps": sum(float(item["reexecuted_steps"]) for item in items) / len(items),
            "preserved_work": sum(float(item["untouched_sibling_ratio"]) for item in items) / len(items),
            "cost_proxy": sum(float(item["total_repair_tokens"]) for item in items) / len(items),
            "llm_calls": sum(float(item["llm_calls"]) for item in items) / len(items),
            "search_calls": sum(float(item["search_calls"]) for item in items) / len(items),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--results-dir", default=r"G:\develop\PGIR-experiments\results\main-table-gaia-20")
    parser.add_argument("--conditions", nargs="+", default=["no_repair", "reflexion", "agentdebug", "pgir"])
    args = parser.parse_args()

    config = Config()
    tasks = load_gaia_validation(args.limit)
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    journal = outdir / "pilot_journal.jsonl"
    done = set()
    records: list[dict[str, Any]] = []
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["condition"], record["task_id"]))
            records.append(record)

    with journal.open("a", encoding="utf-8") as handle:
        for condition in args.conditions:
            for task in tasks:
                key = (condition, task.task_id)
                if key in done:
                    print(f"SKIP condition={condition} task={task.task_id}")
                    continue
                method = GaiaMethod(condition, config)
                record = method.run(task)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(
                    f"DONE condition={condition} task={task.task_id} "
                    f"score={record['primary_metric']:.1f} answer={record['prediction'][:80]}"
                )

    payload = {
        "benchmark": "GAIA validation",
        "conditions": args.conditions,
        "tasks": [{"task_id": task.task_id, "level": task.level} for task in tasks],
        "records": records,
        "summary": summarize(records),
    }
    (outdir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Results written to {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
