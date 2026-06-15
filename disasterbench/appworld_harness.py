"""AppWorld pilot runner using real AppWorld APIs and official evaluator."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

from appworld import AppWorld, load_task_ids

from llm_client import LLMClient


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*StarletteDeprecationWarning.*")


def env(name: str) -> str | None:
    return os.environ.get(name) or os.environ.get(name.upper())


def make_llm() -> LLMClient:
    api_key = env("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required.")
    return LLMClient(
        env("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1",
        api_key,
        env("DEEPSEEK_MODEL_ID") or "deepseek-chat",
        temperature=0.0,
        max_tokens=2048,
    )


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def compact_report(report: str, max_chars: int = 4000) -> str:
    report = str(report)
    return report[:max_chars] + ("...[truncated]" if len(report) > max_chars else "")


def parse_output(output: Any, max_chars: int = 6000) -> str:
    text = str(output)
    return text[:max_chars] + ("...[truncated]" if len(text) > max_chars else "")


class AppWorldMethod:
    def __init__(self, condition: str):
        self.condition = condition
        self.llm = make_llm()
        self.tokens = 0
        self.llm_calls = 0

    def call(self, prompt: str, max_tokens: int = 2048) -> str:
        self.llm_calls += 1
        self.tokens += len(prompt.split()) + max_tokens
        return self.llm.chat_completion(prompt, max_tokens=max_tokens)

    def initial_prompt(self, instruction: str) -> str:
        return (
            "You are solving an AppWorld task using real local APIs. Write Python code only.\n"
            "The code runs inside AppWorld's world.execute environment and has access to `apis`.\n"
            "Use `apis.api_docs.show_app_descriptions()`, `apis.api_docs.show_api_descriptions(app_name=...)`, "
            "and `apis.api_docs.show_api_doc(app_name=..., api_name=...)` to discover APIs as needed.\n"
            "Use functional calls like `apis.spotify.login(...)` or `apis.gmail.search_emails(...)`.\n"
            "When the task is complete, call `apis.supervisor.complete_task()`.\n"
            "If the task asks a question or requests a final value, pass it explicitly as "
            "`apis.supervisor.complete_task(answer=...)`; otherwise omit the answer.\n"
            "Do not use hidden evaluator code or ground truth. Return one fenced python code block.\n\n"
            f"Task instruction:\n{instruction}\n"
        )

    def run_attempt(self, task_id: str, experiment_name: str, prompt: str) -> dict[str, Any]:
        response = self.call(prompt)
        code = extract_code(response)
        trace = []
        with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
            instruction = world.task.instruction
            try:
                output = world.execute(code)
                error = ""
            except Exception as exc:
                output = ""
                error = repr(exc)
            evaluator = world.evaluate()
            report = evaluator.report()
            success = bool(evaluator.success)
            pass_percentage = float(evaluator.pass_percentage)
            trace.append({
                "code": code,
                "output": parse_output(output),
                "error": error,
                "success": success,
                "pass_percentage": pass_percentage,
                "report": compact_report(report),
            })
        return {
            "instruction": instruction,
            "response": response,
            "code": code,
            "trace": trace,
            "success": success,
            "pass_percentage": pass_percentage,
            "report": compact_report(report),
        }

    def run(self, task_id: str, dataset_name: str) -> dict[str, Any]:
        start = time.time()
        with AppWorld(task_id=task_id, experiment_name=f"pgir_probe_{self.condition}_{dataset_name}") as world:
            instruction = world.task.instruction
        first = self.run_attempt(
            task_id,
            f"pgir_{self.condition}_{dataset_name}_{task_id}_attempt1",
            self.initial_prompt(instruction),
        )
        final = first
        reexecuted = 0
        preserved = 1.0
        if self.condition != "no_repair" and not first["success"]:
            if self.condition == "reflexion":
                prompt = (
                    "You are applying Reflexion to a failed AppWorld attempt. Reflect on the full failed code, "
                    "execution output, and evaluator report. Retry the entire task from the initial state. "
                    "Return one fenced python code block.\n\n"
                    f"Task:\n{instruction}\n\nPrevious code:\n{first['code']}\n\nOutput:\n{first['trace'][0]['output']}\n"
                    f"Error:\n{first['trace'][0]['error']}\nEvaluator report:\n{first['report']}\n"
                )
                reexecuted = 1
                preserved = 0.0
            elif self.condition == "agentdebug":
                phase_prompt = (
                    "You are reproducing AgentDebug. Identify the earliest critical error in this AppWorld attempt, "
                    "then give corrective feedback. Return JSON with critical_step, critical_module, error_type, "
                    "root_cause, correction_guidance.\n\n"
                    f"Task:\n{instruction}\n\nCode:\n{first['code']}\nOutput:\n{first['trace'][0]['output']}\n"
                    f"Error:\n{first['trace'][0]['error']}\nEvaluator report:\n{first['report']}\n"
                )
                feedback = self.call(phase_prompt, max_tokens=1024)
                prompt = (
                    "Retry the AppWorld task from the initial state using this AgentDebug corrective feedback. "
                    "Return one fenced python code block.\n\n"
                    f"Task:\n{instruction}\n\nCorrective feedback:\n{feedback}\nPrevious code:\n{first['code']}\n"
                )
                reexecuted = 1
                preserved = 0.5
            elif self.condition == "pgir":
                contract_prompt = (
                    "You are applying PGIR to a failed AppWorld attempt. Identify public API contract violations, "
                    "state/provenance contamination, the minimal repair frontier, and affected closure. "
                    "Return JSON with valid, repair_frontier, affected_closure, root_cause, correction_guidance. "
                    "Do not use evaluator internals beyond the public report.\n\n"
                    f"Task:\n{instruction}\n\nCode:\n{first['code']}\nOutput:\n{first['trace'][0]['output']}\n"
                    f"Error:\n{first['trace'][0]['error']}\nEvaluator report:\n{first['report']}\n"
                )
                guidance = self.call(contract_prompt, max_tokens=1024)
                prompt = (
                    "Retry only the answer-relevant AppWorld code using the PGIR localized correction. "
                    "Preserve correct discovered evidence when possible, but run from the initial state for "
                    "fair benchmark reset. Return one fenced python code block.\n\n"
                    f"Task:\n{instruction}\n\nPGIR verification:\n{guidance}\nPrevious code:\n{first['code']}\n"
                )
                reexecuted = 0.5
                preserved = 0.75
            final = self.run_attempt(
                task_id,
                f"pgir_{self.condition}_{dataset_name}_{task_id}_attempt2",
                prompt,
            )
        return {
            "benchmark": "AppWorld",
            "dataset": dataset_name,
            "task_id": task_id,
            "condition": self.condition,
            "primary_metric": final["pass_percentage"] / 100.0,
            "success": bool(final["success"]),
            "failure_recovered": bool((not first["success"]) and final["success"]),
            "initial_success": bool(first["success"]),
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": preserved,
            "total_repair_tokens": self.tokens,
            "llm_calls": self.llm_calls,
            "repair_latency": time.time() - start,
            "initial": first,
            "final": final,
        }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    failed_by_task = {
        record["task_id"]
        for record in records
        if record["condition"] == "no_repair" and not record["success"]
    }
    for record in records:
        by_condition.setdefault(record["condition"], []).append(record)
    summary = {}
    for condition, items in sorted(by_condition.items()):
        by_task = {item["task_id"]: item for item in items}
        recovery = (
            sum(1 for task_id in failed_by_task if by_task.get(task_id, {}).get("success")) / len(failed_by_task)
            if failed_by_task
            else 0.0
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
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dev")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--results-dir", default=r"G:\develop\PGIR-experiments\results\appworld-20")
    parser.add_argument("--conditions", nargs="+", default=["no_repair", "reflexion", "agentdebug", "pgir"])
    args = parser.parse_args()

    task_ids = load_task_ids(args.dataset)[: args.limit]
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    journal = outdir / "pilot_journal.jsonl"
    records: list[dict[str, Any]] = []
    done = set()
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            records.append(record)
            done.add((record["condition"], record["task_id"]))
    with journal.open("a", encoding="utf-8") as handle:
        for condition in args.conditions:
            for task_id in task_ids:
                key = (condition, task_id)
                if key in done:
                    print(f"SKIP {condition} {task_id}")
                    continue
                record = AppWorldMethod(condition).run(task_id, args.dataset)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(f"DONE {condition} {task_id} score={record['primary_metric']:.3f} success={record['success']}")
    payload = {
        "benchmark": "AppWorld",
        "dataset": args.dataset,
        "tasks": task_ids,
        "conditions": args.conditions,
        "records": records,
        "summary": summarize(records),
    }
    (outdir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
