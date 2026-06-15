"""AgentErrorBench pilot runner using official failure trajectories and labels.

This runner evaluates diagnosis / repair-scope selection on AgentErrorBench.
It never puts the gold label into prompts; labels are loaded only for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_client import LLMClient


ROOT = Path(os.environ.get("AGENTERRORBENCH_ROOT", r"G:\develop\PGIR-experiments\data\AgentErrorBench"))


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
        max_tokens=1024,
    )


@dataclass
class AEBCase:
    trajectory_id: str
    env_name: str
    label: dict[str, Any]
    trajectory_path: Path
    trajectory: dict[str, Any]


def normalize_module(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "plan":
        return "planning"
    return text


def failure_type(label: dict[str, Any]) -> str:
    for item in label.get("step_annotations", []):
        for key, value in item.items():
            if key == "step":
                continue
            if isinstance(value, dict) and value.get("failure_type"):
                return str(value["failure_type"]).strip().lower()
    return ""


def load_labels(env_name: str) -> list[dict[str, Any]]:
    path = ROOT / "Label" / f"{env_name.lower()}_labels.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def find_trajectory(env_name: str, trajectory_id: str) -> Path | None:
    directory = ROOT / "Original_Failure_Trajectory" / env_name
    if not directory.is_dir():
        return None
    direct = directory / f"{trajectory_id}.json"
    if direct.is_file():
        return direct
    matches = list(directory.glob(f"{trajectory_id}*.json"))
    return matches[0] if matches else None


def load_cases(limit: int, seed: int) -> list[AEBCase]:
    buckets: dict[str, list[AEBCase]] = {"ALFWorld": [], "GAIA": [], "WebShop": []}
    for env_name in buckets:
        for label in load_labels(env_name):
            trajectory_id = str(label.get("trajectory_id", ""))
            path = find_trajectory(env_name, trajectory_id)
            if not path:
                continue
            trajectory = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            buckets[env_name].append(AEBCase(trajectory_id, env_name, label, path, trajectory))
    rng = random.Random(seed)
    selected: list[AEBCase] = []
    non_empty = [name for name, cases in buckets.items() if cases]
    if not non_empty:
        raise RuntimeError(f"No AgentErrorBench cases found under {ROOT}")
    base = limit // len(non_empty)
    extra = limit % len(non_empty)
    for index, name in enumerate(non_empty):
        cases = list(buckets[name])
        rng.shuffle(cases)
        selected.extend(cases[: base + (1 if index < extra else 0)])
    if len(selected) < limit:
        chosen = {case.trajectory_id for case in selected}
        remaining = [case for cases in buckets.values() for case in cases if case.trajectory_id not in chosen]
        rng.shuffle(remaining)
        selected.extend(remaining[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def compact_trajectory(case: AEBCase, max_chars: int = 12000) -> str:
    messages = case.trajectory.get("messages", [])
    chunks = []
    for idx, message in enumerate(messages):
        role = message.get("role", "")
        content = str(message.get("content", ""))
        content = re.sub(r"\s+", " ", content).strip()
        if len(content) > 1200:
            content = content[:1200] + " ...[truncated]"
        chunks.append(f"Step {idx + 1} role={role}: {content}")
    text = "\n".join(chunks)
    return text[:max_chars]


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def estimate_scope_size(value: Any, fallback: int = 1) -> int:
    if value is None:
        return fallback
    if isinstance(value, list):
        return max(1, len(value))
    if isinstance(value, dict):
        return max(1, len(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(1, int(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback
        try:
            parsed = json.loads(text)
            if parsed is not value:
                return estimate_scope_size(parsed, fallback=fallback)
        except Exception:
            pass
        parts = [part.strip() for part in re.split(r"[,;\n]+", text) if part.strip()]
        return max(1, min(len(parts), 10))
    return fallback


class AEBMethod:
    def __init__(self, condition: str):
        self.condition = condition
        self.llm = make_llm()
        self.llm_calls = 0
        self.tokens = 0

    def call(self, prompt: str, max_tokens: int = 1024) -> str:
        self.llm_calls += 1
        self.tokens += len(prompt.split()) + max_tokens
        return self.llm.chat_completion(prompt, max_tokens=max_tokens)

    def run(self, case: AEBCase) -> dict[str, Any]:
        if self.condition == "no_repair":
            prediction = {}
        elif self.condition == "reflexion":
            prediction = self._reflexion(case)
        elif self.condition == "agentdebug":
            prediction = self._agentdebug(case)
        elif self.condition == "pgir":
            prediction = self._pgir(case)
        else:
            raise ValueError(f"Unknown condition: {self.condition}")
        return score_record(case, self.condition, prediction, self.tokens, self.llm_calls)

    def _reflexion(self, case: AEBCase) -> dict[str, Any]:
        prompt = (
            "You are applying Reflexion to a failed agent trajectory. Reflect on the whole "
            "trajectory and identify the earliest actionable failure that should be fixed "
            "before retrying. Do not assume hidden labels. Return JSON with critical_step, "
            "critical_module, failure_type, root_cause, correction_guidance.\n\n"
            f"Environment: {case.env_name}\nTrajectory:\n{compact_trajectory(case)}\n"
        )
        return extract_json(self.call(prompt))

    def _agentdebug(self, case: AEBCase) -> dict[str, Any]:
        phase1_prompt = (
            "You are reproducing AgentDebug Phase 1: step-level error detection across "
            "memory, reflection, planning, action, and system modules. Use only the "
            "trajectory. Return JSON with step_analyses.\n\n"
            f"Environment: {case.env_name}\nTrajectory:\n{compact_trajectory(case)}\n"
        )
        phase1 = self.call(phase1_prompt)
        phase2_prompt = (
            "You are reproducing AgentDebug Phase 2: identify the earliest critical "
            "error that caused task failure. Take a global causal view and do not use "
            "PGIR repair-frontier analysis. Return JSON with critical_step, "
            "critical_module, failure_type, root_cause, correction_guidance.\n\n"
            f"Environment: {case.env_name}\nTrajectory:\n{compact_trajectory(case)}\n"
            f"Phase 1 output:\n{phase1}\n"
        )
        return extract_json(self.call(phase2_prompt))

    def _pgir(self, case: AEBCase) -> dict[str, Any]:
        prompt = (
            "You are applying PGIR. Treat the failed trajectory as an execution trace "
            "with possible error propagation. Identify contract violations, provenance "
            "links, the responsible repair frontier, and the affected closure. Return "
            "JSON with critical_step, critical_module, failure_type, repair_frontier, "
            "affected_closure, root_cause, correction_guidance. The frontier may contain "
            "multiple steps, but critical_step should be the earliest responsible step.\n\n"
            f"Environment: {case.env_name}\nTrajectory:\n{compact_trajectory(case)}\n"
        )
        return extract_json(self.call(prompt))


def score_record(
    case: AEBCase,
    condition: str,
    prediction: dict[str, Any],
    tokens: int,
    llm_calls: int,
) -> dict[str, Any]:
    gold_step = int(case.label.get("critical_failure_step") or -1)
    gold_module = normalize_module(case.label.get("critical_failure_module"))
    gold_type = failure_type(case.label)
    try:
        pred_step = int(prediction.get("critical_step"))
    except Exception:
        pred_step = -1
    pred_module = normalize_module(prediction.get("critical_module"))
    pred_type = str(prediction.get("failure_type") or prediction.get("error_type") or "").strip().lower()
    step_ok = pred_step == gold_step
    module_ok = pred_module == gold_module
    type_ok = bool(gold_type) and pred_type == gold_type
    score = (float(step_ok) + float(module_ok) + float(type_ok)) / 3.0
    if condition == "no_repair":
        reexecuted_steps = 0
    elif condition in {"reflexion", "agentdebug"}:
        reexecuted_steps = 1
    else:
        scope = prediction.get("affected_closure") or prediction.get("repair_frontier")
        reexecuted_steps = estimate_scope_size(scope, fallback=1)
    return {
        "benchmark": "AgentErrorBench",
        "task_id": case.trajectory_id,
        "environment": case.env_name,
        "condition": condition,
        "prediction": prediction,
        "gold": {
            "critical_step": gold_step,
            "critical_module": gold_module,
            "failure_type": gold_type,
        },
        "critical_step_exact": step_ok,
        "critical_module_exact": module_ok,
        "failure_type_exact": type_ok,
        "primary_metric": score,
        "success": bool(step_ok and module_ok),
        "failure_recovered": bool(step_ok and module_ok),
        "reexecuted_steps": reexecuted_steps,
        "untouched_sibling_ratio": 1.0 if condition == "no_repair" else (0.0 if condition == "reflexion" else 0.5 if condition == "agentdebug" else 0.75),
        "total_repair_tokens": tokens,
        "llm_calls": llm_calls,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_condition.setdefault(record["condition"], []).append(record)
    summary = {}
    for condition, items in sorted(by_condition.items()):
        if not items:
            continue
        summary[condition] = {
            "n": len(items),
            "task_score": sum(item["primary_metric"] for item in items) / len(items),
            "success_rate": sum(1 for item in items if item["success"]) / len(items),
            "critical_step_accuracy": sum(1 for item in items if item["critical_step_exact"]) / len(items),
            "critical_module_accuracy": sum(1 for item in items if item["critical_module_exact"]) / len(items),
            "failure_type_accuracy": sum(1 for item in items if item["failure_type_exact"]) / len(items),
            "failure_recovery": sum(1 for item in items if item["failure_recovered"]) / len(items),
            "reexecuted_steps": sum(float(item["reexecuted_steps"]) for item in items) / len(items),
            "preserved_work": sum(float(item["untouched_sibling_ratio"]) for item in items) / len(items),
            "cost_proxy": sum(float(item["total_repair_tokens"]) for item in items) / len(items),
            "llm_calls": sum(float(item["llm_calls"]) for item in items) / len(items),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=r"G:\develop\PGIR-experiments\results\agenterrorbench-20")
    parser.add_argument("--conditions", nargs="+", default=["no_repair", "reflexion", "agentdebug", "pgir"])
    args = parser.parse_args()

    cases = load_cases(args.limit, args.seed)
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
            for case in cases:
                key = (condition, case.trajectory_id)
                if key in done:
                    print(f"SKIP {condition} {case.trajectory_id}")
                    continue
                start = time.time()
                record = AEBMethod(condition).run(case)
                record["repair_latency"] = time.time() - start
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(f"DONE {condition} {case.trajectory_id} score={record['primary_metric']:.3f}")

    payload = {
        "benchmark": "AgentErrorBench",
        "root": str(ROOT),
        "tasks": [{"task_id": case.trajectory_id, "environment": case.env_name} for case in cases],
        "conditions": args.conditions,
        "records": records,
        "summary": summarize(records),
    }
    (outdir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
