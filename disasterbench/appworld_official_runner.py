"""Wrapper for AppWorld's official appworld-agents runner.

This script intentionally delegates task execution to the official
``appworld run auto`` CLI and only adds batching plus a compact summary.
Run it from the AppWorld virtual environment that has appworld-agents
installed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from appworld import load_task_ids


DEFAULT_ROOT = Path(os.environ.get("APPWORLD_ROOT", r"G:\develop\PGIR-experiments\data\AppWorld"))
DEFAULT_RESULTS = Path(r"G:\develop\PGIR-experiments\results\appworld-official-react-20")


def run_command(args: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.run(
            args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}. See {log_path}")


def experiment_output_root(root: Path, agent_name: str, model_name: str, dataset_name: str) -> Path:
    return root / "experiments" / "outputs" / agent_name / "deepseek" / model_name / dataset_name


def summarize_outputs(output_root: Path, summary_path: Path) -> dict[str, Any]:
    eval_dir = output_root / "evaluations"
    records = []
    token_totals = {"input_cache_miss": 0, "input_cache_hit": 0, "input_cache_write": 0, "output": 0}
    total_cost = 0.0
    for path in sorted(eval_dir.glob("on_only_*.json")):
        task_id = path.stem.replace("on_only_", "")
        data = json.loads(path.read_text(encoding="utf-8"))
        item = data["individual"][task_id]
        score = float(data["aggregate"]["task_goal_completion"]) / 100.0
        usage_path = output_root / "tasks" / task_id / "misc" / "usage.json"
        usage = {}
        if usage_path.is_file():
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
            for key in token_totals:
                token_totals[key] += int(usage.get("tokens", {}).get(key, 0))
            total_cost += sum(float(value) for value in usage.get("cost", {}).values())
        records.append(
            {
                "task_id": task_id,
                "success": bool(item["success"]),
                "task_score": score,
                "difficulty": item.get("difficulty"),
                "num_tests": item.get("num_tests"),
                "cost": sum(float(value) for value in usage.get("cost", {}).values()) if usage else 0.0,
            }
        )
    summary = {
        "n": len(records),
        "task_score": sum(record["task_score"] for record in records) / len(records) if records else 0.0,
        "success_rate": sum(1 for record in records if record["success"]) / len(records) if records else 0.0,
        "success_count": sum(1 for record in records if record["success"]),
        "token_totals": token_totals,
        "total_cost": total_cost,
        "avg_cost": total_cost / len(records) if records else 0.0,
        "output_root": str(output_root),
        "records": records,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--dataset", default="dev")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--agent-name", default="simplified_react_code_agent")
    parser.add_argument("--model-name", default="deepseek-v3.2-terminus-exp-without-reasoning")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--clear-first", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    results_dir = Path(args.results_dir)
    task_ids = args.task_id or load_task_ids(args.dataset)[: args.limit]
    output_root = experiment_output_root(root, args.agent_name, args.model_name, args.dataset)

    if not args.summarize_only:
        appworld = str(Path(sys.executable).with_name("appworld.exe"))
        for index, task_id in enumerate(task_ids):
            command = [
                appworld,
                "run",
                "auto",
                "--agent-name",
                args.agent_name,
                "--model-name",
                args.model_name,
                "--dataset-name",
                args.dataset,
                "--task-id",
                task_id,
                "--with-evaluation",
                "--root",
                str(root),
            ]
            if args.clear_first and index == 0:
                command.append("--clear-first")
            print(f"RUN {task_id}", flush=True)
            run_command(command, results_dir / "logs" / f"{task_id}.log")
            print(f"DONE {task_id}", flush=True)

    summary = summarize_outputs(output_root, results_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
