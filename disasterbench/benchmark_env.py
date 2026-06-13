"""Benchmark execution adapters for trace construction."""

import json
import os
from typing import Any

DISASTERBENCH_ROOT = os.environ.get(
    "DISASTERBENCH_ROOT",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "third_party", "DisasterBench_Open")
    ),
)


def _get_benchmark_module(benchmark_type: str):
    if benchmark_type == "wildclawbench":
        try:
            import wildclawbench
            return wildclawbench
        except ImportError as exc:
            raise RuntimeError(
                "WildClawBench must be run through its Docker/OpenClaw adapter."
            ) from exc
    raise ValueError(f"Unknown executable benchmark type: {benchmark_type}")


def execute_tool(task, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute or validate a single tool call for non-scoring trace evidence."""
    if task.benchmark_type == "disasterbench":
        manifest_path = os.path.join(
            DISASTERBENCH_ROOT, "interfaces", "tools", "tools_manifest.json"
        )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if tool_name not in manifest:
            return {
                "output": f"unknown_tool:{tool_name}",
                "success": False,
                "error": "tool_not_in_disasterbench_manifest",
            }
        spec = manifest[tool_name]
        required_inputs = set((spec.get("input") or {}).keys())
        provided_inputs = set((params or {}).keys())
        missing_inputs = sorted(required_inputs - provided_inputs)
        if missing_inputs:
            return {
                "output": json.dumps({
                    "tool": tool_name,
                    "validated_against": "DisasterBench tools_manifest",
                    "params": params,
                    "missing_inputs": missing_inputs,
                }, ensure_ascii=False),
                "success": False,
                "error": "missing_required_inputs",
            }
        return {
            "output": json.dumps({
                "tool": tool_name,
                "validated_against": "DisasterBench tools_manifest",
                "params": params,
            }, ensure_ascii=False),
            "success": True,
            "error": "",
        }

    bench = _get_benchmark_module(task.benchmark_type)
    result = bench.call_tool(tool_name, params)
    return {
        "output": result.get("output", ""),
        "success": result.get("success", False),
        "error": result.get("error", ""),
    }


def execute_task_plan(task, plan_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for step in plan_steps:
        step_idx = step["step_idx"]
        tool_name = step["tool"]
        params = step.get("params", {})
        outcome = execute_tool(task, tool_name, params)
        results.append({
            "step_idx": step_idx,
            "tool": tool_name,
            "params": params,
            "output": outcome["output"],
            "success": outcome["success"],
            "error": outcome["error"],
        })
    return results
