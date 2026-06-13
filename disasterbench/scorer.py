"""Scorer-only utilities for final scoring and contamination audits."""

import importlib.util
import json
import os
import re
import sys
from typing import Any

import gold_store

DISASTERBENCH_ROOT = os.environ.get(
    "DISASTERBENCH_ROOT",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "third_party", "DisasterBench_Open")
    ),
)


def audit_prompt_contamination(task_id: str, prompts: list[str]) -> dict[str, Any]:
    """Scan recorded prompts after execution; never called by repair methods."""
    annotations = gold_store.load_gold_data(task_id)
    expected_goal = annotations.get("expected_goal", "")
    first_failure = annotations.get("ground_truth_first_failure_position")

    leak_count = 0
    prompt_evidence = []
    for index, prompt in enumerate(prompts):
        leaks = []
        if expected_goal and expected_goal in prompt:
            leaks.append("expected_goal")
        if first_failure is not None and f"step {first_failure}" in prompt.lower():
            leaks.append("ground_truth_first_failure_position")
        if leaks:
            leak_count += 1
            prompt_evidence.append({
                "prompt_index": index,
                "matched_fields": leaks,
            })

    return {
        "pass": leak_count == 0,
        "leak_count": leak_count,
        "evidence": {
            "task_id": task_id,
            "prompt_evidence": prompt_evidence,
        } if prompt_evidence else {},
    }


def _load_disasterbench_reference(task_id: str) -> list[dict[str, Any]]:
    benchmark_path = os.path.join(DISASTERBENCH_ROOT, "data", "benchmark.jsonl")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("task_id")) == str(task_id):
                return record.get("structured_plan", [])
    raise KeyError(f"DisasterBench task not found: {task_id}")


def _load_disasterbench_evaluator():
    evaluator_path = os.path.join(DISASTERBENCH_ROOT, "evaluators", "evaluators.py")
    spec = importlib.util.spec_from_file_location(
        "disasterbench_evaluators", evaluator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DisasterBench evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("disasterbench_evaluators", module)
    spec.loader.exec_module(module)
    return module.Evaluator()


def score_disasterbench_plan(task_id: str, plan: Any) -> dict[str, Any]:
    """Score a finalized plan against DisasterBench references after repair."""
    evaluator = _load_disasterbench_evaluator()
    reference = _load_disasterbench_reference(task_id)
    canonical_plan = []
    if isinstance(plan, list):
        for index, step in enumerate(plan):
            if not isinstance(step, dict):
                continue
            raw_step = step.get("step")
            if raw_step is None:
                raw_step = step.get("step_idx", index + 1)
                try:
                    raw_step = int(raw_step) - 1
                except Exception:
                    raw_step = index
            dependencies = step.get("dependence", step.get("dependencies", []))
            if isinstance(dependencies, int):
                dependencies = [dependencies]
            normalized_dependencies = []
            for parent in dependencies or []:
                try:
                    parent = int(parent)
                except Exception:
                    continue
                if parent == -1:
                    normalized_dependencies.append(-1)
                elif parent >= 1:
                    normalized_dependencies.append(parent - 1)
            if not normalized_dependencies:
                normalized_dependencies = [-1]
            raw_inputs = step.get("inputs", step.get("params", {})) or {}
            normalized_inputs = {}
            inferred_dependence_content: dict[str, list[str]] = {}
            for key, value in raw_inputs.items():
                if isinstance(value, str):
                    match = re.fullmatch(
                        r"<GENERATED>-(\d+)-<?([^<>]+)>?", value
                    )
                    if match:
                        source = int(match.group(1))
                        output_name = match.group(2)
                        if source in dependencies and source >= 1:
                            source -= 1
                        value = f"<GENERATED>-{source}-<{output_name}>"
                        inferred_dependence_content.setdefault(str(source), []).append(output_name)
                normalized_inputs[key] = value
            dependence_content = step.get(
                "dependence_content", step.get("dependency_content")
            )
            if normalized_dependencies == [-1]:
                dependence_content = None
            elif not isinstance(dependence_content, dict):
                dependence_content = {}
            if normalized_dependencies != [-1] and inferred_dependence_content:
                dependence_content = inferred_dependence_content
            canonical_plan.append({
                "step": raw_step,
                "agent": step.get("agent", step.get("tool", step.get("agent_name"))),
                "inputs": normalized_inputs,
                "outputs": step.get("outputs", []),
                "dependence": normalized_dependencies,
                "dependence_content": dependence_content,
            })
    model_answer = json.dumps(canonical_plan, ensure_ascii=False)
    reference_answer = evaluator.extract_answer_from_gold_solution(reference)
    normalized = evaluator.normalize_answer_for_evaluation("cot", model_answer)
    tools_ok = evaluator.check_tools_correctness(normalized, reference_answer)
    params_ok = evaluator.check_parameters_correctness(normalized, reference_answer)
    deps_ok = evaluator.check_dependencies_correctness(normalized, reference_answer)
    fpof = evaluator.analyze_error_propagation(normalized, reference_answer)
    overall = (
        float(tools_ok) + float(params_ok) + float(deps_ok)
    ) / 3.0
    return {
        "overall": overall,
        "tools": bool(tools_ok),
        "params": bool(params_ok),
        "deps": bool(deps_ok),
        "fpof": fpof,
    }
