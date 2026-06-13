"""Structural fault-injection diagnostic for PGIR repair locality.

This test does not claim task-success performance. It verifies the structural
hypothesis that localized replay is useful only when the responsible frontier
induces a strict subgraph of the execution graph.
"""

import json
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "fault-injection")
os.environ.setdefault("TAVILY_API_KEY", "fault-injection")

from config import Config
from methods import PGIRHiddenTaintAncestorRepair


CASES = {
    "late_chain_error": {
        "deps": {1: [], 2: [1], 3: [2], 4: [3]},
        "responsible": [3],
        "expected_scope": [3, 4],
        "expected_local": True,
    },
    "independent_branch_error": {
        "deps": {1: [], 2: [], 3: [1], 4: [2]},
        "responsible": [1],
        "expected_scope": [1, 3],
        "expected_local": True,
    },
    "multi_root_fanin_error": {
        "deps": {1: [], 2: [], 3: [1], 4: [2, 3]},
        "responsible": [2, 3],
        "expected_scope": [2, 3, 4],
        "expected_local": True,
    },
    "root_plan_error": {
        "deps": {1: [], 2: [1], 3: [2], 4: [3]},
        "responsible": [1],
        "expected_scope": [1, 2, 3, 4],
        "expected_local": False,
    },
}


def make_plan(deps):
    return [
        {
            "step_idx": node,
            "tool": f"tool_{node}",
            "params": {},
            "dependencies": parents,
            "outputs": ["output"],
            "dependence_content": {},
        }
        for node, parents in sorted(deps.items())
    ]


def main():
    method = PGIRHiddenTaintAncestorRepair("deepseek-v4-pro", Config())
    records = []
    for name, case in CASES.items():
        plan = make_plan(case["deps"])
        events = [
            method._make_contamination_event(
                node,
                "blocking_taint",
                "injected_failure",
                [node],
                "controlled fault injection",
                "action_output",
            )
            for node in case["responsible"]
        ]
        frontier = method._select_repair_frontier(events, case["deps"])
        scope = method._repair_scope_from_frontier(frontier, case["deps"], len(plan))
        local = not method._frontier_is_not_local(frontier, scope, plan)
        record = {
            "case": name,
            "frontier": sorted(frontier),
            "scope": sorted(scope),
            "localizable": local,
            "localized_replay_steps": len(scope),
            "full_trace_replay_steps": len(plan),
            "saved_replay_steps": len(plan) - len(scope) if local else 0,
        }
        assert record["scope"] == case["expected_scope"], record
        assert local is case["expected_local"], record
        records.append(record)

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
