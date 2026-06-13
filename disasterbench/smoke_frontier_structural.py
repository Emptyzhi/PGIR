"""Structural smoke test for PGIR frontier repair behavior.

This test intentionally avoids external LLM/API calls. It verifies the control
flow that distinguishes localizable repair frontiers from non-local/global
replan cases after PGIR code changes.
"""

import json
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "structural-smoke")
os.environ.setdefault("TAVILY_API_KEY", "structural-smoke")

import benchmark_env as benv
import methods
from config import Config
from data import Task, ToolSpec
from methods import (
    PGIRDeterministicRulePatchOnly,
    PGIRFullTraceRepair,
    PGIRHiddenTaintAncestorRepair,
    PGIRHardContractOnly,
    PGIRLLMForcedLocalPatch,
    PGIRSelectiveVerifiedFallback,
    PGIRVerifierGuidedLocalPruneOnly,
    PGIRVisibleTaintLabels,
)


methods._deterministic_rule_repair_plan = lambda task, plan, failed_steps: []
methods._rule_failure_indices = lambda task, plan: (set(), None, False)


TOOLS = [
    ToolSpec("collect", "collect", {"query": "string"}),
    ToolSpec("transform", "transform", {"input": "string", "fixed": "string"}),
    ToolSpec("aggregate", "aggregate", {"input": "string"}),
    ToolSpec("unused_specialist", "unused", {"input": "string"}),
]

TASK = Task(
    "synthetic_frontier",
    "repair the contaminated execution trace",
    ["collect", "transform", "aggregate"],
    TOOLS,
    dependencies={1: [], 2: [1], 3: [2]},
    benchmark_type="synthetic",
)

PLAN = [
    {
        "step_idx": 1,
        "tool": "collect",
        "params": {"query": "q"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 2,
        "tool": "transform",
        "params": {"input": "<GENERATED>-1-output"},
        "dependencies": [1],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 3,
        "tool": "aggregate",
        "params": {"input": "<GENERATED>-2-output"},
        "dependencies": [2],
        "outputs": ["output"],
        "dependence_content": {},
    },
]

NO_EXPLICIT_DEPS_PLAN = [
    {
        "step_idx": 1,
        "tool": "collect",
        "params": {"query": "q"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 2,
        "tool": "transform",
        "params": {"input": "<GENERATED>-1-output"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 3,
        "tool": "aggregate",
        "params": {"input": "<GENERATED>-2-output"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
]

FANOUT_PLAN = [
    {
        "step_idx": 1,
        "tool": "collect",
        "params": {"query": "bad"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 2,
        "tool": "transform",
        "params": {"input": "<GENERATED>-1-output", "fixed": "yes"},
        "dependencies": [1],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 3,
        "tool": "aggregate",
        "params": {"input": "<GENERATED>-1-output"},
        "dependencies": [1],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 4,
        "tool": "collect",
        "params": {"query": "independent"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
]

LOCAL_GRAPH_REWRITE_PLAN = [
    {
        "step_idx": 1,
        "tool": "collect",
        "params": {"query": "bad"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 2,
        "tool": "aggregate",
        "params": {"input": "<GENERATED>-1-output"},
        "dependencies": [1],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 3,
        "tool": "collect",
        "params": {"query": "independent"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
]

MULTI_SOURCE_PLAN = [
    {
        "step_idx": 1,
        "tool": "collect",
        "params": {"query": "bad"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 2,
        "tool": "transform",
        "params": {"input": "raw"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 3,
        "tool": "aggregate",
        "params": {
            "input": "combine outputs",
        },
        "dependencies": [1, 2],
        "outputs": ["output"],
        "dependence_content": {"left": "Step 1 output", "right": "Step 2 output"},
    },
    {
        "step_idx": 4,
        "tool": "collect",
        "params": {"query": "independent"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
]

FINAL_COMMIT_PLAN = [
    {
        "step_idx": 1,
        "tool": "collect",
        "params": {"query": "ok"},
        "dependencies": [],
        "outputs": ["output"],
        "dependence_content": {},
    },
    {
        "step_idx": 2,
        "tool": "aggregate",
        "params": {"input": "<GENERATED>-1-output"},
        "dependencies": [1],
        "outputs": ["output"],
        "dependence_content": {},
    },
]


def fake_execute_tool(task, tool_name, params):
    if tool_name == "collect" and params.get("query") == "bad" and params.get("fixed") != "yes":
        return {"output": "bad collect", "success": False, "error": "root_failure"}
    if tool_name == "transform" and params.get("fixed") != "yes":
        return {"output": "bad transform", "success": False, "error": "semantic_failure"}
    return {"output": f"ok {tool_name}", "success": True, "error": ""}


benv.execute_tool = fake_execute_tool
methods.benv.execute_tool = fake_execute_tool


class TestConfig(Config):
    def __init__(self):
        super().__init__()
        self.pgir_frontier_retry_cap = 2
        self.pgir_global_replan_retry_cap = 1


class BaseSynthetic(PGIRHiddenTaintAncestorRepair):
    def _generate_initial_plan(self, task):
        return [dict(step) for step in PLAN]

    def _diagnose_semantic_failures(self, task, plan, step_results):
        return set()

    def _classify_step_outcome(self, step_idx, step, result, contract):
        if step_idx == 2 and not result.get("success"):
            return self._make_contamination_event(
                2,
                "blocking_taint",
                "tool_execution_failure",
                [2],
                result.get("error") or "failed",
                "action_output",
            )
        return self._make_contamination_event(
            step_idx,
            "pass",
            "none",
            [],
            "",
            "action_output",
        )

    def _call_llm_repair(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(
            {
                "patch_steps": [
                    {
                        "step_idx": 2,
                        "tool": "transform",
                        "params": {
                            "input": "<GENERATED>-1-output",
                            "fixed": "yes",
                        },
                        "dependencies": [1],
                        "outputs": ["output"],
                        "dependence_content": {},
                    }
                ]
            }
        )


class LocalizablePGIR(BaseSynthetic):
    pass


class VisiblePGIR(PGIRVisibleTaintLabels, BaseSynthetic):
    pass


class NonLocalPGIR(BaseSynthetic):
    def _classify_step_outcome(self, step_idx, step, result, contract):
        if step_idx == 1:
            return self._make_contamination_event(
                1,
                "blocking_taint",
                "root_plan_violation",
                [1],
                "root contaminates all descendants",
                "action_output",
            )
        return self._make_contamination_event(
            step_idx,
            "pass",
            "none",
            [],
            "",
            "action_output",
        )

    def _call_llm_repair(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(PLAN)


class FullTraceSynthetic(PGIRFullTraceRepair):
    def _generate_initial_plan(self, task):
        return [dict(step) for step in PLAN]

    def _diagnose_semantic_failures(self, task, plan, step_results):
        return set()

    def _classify_step_outcome(self, step_idx, step, result, contract):
        if step_idx == 2 and not result.get("success"):
            return self._make_contamination_event(
                2,
                "blocking_taint",
                "tool_execution_failure",
                [2],
                result.get("error") or "failed",
                "action_output",
            )
        return self._make_contamination_event(
            step_idx,
            "pass",
            "none",
            [],
            "",
            "action_output",
        )

    def _call_llm_repair(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(PLAN)


class FanoutPGIR(PGIRHiddenTaintAncestorRepair):
    def _generate_initial_plan(self, task):
        return [dict(step) for step in FANOUT_PLAN]

    def _diagnose_semantic_failures(self, task, plan, step_results):
        return set()

    def _allow_verifier_guided_local_prune(self):
        return False

    def _allow_deterministic_rule_patch(self):
        return False

    def _classify_step_outcome(self, step_idx, step, result, contract):
        if step_idx == 1 and not result.get("success"):
            return self._make_contamination_event(
                1,
                "blocking_taint",
                "tool_execution_failure",
                [1],
                result.get("error") or "failed",
                "action_output",
            )
        return self._make_contamination_event(
            step_idx,
            "pass",
            "none",
            [],
            "",
            "action_output",
        )

    def _call_llm_repair(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(
            {
                "patch_steps": [
                    {
                        "step_idx": 1,
                        "tool": "collect",
                        "params": {"query": "bad", "fixed": "yes"},
                        "dependencies": [],
                        "outputs": ["output"],
                        "dependence_content": {},
                    }
                ]
            }
        )


class MultiSourcePGIR(PGIRHiddenTaintAncestorRepair):
    def _generate_initial_plan(self, task):
        return [dict(step) for step in MULTI_SOURCE_PLAN]

    def _diagnose_semantic_failures(self, task, plan, step_results):
        return set()

    def _allow_verifier_guided_local_prune(self):
        return False

    def _allow_deterministic_rule_patch(self):
        return False

    def _classify_step_outcome(self, step_idx, step, result, contract):
        if step_idx in {1, 2} and not result.get("success"):
            return self._make_contamination_event(
                step_idx,
                "blocking_taint",
                "tool_execution_failure",
                [step_idx],
                result.get("error") or "failed",
                "action_output",
            )
        return self._make_contamination_event(
            step_idx,
            "pass",
            "none",
            [],
            "",
            "action_output",
        )

    def _call_llm_repair(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(
            {
                "patch_steps": [
                    {
                        "step_idx": 1,
                        "tool": "collect",
                        "params": {"query": "bad", "fixed": "yes"},
                        "dependencies": [],
                        "outputs": ["output"],
                        "dependence_content": {},
                    },
                    {
                        "step_idx": 2,
                        "tool": "transform",
                        "params": {"input": "raw", "fixed": "yes"},
                        "dependencies": [],
                        "outputs": ["output"],
                        "dependence_content": {},
                    },
                ]
            }
        )


class LocalGraphRewritePGIR(PGIRHiddenTaintAncestorRepair):
    def _generate_initial_plan(self, task):
        return [dict(step) for step in LOCAL_GRAPH_REWRITE_PLAN]

    def _diagnose_semantic_failures(self, task, plan, step_results):
        return set()

    def _allow_verifier_guided_local_prune(self):
        return False

    def _allow_deterministic_rule_patch(self):
        return False

    def _classify_step_outcome(self, step_idx, step, result, contract):
        if step_idx == 1 and not result.get("success"):
            return self._make_contamination_event(
                1,
                "blocking_taint",
                "tool_execution_failure",
                [1],
                result.get("error") or "failed",
                "action_output",
            )
        return self._make_contamination_event(
            step_idx, "pass", "none", [], "", "action_output"
        )

    def _call_llm_repair(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(
            {
                "repaired_plan": [
                    {
                        "step_idx": 1,
                        "tool": "collect",
                        "params": {"query": "bad", "fixed": "yes"},
                        "dependencies": [],
                        "outputs": ["output"],
                        "dependence_content": {},
                    },
                    {
                        "step_idx": 2,
                        "tool": "transform",
                        "params": {"input": "<GENERATED>-1-output", "fixed": "yes"},
                        "dependencies": [1],
                        "outputs": ["output"],
                        "dependence_content": {},
                    },
                    {
                        "step_idx": 3,
                        "tool": "aggregate",
                        "params": {"input": "<GENERATED>-2-output"},
                        "dependencies": [2],
                        "outputs": ["output"],
                        "dependence_content": {},
                    },
                    {
                        "step_idx": 4,
                        "tool": "collect",
                        "params": {"query": "independent"},
                        "dependencies": [],
                        "outputs": ["output"],
                        "dependence_content": {},
                    },
                ]
            }
        )


class FinalCommitPGIR(PGIRHiddenTaintAncestorRepair):
    def _generate_initial_plan(self, task):
        return [dict(step) for step in FINAL_COMMIT_PLAN]

    def _diagnose_semantic_failures(self, task, plan, step_results):
        return {2}

    def _allow_verifier_guided_local_prune(self):
        return False

    def _allow_deterministic_rule_patch(self):
        return False

    def _classify_step_outcome(self, step_idx, step, result, contract):
        return self._make_contamination_event(
            step_idx,
            "pass",
            "none",
            [],
            "",
            "action_output",
        )

    def _call_llm_repair(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(
            {
                "patch_steps": [
                    {
                        "step_idx": 2,
                        "tool": "aggregate",
                        "params": {"input": "<GENERATED>-1-output", "fixed": "yes"},
                        "dependencies": [1],
                        "outputs": ["output"],
                        "dependence_content": {"1": ["output"]},
                    }
                ]
            }
        )


class UnrepairableConsumptionPGIR(BaseSynthetic):
    def _allow_verifier_guided_local_prune(self):
        return False

    def _allow_deterministic_rule_patch(self):
        return False

    def _call_llm_repair(self, prompt):
        self.prompts.append(prompt)
        return json.dumps({"patch_steps": []})


class SelectiveFallbackPGIR(UnrepairableConsumptionPGIR):
    def _selective_global_fallback_enabled(self):
        return True

    def _call_llm_repair(self, prompt):
        self.prompts.append(prompt)
        if "Escalate once to a complete ancestor/root-path replan" in prompt:
            repaired = [dict(step) for step in PLAN]
            repaired[1] = dict(repaired[1])
            repaired[1]["params"] = {
                "input": "<GENERATED>-1-output",
                "fixed": "yes",
            }
            return json.dumps(repaired)
        return json.dumps({"patch_steps": []})


class HardContractIgnoresSoftSemanticPGIR(PGIRHardContractOnly):
    def _generate_initial_plan(self, task):
        return [dict(step) for step in FINAL_COMMIT_PLAN]

    def _call_llm_repair(self, prompt):
        raise AssertionError("hard-contract-only condition should not call repair for soft semantics")


class RuleOnlySynthetic(PGIRDeterministicRulePatchOnly, BaseSynthetic):
    pass


class ForcedLLMSynthetic(PGIRLLMForcedLocalPatch, BaseSynthetic):
    pass


class VerifierPruneSynthetic(PGIRVerifierGuidedLocalPruneOnly, BaseSynthetic):
    def run_task_and_repair(self, task):
        self._diagnosed_step_budget = 1
        return super().run_task_and_repair(task)


def main():
    cfg = TestConfig()
    runners = {
        "local": LocalizablePGIR("deepseek-v4-pro", cfg),
        "visible": VisiblePGIR("deepseek-v4-pro", cfg),
        "nonlocal": NonLocalPGIR("deepseek-v4-pro", cfg),
        "rule_only": RuleOnlySynthetic("deepseek-v4-pro", cfg),
        "verifier_prune": VerifierPruneSynthetic("deepseek-v4-pro", cfg),
        "llm_forced": ForcedLLMSynthetic("deepseek-v4-pro", cfg),
        "pgir_full_trace": FullTraceSynthetic("deepseek-v4-pro", cfg),
        "fanout": FanoutPGIR("deepseek-v4-pro", cfg),
        "multi_source": MultiSourcePGIR("deepseek-v4-pro", cfg),
        "local_graph_rewrite": LocalGraphRewritePGIR("deepseek-v4-pro", cfg),
        "final_commit": FinalCommitPGIR("deepseek-v4-pro", cfg),
        "unrepairable_consumption": UnrepairableConsumptionPGIR("deepseek-v4-pro", cfg),
        "selective_fallback": SelectiveFallbackPGIR("deepseek-v4-pro", cfg),
        "hard_contract_only": HardContractIgnoresSoftSemanticPGIR("deepseek-v4-pro", cfg),
    }
    results = {name: runner.run_task_and_repair(TASK) for name, runner in runners.items()}

    helper = LocalizablePGIR("deepseek-v4-pro", cfg)
    deps = {1: [], 2: [1], 3: [2]}
    child_map = helper._child_map(deps)
    nonblocking = helper._classify_plan_boundary_violation(
        3,
        PLAN
        + [
            {
                "step_idx": 4,
                "tool": "collect",
                "params": {"query": "unused"},
                "dependencies": [],
                "outputs": ["output"],
                "dependence_content": {},
            }
        ],
        deps,
        child_map,
        plan_budget=3,
    )
    nonblocking_boundary = helper._is_repair_boundary(
        3,
        PLAN,
        deps,
        child_map,
        [nonblocking],
        set(),
    )
    inferred_deps = methods._plan_dependencies(NO_EXPLICIT_DEPS_PLAN)
    dependence_content_only_deps = methods._plan_dependencies([
        {
            "step_idx": 1,
            "tool": "collect",
            "params": {"query": "q"},
            "dependencies": [],
            "outputs": ["output"],
            "dependence_content": {},
        },
        {
            "step_idx": 2,
            "tool": "aggregate",
            "params": {"input": "plain text"},
            "dependencies": [],
            "outputs": ["output"],
            "dependence_content": {
                "left": "Uses Step 1 output",
                "right": {"source": "parent 1"},
            },
        },
    ])
    params_text_ref_deps = methods._plan_dependencies([
        {
            "step_idx": 1,
            "tool": "collect",
            "params": {"query": "q"},
            "dependencies": [],
            "outputs": ["output"],
            "dependence_content": {},
        },
        {
            "step_idx": 2,
            "tool": "aggregate",
            "params": {
                "input": "Uses Step 1 output",
                "source": "parent 1",
            },
            "dependencies": [],
            "outputs": ["output"],
            "dependence_content": {},
        },
    ])
    fanout_deps = {1: [], 2: [1], 3: [1], 4: [1]}
    fanout_child_map = helper._child_map(fanout_deps)
    fanout_event = helper._make_contamination_event(
        1,
        "blocking_taint",
        "output_contract_violation",
        [1],
        "root output invalid",
        "action_output",
    )
    fanout_boundary_type = helper._repair_boundary_type(
        1,
        NO_EXPLICIT_DEPS_PLAN
        + [
            {
                "step_idx": 4,
                "tool": "aggregate",
                "params": {"input": "<GENERATED>-1-output"},
                "dependencies": [1],
                "outputs": ["output"],
                "dependence_content": {},
            }
        ],
        fanout_deps,
        fanout_child_map,
        [fanout_event],
        {1},
    )
    incomplete_scope_nonlocal = helper._frontier_is_not_local({1}, {1}, FANOUT_PLAN)
    fanin_events = helper._verify_dependency_consumption(
        3,
        {1: [], 2: [], 3: [1, 2]},
        {1, 2},
    )

    summary = {}
    for name, result in results.items():
        summary[name] = {
            key: result.get(key)
            for key in [
                "global_escalations",
                "local_patch_repairs",
                "repair_frontier_nodes",
                "affected_nodes",
                "repair_attempts",
                "visible_taint_exposure",
                "reexecuted_steps",
                "repair_boundary_triggers",
                "dependency_consumption_boundary_triggers",
                "fan_in_boundary_triggers",
                "fan_out_boundary_triggers",
                "stopped_at_unrepaired_boundary",
                "unrepaired_boundary_type",
                "unrepaired_boundary_step",
                "unrepaired_boundary_reason",
                "executed_step_count",
                "final_output",
            ]
        }
    summary["visible"]["has_visible_prompt"] = any(
        "VISIBLE_TAINT_LABELS" in prompt for prompt in runners["visible"].prompts
    )
    summary["local"]["hidden_prompt_has_visible_labels"] = any(
        "VISIBLE_TAINT_LABELS" in prompt for prompt in runners["local"].prompts
    )
    summary["local"]["hidden_prompt_has_contamination_event_json"] = any(
        '"violation_type"' in prompt or '"responsible_nodes"' in prompt
        for prompt in runners["local"].prompts
    )
    summary["nonblocking_helper"] = nonblocking
    summary["nonblocking_triggers_repair"] = nonblocking_boundary
    summary["inferred_deps"] = inferred_deps
    summary["dependence_content_only_deps"] = dependence_content_only_deps
    summary["params_text_ref_deps"] = params_text_ref_deps
    summary["fanout_boundary_type"] = fanout_boundary_type
    summary["incomplete_scope_nonlocal"] = incomplete_scope_nonlocal
    summary["fanin_events"] = fanin_events
    removed_adapter_symbol = "_schema" + "_text" + "_direct" + "_plan"
    summary["adapter_removed"] = not hasattr(methods, removed_adapter_symbol)
    summary["repair_catalog_has_unused_tool"] = (
        "unused_specialist" in helper._repair_tool_catalog(TASK, PLAN)
    )
    selective_policy = PGIRSelectiveVerifiedFallback("deepseek-v4-pro", cfg)
    summary["selective_fallback_unvetoed"] = selective_policy._global_replan_is_acceptable(
        [PLAN[0]], PLAN, TASK
    )
    summary["selective_fallback_skips_global_rules"] = (
        not selective_policy._allow_global_deterministic_rule_patch()
    )

    assert results["local"]["global_escalations"] == 0, summary
    assert results["local"]["local_patch_repairs"] == 1, summary
    assert results["local"]["repair_frontier_nodes"] == [2], summary
    assert results["local"]["affected_nodes"] == [2, 3], summary
    assert results["local"]["dependency_consumption_boundary_triggers"] == 1, summary
    assert not summary["local"]["hidden_prompt_has_visible_labels"], summary
    assert not summary["local"]["hidden_prompt_has_contamination_event_json"], summary
    assert results["visible"]["visible_taint_exposure"] == 1.0, summary
    assert results["visible"]["visible_taint_prompt_exposure"] == 1.0, summary
    assert summary["visible"]["has_visible_prompt"], summary
    assert results["nonlocal"]["global_escalations"] == 1, summary
    assert results["nonlocal"]["affected_nodes"] == [1, 2, 3], summary
    assert results["rule_only"]["local_patch_repairs"] == 0, summary
    assert (
        results["rule_only"]["repair_attempts"][0]["source"]
        == "local_operator_disabled"
    ), summary
    assert results["verifier_prune"]["local_patch_repairs"] == 1, summary
    assert results["verifier_prune"]["verifier_guided_local_prune_repairs"] == 1, summary
    assert results["verifier_prune"]["repair_attempts"][0]["source"] == "verifier_guided_local_prune", summary
    assert results["verifier_prune"]["global_escalations"] == 0, summary
    assert results["llm_forced"]["local_patch_repairs"] == 1, summary
    assert results["llm_forced"]["llm_local_patch_repairs"] == 1, summary
    assert results["llm_forced"]["deterministic_rule_patch_repairs"] == 0, summary
    assert results["pgir_full_trace"]["global_escalations"] == 1, summary
    assert results["pgir_full_trace"]["affected_nodes"] == [1, 2, 3], summary
    assert results["fanout"]["global_escalations"] == 0, summary
    assert results["fanout"]["local_patch_repairs"] == 1, summary
    assert results["fanout"]["repair_frontier_nodes"] == [1], summary
    assert results["fanout"]["affected_nodes"] == [1, 2, 3], summary
    assert results["fanout"]["fan_out_boundary_triggers"] == 1, summary
    assert results["fanout"]["repair_boundary_triggers"][0]["type"] == "fan_out_spread", summary
    assert results["fanout"]["repair_boundary_triggers"][0]["step_idx"] == 1, summary
    assert results["multi_source"]["global_escalations"] == 0, summary
    assert results["multi_source"]["local_patch_repairs"] == 1, summary
    assert results["multi_source"]["repair_frontier_nodes"] == [1, 2], summary
    assert results["multi_source"]["affected_nodes"] == [1, 2, 3], summary
    assert results["multi_source"]["fan_in_boundary_triggers"] == 1, summary
    assert results["local_graph_rewrite"]["global_escalations"] == 0, summary
    assert results["local_graph_rewrite"]["local_patch_repairs"] == 1, summary
    assert results["local_graph_rewrite"]["repair_frontier_nodes"] == [1], summary
    assert results["local_graph_rewrite"]["affected_nodes"] == [1, 2], summary
    assert results["local_graph_rewrite"]["reexecuted_steps"] == 3, summary
    assert len(runners["local_graph_rewrite"].last_plan) == 4, summary
    assert runners["local_graph_rewrite"].last_plan[3]["params"]["query"] == "independent", summary
    assert (
        results["local_graph_rewrite"]["repair_attempts"][0]["reason"]
        == "accepted_interface_preserving_local_graph_rewrite"
    ), summary
    assert results["final_commit"]["global_escalations"] == 0, summary
    assert results["final_commit"]["local_patch_repairs"] == 1, summary
    assert results["final_commit"]["final_boundary_triggers"] == 1, summary
    assert results["final_commit"]["repair_boundary_triggers"][0]["type"] == "final_commit", summary
    assert results["unrepairable_consumption"]["stopped_at_unrepaired_boundary"] is True, summary
    assert results["unrepairable_consumption"]["unrepaired_boundary_type"] == "dependency_consumption", summary
    assert results["unrepairable_consumption"]["unrepaired_boundary_step"] == 3, summary
    assert results["unrepairable_consumption"]["executed_step_count"] == 2, summary
    assert results["unrepairable_consumption"]["final_output"] == "bad transform", summary
    assert results["selective_fallback"]["stopped_at_unrepaired_boundary"] is False, summary
    assert results["selective_fallback"]["global_escalations"] == 1, summary
    assert results["selective_fallback"]["selective_fallbacks"] == 1, summary
    assert results["hard_contract_only"]["repair_calls"] == 0, summary
    assert results["hard_contract_only"]["verifier_policy"] == "hard_contract_only", summary
    assert nonblocking["status"] == "non_blocking_deviation", summary
    assert nonblocking_boundary is False, summary
    assert inferred_deps == {1: [], 2: [1], 3: [2]}, summary
    assert dependence_content_only_deps == {1: [], 2: [1]}, summary
    assert params_text_ref_deps == {1: [], 2: [1]}, summary
    assert fanout_boundary_type == "fan_out_spread", summary
    assert incomplete_scope_nonlocal is True, summary
    assert fanin_events and fanin_events[0]["boundary"] == "fan_in_aggregation", summary
    assert fanin_events[0]["responsible_nodes"] == [1, 2], summary
    assert summary["adapter_removed"], summary
    assert summary["repair_catalog_has_unused_tool"], summary
    assert summary["selective_fallback_unvetoed"], summary
    assert summary["selective_fallback_skips_global_rules"], summary

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
