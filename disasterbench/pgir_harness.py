"""
Experiment harness that runs phases, collects results, and emits results.json
with tuning_parity and contamination_audit records.
"""
import json
import os
import math
import time
from config import Config
from data import load_dataset
from methods import (
    PGIRHiddenTaintAncestorRepair,
    FullTraceRetryControl,
    BinaryCheckpointRollbackControl,
    NoContractRetryControl,
    ReflexionVerbalRetry,
    PostToolReflectionRAGRepair,
    AgentRxDiagnosisFailureLocalization,
    AgentFixerSingleTraceRecommendationRetry,
    LocalLeafRetryNoAncestorControl,
    NoRepairControl,
    PGIRVisibleTaintLabels,
    PGIRNoProvenanceTaint,
    PGIRAutoContractTree,
    PGIRFullTraceRepair,
    PGIRDeterministicRulePatchOnly,
    PGIRVerifierGuidedLocalPruneOnly,
    PGIRLLMForcedLocalPatch,
    PGIRSelectiveVerifiedFallback,
    PGIRHardContractOnly,
    PGIRHardContractSelectiveFallback,
)
from scorer import audit_prompt_contamination, score_disasterbench_plan

CONDITION_CLASSES = {
    "pgir_hidden_taint_ancestor_repair": PGIRHiddenTaintAncestorRepair,
    "pgir_explicit_taint_ancestor_repair": PGIRVisibleTaintLabels,
    "pgir_visible_taint_ancestor_repair": PGIRVisibleTaintLabels,
    "pgir_full_trace_repair": PGIRFullTraceRepair,
    "pgir_deterministic_rule_patch_only": PGIRDeterministicRulePatchOnly,
    "pgir_verifier_guided_local_prune_only": PGIRVerifierGuidedLocalPruneOnly,
    "pgir_llm_forced_local_patch": PGIRLLMForcedLocalPatch,
    "pgir_selective_verified_fallback": PGIRSelectiveVerifiedFallback,
    "pgir_hard_contract_only": PGIRHardContractOnly,
    "pgir_hard_contract_selective_fallback": PGIRHardContractSelectiveFallback,
    "no_repair_control": NoRepairControl,
    "full_trace_retry_control": FullTraceRetryControl,
    "binary_checkpoint_rollback_control": BinaryCheckpointRollbackControl,
    "no_contract_retry_control": NoContractRetryControl,
    "reflexion_verbal_retry": ReflexionVerbalRetry,
    "post_tool_reflection_rag_repair": PostToolReflectionRAGRepair,
    "agentrx_diagnosis_failure_localization": AgentRxDiagnosisFailureLocalization,
    "agentfixer_single_trace_recommendation_retry": AgentFixerSingleTraceRecommendationRetry,
    "local_leaf_retry_no_ancestor_control": LocalLeafRetryNoAncestorControl,
    "pgir_visible_taint_labels": PGIRVisibleTaintLabels,
    "pgir_no_provenance_taint": PGIRNoProvenanceTaint,
    "pgir_auto_contract_tree": PGIRAutoContractTree,
}

class ExperimentHarness:
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        self.journal_path = os.path.join(self.config.results_dir, "pilot_journal.jsonl")
        self.completed_units = self._load_completed_units()

    def _unit_key(self, phase, model, dataset, condition, seed, task_id):
        return json.dumps(
            [phase, model, dataset, condition, int(seed), str(task_id)],
            ensure_ascii=False,
        )

    def _load_completed_units(self):
        completed = {}
        if not os.path.isfile(self.journal_path):
            return completed
        with open(self.journal_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("_unit_key")
                if key and "error" not in record:
                    completed[key] = record
        return completed

    def _append_journal(self, record):
        os.makedirs(self.config.results_dir, exist_ok=True)
        with open(self.journal_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_phases(self, phases=None):
        if phases is None:
            phases = self.config.execution_order
        for phase_name in phases:
            print(f"Starting phase: {phase_name}")
            phase_config = self.config.get_phase_config(phase_name)
            models = phase_config['models']
            datasets = phase_config['datasets']
            conditions = phase_config['conditions']
            phase_results = []
            for model in models:
                for dataset_name in datasets:
                    subset_size = phase_config['subset_sizes'].get(dataset_name)
                    stratify = phase_config.get('stratify', False)
                    stratify_keys = phase_config.get('stratify_keys')
                    wilclawbench_categories = phase_config.get('wilclawbench_categories')
                    load_subset_size = None if phase_config.get("fixed_task_ids") else subset_size
                    tasks = load_dataset(
                        dataset_name,
                        load_subset_size,
                        stratify,
                        stratify_keys,
                        wilclawbench_categories,
                    )
                    tasks = self._select_fixed_tasks(tasks, phase_config)
                    print(f"  Loaded {len(tasks)} tasks from {dataset_name}")
                    seeds = phase_config.get("seeds", [0])
                    for seed in seeds:
                        print(f"  Seed {seed}")
                        for task in tasks:
                            task._experiment_seed = seed
                        for cond_name in conditions:
                            print(f"    Condition {cond_name}")
                            if cond_name not in CONDITION_CLASSES:
                                raise ValueError(f"Unknown condition: {cond_name}")
                            constructor = CONDITION_CLASSES[cond_name]
                            for task in tasks:
                                unit_key = self._unit_key(
                                    phase_name, model, dataset_name, cond_name, seed, task.task_id
                                )
                                if unit_key in self.completed_units:
                                    phase_results.append(self.completed_units[unit_key])
                                    print(
                                        f"      SKIP completed condition={cond_name} seed={seed} "
                                        f"task={task.task_id}"
                                    )
                                    continue
                                try:
                                    # Isolate token counters, prompt audit records, and state per task.
                                    repair_method = constructor(model, self.config)
                                    repair_method.seed = seed
                                    result = repair_method.run_task_and_repair(task)
                                    result['phase'] = phase_name
                                    result['model'] = model
                                    result['dataset'] = dataset_name
                                    result['condition'] = cond_name
                                    result['seed'] = seed
                                    result.setdefault(
                                        "final_plan",
                                        getattr(repair_method, "last_plan", []),
                                    )
                                    result.setdefault(
                                        "initial_plan_fingerprint",
                                        getattr(repair_method, "initial_plan_fingerprint", ""),
                                    )
                                    result.setdefault(
                                        "initial_plan",
                                        getattr(repair_method, "initial_plan_snapshot", []),
                                    )
                                    result.setdefault("llm_calls", getattr(repair_method, "llm_calls", 0))
                                    result.setdefault(
                                        "repair_llm_calls",
                                        getattr(repair_method, "repair_llm_calls", 0),
                                    )
                                    result.setdefault(
                                        "search_calls",
                                        getattr(getattr(repair_method, "search", None), "calls_made", 0),
                                    )
                                    if getattr(task, "benchmark_type", "") == "wildclawbench":
                                        raise RuntimeError(
                                            "WildClawBench scoring requires the official "
                                            "OpenClaw/Docker evaluator. This PGIR planning "
                                            "harness may only preflight/sample WildClaw tasks."
                                        )
                                    result["scoring"] = score_disasterbench_plan(
                                        task.task_id,
                                        result.get("final_plan", []),
                                    )
                                    result["primary_metric"] = result["scoring"]["overall"]
                                    result.setdefault(
                                        "effective_budget",
                                        self._budget_for_result(result),
                                    )
                                    phase_results.append(result)
                                    result["_unit_key"] = unit_key
                                    self._append_journal(result)
                                    self.completed_units[unit_key] = result
                                    print(
                                        f"      DONE condition={cond_name} seed={seed} "
                                        f"task={task.task_id} primary={result['primary_metric']:.6f}"
                                    )
                                except Exception as e:
                                    print(f"ERROR on task {task.task_id} seed {seed} with condition {cond_name}: {e}")
                                    # store error record
                                    error_record = {
                                        "task_id": task.task_id,
                                        "seed": seed,
                                        "error": str(e),
                                        "phase": phase_name,
                                        "model": model,
                                        "dataset": dataset_name,
                                        "condition": cond_name,
                                        "_unit_key": unit_key,
                                    }
                                    phase_results.append(error_record)
                                    self._append_journal(error_record)
            self.results[phase_name] = phase_results
        self.compute_primary_metrics()
        self.check_tuning_parity()
        self.check_baseline_fairness()
        self.check_contamination()
        self.emit_results()

    def preflight(self, phases=None):
        if phases is None:
            phases = self.config.execution_order
        for phase_name in phases:
            phase_config = self.config.get_phase_config(phase_name)
            unknown = [
                condition for condition in phase_config["conditions"]
                if condition not in CONDITION_CLASSES
            ]
            if unknown:
                raise ValueError(f"Unknown condition(s) in {phase_name}: {unknown}")
            for model in phase_config["models"]:
                self.config.get_model_endpoint(model)
                self.config.get_model_api_key(model)
                self.config.get_model_id(model)
            for dataset_name in phase_config["datasets"]:
                subset_size = phase_config["subset_sizes"].get(dataset_name)
                load_subset_size = None if phase_config.get("fixed_task_ids") else subset_size
                tasks = load_dataset(
                    dataset_name,
                    load_subset_size,
                    phase_config.get("stratify", False),
                    phase_config.get("stratify_keys"),
                    phase_config.get("wilclawbench_categories"),
                )
                tasks = self._select_fixed_tasks(tasks, phase_config)
                print(
                    f"PREFLIGHT phase={phase_name} dataset={dataset_name} "
                    f"tasks={len(tasks)} models={phase_config['models']} "
                    f"conditions={len(phase_config['conditions'])}"
                )
                if dataset_name.startswith("WildClawBench"):
                    print(
                        "PREFLIGHT note: WildClawBench tasks require official "
                        "OpenClaw/Docker execution for scoring; this harness only "
                        "validates task discovery and sampling."
                    )

    def _select_fixed_tasks(self, tasks, phase_config):
        fixed_task_ids = [str(task_id) for task_id in phase_config.get("fixed_task_ids", [])]
        if not fixed_task_ids:
            return tasks
        by_id = {str(task.task_id): task for task in tasks}
        missing = [task_id for task_id in fixed_task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Configured fixed_task_ids not found: {missing}")
        return [by_id[task_id] for task_id in fixed_task_ids]

    def _budget_for_result(self, result):
        return {
            "model": result.get("model"),
            "temperature": self.config.temperature,
            "initial_token_cap": self.config.max_tokens_per_step,
            "repair_token_cap": self.config.max_repair_tokens,
            "frontier_retry_cap": getattr(self.config, "pgir_frontier_retry_cap", 1),
            "global_replan_retry_cap": getattr(self.config, "pgir_global_replan_retry_cap", 1),
            "tavily_call_cap": self.config.tavily_calls_per_task,
            "source": "config_effective_budget",
        }

    def compute_primary_metrics(self):
        summaries = {}
        for phase_name, phase_results in self.results.items():
            if not isinstance(phase_results, list):
                continue
            by_condition = {}
            for result in phase_results:
                if "error" in result:
                    continue
                condition = result.get("condition")
                by_condition.setdefault(condition, []).append(
                    float(result.get("primary_metric", 0.0))
                )
            summaries[phase_name] = {
                condition: {
                    "mean": sum(values) / len(values) if values else 0.0,
                    "n": len(values),
                }
                for condition, values in sorted(by_condition.items())
            }
        self.results["primary_metric_summary"] = summaries

    def check_tuning_parity(self):
        evidence = {}
        missing = []
        for phase_name, phase_results in self.results.items():
            if not isinstance(phase_results, list):
                continue
            for result in phase_results:
                if "error" in result:
                    missing.append({
                        "phase": phase_name,
                        "task_id": result.get("task_id"),
                        "condition": result.get("condition"),
                        "reason": "error_result_has_no_effective_budget",
                    })
                    continue
                budget = result.get("effective_budget")
                if not budget:
                    missing.append({
                        "phase": phase_name,
                        "task_id": result.get("task_id"),
                        "condition": result.get("condition"),
                        "reason": "missing_effective_budget",
                    })
                    continue
                group_key = json.dumps({
                    "phase": phase_name,
                    "model": result.get("model"),
                    "dataset": result.get("dataset"),
                    "task_id": result.get("task_id"),
                    "seed": result.get("seed", 0),
                }, sort_keys=True)
                evidence.setdefault(group_key, {})
                condition = result.get("condition")
                evidence[group_key][condition] = budget

        groups = []
        parity_failures = []
        for group_key, condition_budgets in evidence.items():
            canonical = {
                condition: json.dumps(budget, sort_keys=True)
                for condition, budget in condition_budgets.items()
            }
            unique = sorted(set(canonical.values()))
            group_record = {
                "group": json.loads(group_key),
                "conditions": sorted(condition_budgets),
                "unique_budget_count": len(unique),
                "budgets": condition_budgets,
            }
            groups.append(group_record)
            if len(unique) != 1:
                parity_failures.append(group_record)

        self.results['tuning_parity'] = {
            "pass": not missing and not parity_failures and bool(groups),
            "missing": missing,
            "failures": parity_failures,
            "groups_checked": len(groups),
            "evidence": groups,
        }

    def check_contamination(self):
        leak_count = 0
        evidence = []
        missing = []
        checked = 0
        promptless_checked = 0
        for phase_name, phase_results in self.results.items():
            if not isinstance(phase_results, list):
                continue
            for result in phase_results:
                if "error" in result:
                    missing.append({
                        "phase": phase_name,
                        "task_id": result.get("task_id"),
                        "condition": result.get("condition"),
                        "reason": "error_result_has_no_prompt_record",
                    })
                    continue
                prompts = result.get("prompts")
                if not prompts:
                    checked += 1
                    promptless_checked += 1
                    continue
                audit = audit_prompt_contamination(result.get("task_id"), prompts)
                checked += 1
                leak_count += audit["leak_count"]
                if audit["evidence"]:
                    evidence.append({
                        "phase": phase_name,
                        "model": result.get("model"),
                        "dataset": result.get("dataset"),
                        "condition": result.get("condition"),
                        "audit": audit,
                    })
        self.results['contamination_audit'] = {
            "pass": checked > 0 and not missing and leak_count == 0,
            "checked": checked,
            "promptless_checked": promptless_checked,
            "leak_count": leak_count,
            "missing": missing,
            "evidence": evidence,
        }

    def check_baseline_fairness(self):
        phases = {}
        for phase_name, phase_results in self.results.items():
            if not isinstance(phase_results, list):
                continue
            groups = {}
            errors = []
            for result in phase_results:
                if "error" in result:
                    errors.append({
                        "task_id": result.get("task_id"),
                        "condition": result.get("condition"),
                        "error": result.get("error"),
                    })
                    continue
                key = json.dumps({
                    "model": result.get("model"),
                    "dataset": result.get("dataset"),
                    "seed": result.get("seed", 0),
                    "task_id": str(result.get("task_id")),
                }, sort_keys=True)
                groups.setdefault(key, []).append(result)
            mismatches = []
            coverage = {}
            for key, records in groups.items():
                fingerprints = {
                    record.get("condition"): record.get("initial_plan_fingerprint", "")
                    for record in records
                }
                coverage[key] = sorted(fingerprints)
                if not fingerprints or "" in fingerprints.values() or len(set(fingerprints.values())) != 1:
                    mismatches.append({
                        "group": json.loads(key),
                        "fingerprints": fingerprints,
                    })
            expected_conditions = set(
                self.config.phases_definition.get(phase_name, {}).get("conditions", [])
            )
            missing_conditions = [
                {"group": json.loads(key), "missing": sorted(expected_conditions - set(conditions))}
                for key, conditions in coverage.items()
                if expected_conditions - set(conditions)
            ]
            phases[phase_name] = {
                "pass": bool(groups) and not errors and not mismatches and not missing_conditions,
                "groups_checked": len(groups),
                "errors": errors,
                "initial_plan_mismatches": mismatches,
                "missing_conditions": missing_conditions,
                "fixed_task_ids": self.config.phases_definition.get(phase_name, {}).get(
                    "fixed_task_ids", []
                ),
                "fairness_policy": {
                    "same_task_model_seed_initial_plan": True,
                    "same_public_tool_catalog": True,
                    "same_configured_token_caps": True,
                    "method_intrinsic_extra_calls_allowed_and_reported": True,
                    "gold_data_in_prompts_forbidden": True,
                },
            }
        self.results["baseline_fairness_audit"] = {
            "pass": bool(phases) and all(record["pass"] for record in phases.values()),
            "phases": phases,
        }

    def emit_results(self):
        os.makedirs(self.config.results_dir, exist_ok=True)
        outpath = os.path.join(self.config.results_dir, "results.json")
        with open(outpath, 'w') as f:
            json.dump(self.results, f, indent=2)
        metric_names = [
            "primary_metric",
            "total_repair_tokens",
            "repair_prompt_tokens",
            "reexecuted_steps",
            "untouched_sibling_ratio",
            "contract_pass_rate",
            "final_contract_pass_rate",
            "contract_activation_rate",
            "visible_taint_exposure",
            "visible_taint_configured",
            "visible_taint_prompt_exposure",
            "taint_precision",
            "cascade_depth",
            "repair_calls",
            "llm_calls",
            "repair_llm_calls",
            "search_calls",
            "boundary_repairs",
            "plan_boundary_triggers",
            "dependency_consumption_boundary_triggers",
            "fan_in_boundary_triggers",
            "fan_out_boundary_triggers",
            "final_boundary_triggers",
            "stopped_at_unrepaired_boundary",
            "executed_step_count",
            "local_patch_repairs",
            "deterministic_rule_patch_repairs",
            "llm_local_patch_repairs",
            "verifier_guided_local_prune_repairs",
            "interface_preserving_local_graph_rewrites",
            "llm_requested_global_replans",
            "rejected_patch_repairs",
            "selective_fallbacks",
            "unresolved_soft_semantic_advisories",
            "global_escalations",
            "blocking_taint_count",
            "non_blocking_deviation_count",
            "repair_frontier_size",
            "repair_frontier_ratio",
            "attempted_repair_frontier_size",
            "attempted_repair_frontier_ratio",
            "attempted_repair_scope_size",
            "attempted_repair_scope_ratio",
        ]
        for phase_name, phase_results in self.results.items():
            if not isinstance(phase_results, list):
                continue
            for result in phase_results:
                if "error" in result:
                    continue
                condition = result.get("condition")
                task_id = result.get("task_id")
                seed = result.get("seed", 0)
                for metric_name in metric_names:
                    value = float(result.get(metric_name, 0.0) or 0.0)
                    print(
                        f"condition={condition} phase={phase_name} seed={seed} "
                        f"task={task_id} {metric_name}: {value:.6f}"
                    )
            by_condition = {}
            for result in phase_results:
                if "error" in result:
                    continue
                by_condition.setdefault(result.get("condition"), []).append(result)
            for condition, records in sorted(by_condition.items()):
                for metric_name in metric_names:
                    values = [float(record.get(metric_name, 0.0) or 0.0) for record in records]
                    if not values:
                        continue
                    mean = sum(values) / len(values)
                    variance = sum((value - mean) ** 2 for value in values) / len(values)
                    std = math.sqrt(variance)
                    print(
                        f"SUMMARY condition={condition} metric={metric_name} "
                        f"mean={mean:.6f} std={std:.6f}"
                    )
                    print(f"condition={condition} {metric_name}: {mean:.6f}")
        print(f"Results written to {outpath}")
