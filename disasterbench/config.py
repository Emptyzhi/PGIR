"""Experiment configuration, phase definitions, model endpoints."""
import os
from typing import Dict, Any, List

def _env(name: str):
    return os.environ.get(name) or os.environ.get(name.upper())

class Config:
    """PGIR experiment configuration."""
    def __init__(self):
        self.deepseek_api_key = _env("DEEPSEEK_API_KEY")
        self.yunwu_api_key = _env("YUNWU_API_KEY") or _env("MY_PROXY_API_KEY")
        self.aihubmix_api_key = _env("AIHUBMIX_API_KEY")
        self.gemini_api_key = _env("GEMINI_API_KEY")
        self.openai_api_key = _env("OPENAI_API_KEY")
        self.tavily_api_key = _env("TAVILY_API_KEY")
        self.yunwu_base_url = _env("YUNWU_BASE_URL") or "https://yunwu.ai/v1"
        self.aihubmix_base_url = _env("AIHUBMIX_BASE_URL") or "https://aihubmix.com/v1"
        self.proxy_api_key = self.yunwu_api_key or self.aihubmix_api_key
        self.proxy_base_url = (
            self.yunwu_base_url if self.yunwu_api_key
            else self.aihubmix_base_url if self.aihubmix_api_key
            else ""
        )

        if not self.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is required.")
        if not self.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable is required. Search backend is mandatory.")

        self.api_base_urls = {
            "deepseek-v4-pro": "https://api.deepseek.com/v1",
            "gemini-3-flash-preview": (
                self.proxy_base_url
                if self.proxy_api_key
                else "https://generativelanguage.googleapis.com/v1beta/openai"
            ),
            "gpt-4o-mini": (
                self.proxy_base_url
                if self.proxy_api_key
                else "https://api.openai.com/v1"
            )
        }
        self.api_keys = {
            "deepseek-v4-pro": self.deepseek_api_key,
            "gemini-3-flash-preview": self.proxy_api_key or self.gemini_api_key,
            "gpt-4o-mini": self.proxy_api_key or self.openai_api_key
        }
        self.model_ids = {
            "deepseek-v4-pro": "deepseek-chat",
            "gemini-3-flash-preview": _env("GEMINI_MODEL_ID") or "gemini-3-flash-preview",
            "gpt-4o-mini": _env("GPT_4O_MINI_MODEL_ID") or "gpt-4o-mini"
        }
        self.temperature = 0.0
        self.max_tokens_per_step = 2048
        self.max_repair_tokens = 1024
        self.pgir_frontier_retry_cap = int(_env("PGIR_FRONTIER_RETRY_CAP") or 1)
        self.pgir_global_replan_retry_cap = int(_env("PGIR_GLOBAL_REPLAN_RETRY_CAP") or 1)
        self.results_dir = _env("PGIR_RESULTS_DIR") or "."
        self.seed_policy = "fixed_per_task"

        self.contract_tree_manual_path = "data/manual_contract_trees.json"
        self.contract_tree_auto_path = "data/auto_contract_trees.json"

        self.tavily_calls_per_task = 3

        self.phases_definition = {
            "selective_fallback_unvetoed_10": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_local_graph_rewrite"],
                "conditions": [
                    "full_trace_retry_control",
                    "pgir_selective_verified_fallback",
                ],
                "subset_sizes": {"DisasterBench_local_graph_rewrite": 10},
                "stratify": False,
                "seeds": [0],
            },
            "selective_repair_diagnostic_10": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_local_graph_rewrite"],
                "conditions": [
                    "full_trace_retry_control",
                    "pgir_hidden_taint_ancestor_repair",
                    "pgir_selective_verified_fallback",
                    "pgir_hard_contract_only",
                    "pgir_hard_contract_selective_fallback",
                ],
                "subset_sizes": {"DisasterBench_local_graph_rewrite": 10},
                "stratify": False,
                "seeds": [0],
            },
            "local_graph_rewrite_operator_probe_3": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_local_graph_rewrite_probe"],
                "conditions": ["pgir_llm_forced_local_patch"],
                "subset_sizes": {"DisasterBench_local_graph_rewrite_probe": 3},
                "stratify": False,
                "seeds": [0],
            },
            "local_graph_rewrite_10": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_local_graph_rewrite"],
                "conditions": [
                    "full_trace_retry_control",
                    "pgir_hidden_taint_ancestor_repair",
                ],
                "subset_sizes": {"DisasterBench_local_graph_rewrite": 10},
                "stratify": False,
                "seeds": [0],
            },
            "targeted_disasterbench_ablation_18": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_targeted_ablation"],
                "conditions": [
                    "full_trace_retry_control",
                    "local_leaf_retry_no_ancestor_control",
                    "no_contract_retry_control",
                    "pgir_no_provenance_taint",
                    "pgir_hidden_taint_ancestor_repair",
                ],
                "subset_sizes": {"DisasterBench_targeted_ablation": 18},
                "stratify": False,
                "seeds": [0],
            },
            "pilot_disasterbench_50": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_pilot"],
                "conditions": [
                    "no_repair_control",
                    "full_trace_retry_control",
                    "pgir_hidden_taint_ancestor_repair",
                ],
                "subset_sizes": {"DisasterBench_pilot": 50},
                "stratify": True,
                "stratify_keys": ["workflow_structure", "failure_type"],
                "seeds": [0],
            },
            "smoke_test": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_smoke"],
                "conditions": [
                    "no_repair_control",
                    "pgir_hidden_taint_ancestor_repair",
                    "pgir_visible_taint_ancestor_repair",
                    "pgir_verifier_guided_local_prune_only",
                    "pgir_deterministic_rule_patch_only",
                    "pgir_llm_forced_local_patch",
                    "pgir_no_provenance_taint",
                    "reflexion_verbal_retry",
                    "post_tool_reflection_rag_repair",
                    "agentrx_diagnosis_failure_localization",
                    "binary_checkpoint_rollback_control",
                    "local_leaf_retry_no_ancestor_control",
                    "no_contract_retry_control",
                    "full_trace_retry_control"
                ],
                "subset_sizes": {"DisasterBench_smoke": 12},
                "stratify": False,
                "seeds": [0, 1, 2]
            },
            "cross_model_smoke": {
                "models": ["gemini-3-flash-preview", "gpt-4o-mini"],
                "datasets": ["DisasterBench_smoke"],
                "conditions": [
                    "pgir_hidden_taint_ancestor_repair",
                    "full_trace_retry_control"
                ],
                "subset_sizes": {"DisasterBench_smoke": 2},
                "stratify": False,
                "seeds": [0]
            },
            "pilot_main_paper_initial": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_pilot", "WildClawBench_pilot"],
                "conditions": [
                    "pgir_hidden_taint_ancestor_repair",
                    "pgir_visible_taint_ancestor_repair",
                    "reflexion_verbal_retry",
                    "post_tool_reflection_rag_repair",
                    "full_trace_retry_control",
                    "binary_checkpoint_rollback_control",
                    "local_leaf_retry_no_ancestor_control"
                ],
                "subset_sizes": {"DisasterBench_pilot": 80, "WildClawBench_pilot": 30},
                "stratify": True,
                "stratify_keys": ["workflow_structure", "failure_type"],
                "wilclawbench_categories": 6,
                "seeds": [0, 1, 2]
            },
            "formal_main_experiment": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_full_main", "WildClawBench_full_main"],
                "conditions": [
                    "pgir_hidden_taint_ancestor_repair",
                    "pgir_visible_taint_ancestor_repair",
                    "pgir_full_trace_repair",
                    "pgir_no_provenance_taint",
                    "reflexion_verbal_retry",
                    "post_tool_reflection_rag_repair",
                    "full_trace_retry_control",
                    "binary_checkpoint_rollback_control",
                    "local_leaf_retry_no_ancestor_control",
                    "no_contract_retry_control"
                ],
                "subset_sizes": {"DisasterBench_full_main": 233, "WildClawBench_full_main": 60},
                "stratify": False,
                "seeds": [0, 1, 2]
            },
            "low_cost_cross_model_validation": {
                "models": ["gemini-3-flash-preview", "gpt-4o-mini"],
                "datasets": ["DisasterBench_cross_model_subset", "WildClawBench_cross_model_subset"],
                "conditions": [
                    "pgir_hidden_taint_ancestor_repair",
                    "pgir_visible_taint_ancestor_repair",
                    "reflexion_verbal_retry",
                    "post_tool_reflection_rag_repair",
                    "full_trace_retry_control",
                    "binary_checkpoint_rollback_control"
                ],
                "subset_sizes": {"DisasterBench_cross_model_subset": 50, "WildClawBench_cross_model_subset": 18},
                "stratify": True,
                "stratify_keys": ["workflow_structure", "failure_type"],
                "wilclawbench_categories": 6,
                "seeds": [0, 1, 2]
            },
            "ablation_subset": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_cross_model_subset", "WildClawBench_cross_model_subset"],
                "conditions": [
                    "pgir_hidden_taint_ancestor_repair",
                    "pgir_visible_taint_ancestor_repair",
                    "pgir_full_trace_repair",
                    "pgir_verifier_guided_local_prune_only",
                    "pgir_deterministic_rule_patch_only",
                    "pgir_llm_forced_local_patch",
                    "pgir_no_provenance_taint",
                    "pgir_auto_contract_tree",
                    "no_contract_retry_control"
                ],
                "subset_sizes": {"DisasterBench_cross_model_subset": 50, "WildClawBench_cross_model_subset": 18},
                "stratify": True,
                "stratify_keys": ["workflow_structure", "failure_type"],
                "wilclawbench_categories": 6,
                "seeds": [0, 1, 2]
            },
            "diagnostic_secondary_baseline_subset": {
                "models": ["deepseek-v4-pro"],
                "datasets": ["DisasterBench_cross_model_subset", "WildClawBench_cross_model_subset"],
                "conditions": [
                    "pgir_hidden_taint_ancestor_repair",
                    "agentrx_diagnosis_failure_localization"
                ],
                "subset_sizes": {"DisasterBench_cross_model_subset": 50, "WildClawBench_cross_model_subset": 18},
                "stratify": True,
                "stratify_keys": ["workflow_structure", "failure_type"],
                "wilclawbench_categories": 6
            }
        }

        self.execution_order = [
            "smoke_test"
        ]

    def get_phase_config(self, phase_name):
        if phase_name not in self.phases_definition:
            raise ValueError(f"Unknown phase: {phase_name}")
        return self.phases_definition[phase_name]

    def get_model_endpoint(self, model_name):
        return self.api_base_urls.get(model_name, "")

    def get_model_api_key(self, model_name):
        api_key = self.api_keys.get(model_name, "")
        if not api_key:
            raise RuntimeError(f"Missing API key for requested model: {model_name}")
        return api_key

    def get_model_id(self, model_name):
        return self.model_ids.get(model_name, model_name)
