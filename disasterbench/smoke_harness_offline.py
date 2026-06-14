"""Offline harness smoke test for PGIR experiment plumbing.

The real LLM smoke is provider-dependent. This script does not claim model
performance; it replaces LLM calls with a fixed public-task-derived plan so the
ExperimentHarness -> DisasterBench scorer -> results.json path can be checked
without network access or API balance.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "offline-smoke")
os.environ.setdefault("TAVILY_API_KEY", "offline-smoke")

import methods
from config import Config
from pgir_harness import ExperimentHarness


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "smoke_harness_offline_results"

OFFLINE_PLAN = [
    {
        "step_idx": 1,
        "tool": "Low-Light_Object_Detection",
        "params": {"image_path": "/data/satellite_images/low_light_image_01.jpg"},
        "dependencies": [],
        "outputs": ["detections_path"],
        "dependence_content": {},
    },
    {
        "step_idx": 2,
        "tool": "Foggy_Scenario_Object_Detection",
        "params": {"image_path": "/data/satellite_images/foggy_image_01.jpg"},
        "dependencies": [],
        "outputs": ["detected_objects_path"],
        "dependence_content": {},
    },
]


def fake_call_llm(self, prompt, max_tokens=None):
    if max_tokens is None:
        max_tokens = self.config.max_tokens_per_step
    self.total_tokens += len(prompt.split()) + max_tokens
    self.llm_calls += 1
    self.prompts.append(prompt)
    return json.dumps(OFFLINE_PLAN)


def main():
    methods.BaseRepairMethod.call_llm = fake_call_llm

    cfg = Config()
    cfg.results_dir = str(RESULTS_DIR)
    cfg.temperature = 0.0
    cfg.max_tokens_per_step = 256
    cfg.max_repair_tokens = 128
    cfg.phases_definition["codex_offline_harness_smoke"] = {
        "models": ["deepseek-v4-pro"],
        "datasets": ["DisasterBench_smoke"],
        "conditions": [
            "no_repair_control",
            "binary_checkpoint_rollback_control",
            "local_leaf_retry_no_ancestor_control",
            "no_contract_retry_control",
            "reflexion_verbal_retry",
            "post_tool_reflection_rag_repair",
            "agentrx_diagnosis_failure_localization",
            "agentfixer_single_trace_recommendation_retry",
            "pgir_hidden_taint_ancestor_repair",
            "full_trace_retry_control",
        ],
        "subset_sizes": {"DisasterBench_smoke": 1},
        "stratify": False,
        "seeds": [0],
    }
    harness = ExperimentHarness(cfg)
    harness.run_phases(["codex_offline_harness_smoke"])

    results_path = RESULTS_DIR / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    records = results["codex_offline_harness_smoke"]
    assert len(records) == 10, records
    assert not any("error" in record for record in records), records
    assert "primary_metric_summary" in results, results.keys()
    assert "tuning_parity" in results, results.keys()
    assert results["baseline_fairness_audit"]["pass"], results["baseline_fairness_audit"]
    assert "contamination_audit" in results, results.keys()

    summary = {
        "results_path": str(results_path),
        "records": [
            {
                "condition": record["condition"],
                "primary_metric": record.get("primary_metric"),
                "tools": record.get("scoring", {}).get("tools"),
                "params": record.get("scoring", {}).get("params"),
                "deps": record.get("scoring", {}).get("deps"),
                "repair_calls": record.get("repair_calls"),
                "global_escalations": record.get("global_escalations"),
                "local_patch_repairs": record.get("local_patch_repairs"),
            }
            for record in records
        ],
        "tuning_parity_pass": results["tuning_parity"].get("pass"),
        "baseline_fairness_audit_pass": results["baseline_fairness_audit"].get("pass"),
        "contamination_audit_pass": results["contamination_audit"].get("pass"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
