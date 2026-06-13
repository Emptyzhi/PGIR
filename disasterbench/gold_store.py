"""Load gold annotations for scoring. Only scorer imports this."""
import json
import os

_GOLD_CACHE = {}

def load_gold_data(task_id: str) -> dict:
    base_dir = None
    filepath = None
    if not _GOLD_CACHE:
        base_dir = "datasets"
        filepath = os.path.join(base_dir, "task_gold.json")
        if not os.path.exists(filepath):
            return {"expected_goal": "", "ground_truth_first_failure_position": None}
        with open(filepath, "r") as f:
            data = json.load(f)
        for entry in data:
            _GOLD_CACHE[entry["task_id"]] = {
                "expected_goal": entry.get("expected_goal", ""),
                "ground_truth_first_failure_position": entry.get("failure_step_index", None)
            }
    return _GOLD_CACHE.get(task_id, {"expected_goal": "", "ground_truth_first_failure_position": None})
