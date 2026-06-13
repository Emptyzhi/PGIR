"""Load benchmark tasks from official JSON files."""
import json
import os
import random
import re
from typing import List, Optional, Dict, Any

DISASTERBENCH_ROOT = os.environ.get(
    "DISASTERBENCH_ROOT",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "third_party", "DisasterBench_Open")
    ),
)
WILDCLAWBENCH_ROOT = os.environ.get(
    "WILDCLAWBENCH_ROOT",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "third_party", "WildClawBench")
    ),
)

class ToolSpec:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any],
                 outputs: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.outputs = outputs or {}

class Task:
    """A benchmark task as seen by repair methods."""
    def __init__(self, task_id: str, description: str,
                 steps_tools: List[str],
                 tools: List[ToolSpec],
                 workflow_structure: str = "linear",
                 failure_type: str = "tool_selection",
                 dependencies: Optional[Dict[int, List[int]]] = None,
                 benchmark_type: str = "disasterbench",
                 category: Optional[str] = None,
                 manual_contract_tree: Optional[dict] = None,
                 auto_contract_tree: Optional[dict] = None):
        self.task_id = task_id
        self.description = description
        self.steps_tools = steps_tools
        self.tools = tools
        self.workflow_structure = workflow_structure
        self.failure_type = failure_type
        self.benchmark_type = benchmark_type
        self.category = category if category else failure_type
        self.manual_contract_tree = manual_contract_tree
        self.auto_contract_tree = auto_contract_tree
        deps = None
        if dependencies is None:
            self.dependencies = {}
        else:
            self.dependencies = {int(k): v for k, v in dependencies.items()}

    @property
    def num_steps(self):
        return len(self.steps_tools)


def _sampling_labels_from_gold(raw: Dict[str, Any]) -> tuple[str, str]:
    """Derive hidden sampling strata without exposing the gold plan to methods."""
    plan = raw.get("structured_plan") or []
    if len(plan) <= 1:
        structure = "node"
    else:
        dependencies = []
        for step in plan:
            raw_deps = step.get("dependence") or []
            if not isinstance(raw_deps, list):
                raw_deps = [raw_deps]
            dependencies.append([dep for dep in raw_deps if dep not in (-1, "-1")])
        if any(len(deps) > 1 for deps in dependencies):
            structure = "dag"
        elif all(not deps if index == 0 else deps == [index - 1]
                 for index, deps in enumerate(dependencies)):
            structure = "chain"
        else:
            structure = "dag"

    has_dependency_content = any(
        bool(step.get("dependence_content")) for step in plan
    )
    has_fan_in = any(
        len([dep for dep in (step.get("dependence") or []) if dep not in (-1, "-1")]) > 1
        for step in plan
    )
    if has_fan_in:
        failure_type = "multi_dependency_propagation"
    elif has_dependency_content:
        failure_type = "dependency_propagation"
    elif len(plan) <= 1:
        failure_type = "single_tool_selection"
    else:
        failure_type = "typed_interface_planning"
    return structure, failure_type

def _stratified_sample(tasks: List[Task], target_size: int, stratify_keys: List[str],
                       categories_override: Optional[Dict[str, int]] = None) -> List[Task]:
    rng = random.Random(42)
    groups = {}
    for t in tasks:
        key = tuple(getattr(t, k, None) for k in stratify_keys)
        groups.setdefault(key, []).append(t)
    sample = []
    total = None
    selected_ids = None
    remaining_pool = None
    needed = None
    if categories_override:
        for cat_key, count in categories_override.items():
            matching = [t for t in tasks if t.category == cat_key]
            if len(matching) < count:
                raise ValueError(f"Not enough tasks for category {cat_key}: {len(matching)} < {count}")
            sample.extend(rng.sample(matching, count))
    else:
        total = len(tasks)
        remaining = target_size
        for key, group in groups.items():
            group_count = max(1, int(round(len(group) / total * target_size)))
            group_count = min(group_count, len(group))
            sample.extend(rng.sample(group, group_count))
            remaining -= group_count
        if len(sample) > target_size:
            sample = rng.sample(sample, target_size)
        elif len(sample) < target_size and len(sample) < len(tasks):
            selected_ids = {t.task_id for t in sample}
            remaining_pool = [t for t in tasks if t.task_id not in selected_ids]
            needed = target_size - len(sample)
            sample.extend(rng.sample(remaining_pool, needed))
    rng.shuffle(sample)
    return sample[:target_size]

def load_dataset(dataset_name: str, subset_size: Optional[int] = None,
                 stratify: bool = False, stratify_keys: Optional[List[str]] = None,
                 wilclawbench_categories: Optional[int] = None) -> List[Task]:
    filename = None
    bench_type = None
    if dataset_name.startswith("DisasterBench"):
        filename = os.path.join(DISASTERBENCH_ROOT, "data", "benchmark.jsonl")
        bench_type = "disasterbench"
    elif dataset_name.startswith("WildClawBench"):
        filename = os.path.join(WILDCLAWBENCH_ROOT, "tasks")
        bench_type = "wildclawbench"
    else:
        raise ValueError(f"Unknown dataset prefix: {dataset_name}")

    filepath = filename
    if bench_type == "wildclawbench":
        return _load_wildclawbench_tasks(
            subset_size=subset_size,
            stratify=stratify,
            stratify_keys=stratify_keys,
            wilclawbench_categories=wilclawbench_categories,
        )
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Benchmark file {filepath} not found. "
            "Set DISASTERBENCH_ROOT to the official DisasterBench checkout."
        )

    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            raw_tasks = [json.loads(line) for line in f if line.strip()]
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)

    manifest_path = os.path.join(
        DISASTERBENCH_ROOT, "interfaces", "tools", "tools_manifest.json"
    )
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    all_tasks = []
    for raw in raw_tasks:
        sampling_structure, sampling_failure_type = _sampling_labels_from_gold(raw)
        if raw.get("tools"):
            tools = [ToolSpec(t["name"], t["description"], t.get("parameters", {}), t.get("outputs", {}))
                     for t in raw.get("tools", [])]
        else:
            tools = [
                ToolSpec(name, spec.get("desc", ""), spec.get("input", {}), spec.get("output", {}))
                for name, spec in manifest.items()
            ]
        deps_raw = raw.get("dependencies")
        dependencies = None
        if deps_raw is not None:
            dependencies = {int(k): v for k, v in deps_raw.items()}
        task = Task(
            task_id=str(raw["task_id"]),
            description=raw.get("description", raw.get("task_desc", "")),
            steps_tools=[tool.name for tool in tools],
            tools=tools,
            workflow_structure=raw.get("workflow_structure", sampling_structure),
            failure_type=raw.get("failure_type", sampling_failure_type),
            dependencies=dependencies,
            benchmark_type=bench_type,
            category=raw.get("category"),
            manual_contract_tree=None,
            auto_contract_tree=None
        )
        all_tasks.append(task)

    fixed_task_ids = [
        item.strip()
        for item in os.environ.get("PGIR_FIXED_TASK_IDS", "").split(",")
        if item.strip()
    ]
    if fixed_task_ids and bench_type == "disasterbench":
        by_id = {task.task_id: task for task in all_tasks}
        missing = [task_id for task_id in fixed_task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Unknown PGIR_FIXED_TASK_IDS: {missing}")
        selected = [by_id[task_id] for task_id in fixed_task_ids]
        if subset_size is not None and len(selected) != subset_size:
            raise ValueError(
                f"PGIR_FIXED_TASK_IDS selected {len(selected)} tasks, expected {subset_size}"
            )
        return selected

    unique_cats = None
    per_cat = None
    remainder = None
    counts = None
    rng = None
    if subset_size is not None and subset_size < len(all_tasks):
        if stratify and stratify_keys:
            if bench_type == "wildclawbench" and wilclawbench_categories:
                unique_cats = list({t.category for t in all_tasks})
                per_cat = subset_size // wilclawbench_categories
                remainder = subset_size % wilclawbench_categories
                counts = {}
                for i, cat in enumerate(unique_cats):
                    count = per_cat + (1 if i < remainder else 0)
                    counts[cat] = count
                return _stratified_sample(all_tasks, subset_size, stratify_keys, counts)
            else:
                return _stratified_sample(all_tasks, subset_size, stratify_keys)
        else:
            rng = random.Random(42)
            rng.shuffle(all_tasks)
            return all_tasks[:subset_size]
    return all_tasks


def _parse_wildclaw_task_md(path: str) -> Task:
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    metadata: Dict[str, str] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
            body = parts[2]
    prompt_match = re.search(r"(?ms)^## Prompt\s*(.*?)(?=^## |\Z)", body)
    prompt = prompt_match.group(1).strip() if prompt_match else body.strip()
    task_id = metadata.get("id") or os.path.splitext(os.path.basename(path))[0]
    category = metadata.get("category") or os.path.basename(os.path.dirname(path))
    tools = [
        ToolSpec("shell", "Run shell commands in the task workspace.", {"command": "string"}, {"stdout": "string"}),
        ToolSpec("python", "Run Python scripts for data processing.", {"code": "string"}, {"stdout": "string"}),
        ToolSpec("browser", "Browse and inspect public web pages.", {"url": "string"}, {"page": "string"}),
        ToolSpec("search", "Search the web through the configured search backend.", {"query": "string"}, {"results": "list"}),
        ToolSpec("file_write", "Write final files under /tmp_workspace/results.", {"path": "string", "content": "string"}, {"path": "string"}),
    ]
    return Task(
        task_id=task_id,
        description=prompt,
        steps_tools=[tool.name for tool in tools],
        tools=tools,
        workflow_structure="wildclaw_agent_task",
        failure_type="end_to_end_agent_failure",
        dependencies={},
        benchmark_type="wildclawbench",
        category=category,
    )


def _load_wildclawbench_tasks(
    subset_size: Optional[int] = None,
    stratify: bool = False,
    stratify_keys: Optional[List[str]] = None,
    wilclawbench_categories: Optional[int] = None,
) -> List[Task]:
    tasks_root = os.path.join(WILDCLAWBENCH_ROOT, "tasks")
    if not os.path.isdir(tasks_root):
        raise FileNotFoundError(
            f"WildClawBench tasks directory not found: {tasks_root}. "
            "Set WILDCLAWBENCH_ROOT to the official checkout."
        )
    task_files = []
    for root, _, files in os.walk(tasks_root):
        for filename in files:
            if filename.endswith(".md") and "_task_" in filename:
                task_files.append(os.path.join(root, filename))
    task_files.sort()
    all_tasks = [_parse_wildclaw_task_md(path) for path in task_files]
    if not all_tasks:
        raise FileNotFoundError(f"No WildClawBench task markdown files found under {tasks_root}")
    if subset_size is not None and subset_size < len(all_tasks):
        if stratify:
            categories = sorted({task.category for task in all_tasks})
            if wilclawbench_categories:
                categories = categories[:wilclawbench_categories]
            per_cat = subset_size // max(len(categories), 1)
            remainder = subset_size % max(len(categories), 1)
            selected = []
            rng = random.Random(42)
            for index, category in enumerate(categories):
                candidates = [task for task in all_tasks if task.category == category]
                count = min(len(candidates), per_cat + (1 if index < remainder else 0))
                selected.extend(rng.sample(candidates, count))
            if len(selected) < subset_size:
                selected_ids = {task.task_id for task in selected}
                remaining = [task for task in all_tasks if task.task_id not in selected_ids]
                selected.extend(rng.sample(remaining, min(subset_size - len(selected), len(remaining))))
            rng.shuffle(selected)
            return selected[:subset_size]
        rng = random.Random(42)
        shuffled = list(all_tasks)
        rng.shuffle(shuffled)
        return shuffled[:subset_size]
    return all_tasks
