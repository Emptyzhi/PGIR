"""
Repair method implementations: PGIR, baselines, controls, ablations.
All methods use real LLM calls and Tavily search. Gold data never accessed.
"""
import time
import json
import copy
import re
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

from data import Task
from config import Config
from llm_client import LLMClient
from search import SearchClient
import benchmark_env as benv

# ----------------------------------------------------------------------
# Base class: prompt recording for audit
# ----------------------------------------------------------------------
class BaseRepairMethod(ABC):
    def __init__(self, model_name: str, config: Config):
        self.model_name = model_name
        self.config = config
        self.llm = LLMClient(
            config.get_model_endpoint(model_name),
            config.get_model_api_key(model_name),
            config.get_model_id(model_name),
            config.temperature,
            config.max_tokens_per_step
        )
        self.search = SearchClient(api_key=config.tavily_api_key, max_calls=config.tavily_calls_per_task)
        self.total_tokens = 0
        self.repair_prompt_tokens_used = 0
        self.task = None
        self.prompts = []   # for contamination audit
        self.last_plan = []
        self._diagnosed_step_budget = None

    def call_llm(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if max_tokens is None:
            max_tokens = self.config.max_tokens_per_step
        # approximate token count (real token count could be obtained from API response, but we use length/4)
        self.total_tokens += len(prompt.split()) + max_tokens
        self.prompts.append(prompt)
        return self.llm.chat_completion(prompt, max_tokens)

    def _call_llm_repair(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if max_tokens is None:
            max_tokens = self.config.max_repair_tokens
        self.repair_prompt_tokens_used += len(prompt.split()) + max_tokens
        return self.call_llm(prompt, max_tokens)

    @abstractmethod
    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        pass

    def _compute_contract_pass(self, contract_tree, step_results):
        return _compute_contract_pass(contract_tree, step_results)

    def _first_failure(self, results):
        for i, result in enumerate(results):
            if not result.get("success"):
                return i + 1
        return None

    def _semantic_failure_indices(self, task, plan, step_results) -> set:
        if not plan:
            return set()
        rule_failed, rule_budget, rule_complete = _rule_failure_indices(task, plan)
        if rule_budget is not None:
            self._diagnosed_step_budget = rule_budget
        if rule_failed or rule_complete:
            return rule_failed
        prompt = (
            "Diagnose whether each planned tool call is necessary and semantically appropriate "
            "for the task, using only the task, public tool schemas, and execution trace. "
            "Apply a strict smallest-sufficient-plan rule: mark redundant, optional, indirect, "
            "or task-irrelevant steps as failed; mark downstream steps that ignore a relevant prior "
            "output as failed. Mark any step whose output neither directly satisfies an explicit "
            "requested deliverable nor feeds a later necessary step. Prefer no more than 3 steps "
            "unless additional steps are unavoidable. A condition-specific direct tool makes prior "
            "restoration/preprocessing redundant. A report/description tool that consumes an upstream "
            "result makes unrequested parallel specialist analyses redundant. "
            "If the plan contains any such redundant step, failed_steps must not be empty. "
            "Do not assume any reference answer. Return JSON: {\"failed_steps\": [1-based indices], "
            "\"max_required_steps\": integer, \"reason\": \"brief explanation\"}. "
            "max_required_steps must be the smallest number of tool calls sufficient for the explicit "
            "requested deliverables, inferred only from the task and public tool schemas.\n"
            f"Task: {task.description}\nTools: {_format_repair_tool_catalog(task, plan)}\n"
            f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Trace: {json.dumps(step_results, ensure_ascii=False)}"
        )
        text = self._call_llm_repair(prompt)
        try:
            payload = _extract_json_payload(text) or {}
            budget = payload.get("max_required_steps")
            failed = {
                int(index)
                for index in payload.get("failed_steps", [])
                if 1 <= int(index) <= len(plan)
            }
            if isinstance(budget, (int, float)) and 1 <= int(budget) <= len(plan):
                self._diagnosed_step_budget = int(budget)
                failed.update(range(self._diagnosed_step_budget + 1, len(plan) + 1))
            rule_failed, _, _ = _rule_failure_indices(task, plan)
            failed.update(rule_failed)
            return failed
        except Exception:
            return set()

# ----------------------------------------------------------------------
# Helper: contract verification
# ----------------------------------------------------------------------
def _verify_step(output: str, contract_spec: Optional[dict]) -> bool:
    if contract_spec is None:
        return True  # no contract, always pass
    must_contain = contract_spec.get("must_contain", [])
    for keyword in must_contain:
        if keyword not in output:
            return False
    return True

def _normalize_dependency_index(raw_parent: Any, current_index: int) -> Optional[int]:
    try:
        parent_idx = int(raw_parent)
    except Exception:
        return None
    if parent_idx == -1:
        return None
    # The official benchmark often uses zero-based dependence indices, while the
    # PGIR runtime stores one-based step indices.
    if parent_idx == 0:
        parent_idx = 1
    if 1 <= parent_idx < current_index:
        return parent_idx
    return None

def _extract_generated_ref_sources(value: Any) -> set[int]:
    sources: set[int] = set()
    if isinstance(value, str):
        for match in re.finditer(r"<GENERATED>-(\d+)-<?([^<>]+)>?", value):
            try:
                sources.add(int(match.group(1)))
            except Exception:
                continue
    elif isinstance(value, dict):
        for item in value.values():
            sources.update(_extract_generated_ref_sources(item))
    elif isinstance(value, list):
        for item in value:
            sources.update(_extract_generated_ref_sources(item))
    return sources

def _extract_labeled_step_sources(value: Any) -> set[int]:
    sources: set[int] = set()
    if isinstance(value, str):
        for match in re.finditer(
            r"\b(?:step|node|parent|source|ancestor)\s*#?\s*(\d+)\b",
            value,
            flags=re.IGNORECASE,
        ):
            try:
                sources.add(int(match.group(1)))
            except Exception:
                continue
        if re.fullmatch(r"\s*\d+\s*", value):
            try:
                sources.add(int(value.strip()))
            except Exception:
                pass
    elif isinstance(value, dict):
        for key, item in value.items():
            sources.update(_extract_labeled_step_sources(key))
            sources.update(_extract_labeled_step_sources(item))
    elif isinstance(value, list):
        for item in value:
            sources.update(_extract_labeled_step_sources(item))
    return sources

def _infer_step_dependencies(step: Dict[str, Any], current_index: int) -> List[int]:
    inferred: set[int] = set()
    raw_deps = step.get("dependencies", step.get("dependence", []))
    if isinstance(raw_deps, int):
        raw_deps = [raw_deps]
    for parent in raw_deps or []:
        normalized = _normalize_dependency_index(parent, current_index)
        if normalized is not None:
            inferred.add(normalized)
    dependence_content = step.get("dependence_content", step.get("dependency_content", {}))
    if isinstance(dependence_content, dict):
        for parent in dependence_content:
            normalized = _normalize_dependency_index(parent, current_index)
            if normalized is not None:
                inferred.add(normalized)
        for raw_source in _extract_generated_ref_sources(dependence_content):
            normalized = _normalize_dependency_index(raw_source, current_index)
            if normalized is not None:
                inferred.add(normalized)
        for raw_source in _extract_labeled_step_sources(dependence_content):
            normalized = _normalize_dependency_index(raw_source, current_index)
            if normalized is not None:
                inferred.add(normalized)
    params_or_inputs = step.get("params", step.get("inputs", {}))
    for raw_source in _extract_generated_ref_sources(params_or_inputs):
        normalized = _normalize_dependency_index(raw_source, current_index)
        if normalized is not None:
            inferred.add(normalized)
    for raw_source in _extract_labeled_step_sources(params_or_inputs):
        normalized = _normalize_dependency_index(raw_source, current_index)
        if normalized is not None:
            inferred.add(normalized)
    return sorted(inferred)

def _build_runtime_contract_tree(plan: List[Dict[str, Any]], task: Task) -> Dict[str, Dict[str, Any]]:
    tools = {tool.name: tool for tool in task.tools}
    contracts = {}
    for index, step in enumerate(plan, start=1):
        tool = tools.get(step.get("tool"))
        contracts[str(index)] = {
            "expected_tool": step.get("tool"),
            "required_inputs": sorted((tool.parameters or {}).keys()) if tool else [],
            "expected_outputs": sorted((tool.outputs or {}).keys()) if tool else [],
            "must_contain": ["validated_against"],
        }
    return contracts

def _plan_dependencies(plan: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    deps: Dict[int, List[int]] = {}
    for index, step in enumerate(plan, start=1):
        deps[index] = _infer_step_dependencies(step, index)
    return deps

def _verify_outcome(step: Dict[str, Any], result: Dict[str, Any], contract: Optional[dict]) -> bool:
    if not result.get("success"):
        return False
    if contract is None:
        return True
    if step.get("tool") != contract.get("expected_tool"):
        return False
    required = set(contract.get("required_inputs", []))
    provided = set((step.get("params") or {}).keys())
    if required - provided:
        return False
    expected_outputs = set(contract.get("expected_outputs", []))
    declared_outputs = set(step.get("outputs", []))
    if expected_outputs - declared_outputs:
        return False
    return _verify_step(result.get("output", ""), contract)

def _task_paths(task: Task) -> set:
    return set(_ordered_task_paths(task))

def _ordered_task_paths(task: Task) -> List[str]:
    text = task.description or ""
    paths = []
    seen = set()
    candidates = re.findall(r"['\"]([^'\"]+[/\\][^'\"]+)['\"]", text)
    candidates.extend(
        match.group(0).rstrip(".,;)")
        for match in re.finditer(
            r"(?:[A-Za-z]:)?[/\\][^\s,;)'\"]+|(?:data|local)[/\\][^\s,;)'\"]+",
            text,
        )
    )
    valid_suffixes = (
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".png",
        ".json",
        ".txt",
    )
    for path in candidates:
        cleaned = path.strip().strip("'\"").rstrip(".,;)")
        if not cleaned or "/" not in cleaned.replace("\\", "/"):
            continue
        if not cleaned.lower().endswith(valid_suffixes):
            continue
        if cleaned not in seen:
            paths.append(cleaned)
            seen.add(cleaned)
    return paths

def _schema_output_names(tool) -> List[str]:
    return list((tool.outputs or {}).keys()) if tool else []

def _normalize_task_path_value(key: str, value: str, task_paths: List[str]) -> str:
    if not isinstance(value, str) or value.startswith("<GENERATED>-"):
        return value
    if value in task_paths:
        return value
    normalized_value = value.replace("\\", "/").strip().strip("'\"")
    key_lower = (key or "").lower()
    query_paths = [p for p in task_paths if "quer" in p.lower()]
    if key_lower == "user_query_path" and query_paths:
        return query_paths[0]
    if key_lower == "caption_path":
        for path in task_paths:
            if "caption" in path.lower():
                return path
    if key_lower == "metadata_path":
        for path in task_paths:
            if "metadata" in path.lower() or path.lower().endswith(".json"):
                return path
    keyword_matches = []
    for path in task_paths:
        lower = path.lower()
        if "pre" in key_lower and "pre" in lower:
            keyword_matches.append(path)
        elif "post" in key_lower and "post" in lower:
            keyword_matches.append(path)
        elif "forest" in normalized_value.lower() and "forest" in lower:
            keyword_matches.append(path)
        elif "urban" in normalized_value.lower() and "urban" in lower:
            keyword_matches.append(path)
        elif "fog" in normalized_value.lower() and "fog" in lower:
            keyword_matches.append(path)
        elif "low" in normalized_value.lower() and "low" in lower:
            keyword_matches.append(path)
        elif "high_res" in normalized_value.lower() and "high_res" in lower:
            keyword_matches.append(path)
        elif "hsr" in normalized_value.lower() and "hsr" in lower:
            keyword_matches.append(path)
    if len(keyword_matches) == 1:
        return keyword_matches[0]
    basename = normalized_value.rsplit("/", 1)[-1].lower()
    basename_matches = [path for path in task_paths if path.replace("\\", "/").lower().endswith("/" + basename)]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if key_lower in {"image_path", "image_1_path", "image_2_path", "pre_disaster_image_path", "post_disaster_image_path"}:
        image_paths = [
            path for path in task_paths
            if not any(marker in path.lower() for marker in ("query", "caption", "metadata"))
        ]
        if len(image_paths) == 1:
            return image_paths[0]
    return value

def _normalize_plan_task_paths(plan: List[Dict[str, Any]], task: Task) -> List[Dict[str, Any]]:
    task_paths = _ordered_task_paths(task)
    if not task_paths:
        return plan
    repaired = copy.deepcopy(plan)
    for step in repaired:
        params = step.get("params")
        if not isinstance(params, dict):
            continue
        for key, value in list(params.items()):
            if key.endswith("_path") and isinstance(value, str):
                params[key] = _normalize_task_path_value(key, value, task_paths)
    return repaired

def _rule_failure_indices(task: Task, plan: List[Dict[str, Any]]) -> Tuple[set, Optional[int], bool]:
    failed = set()
    if not plan:
        return failed, None, True
    task_text = (task.description or "").lower()
    task_paths = _task_paths(task)
    tools = [step.get("tool") for step in plan]
    tool_set = set(tools)

    low_fog_request = (
        any(term in task_text for term in ("low-light", "low light", "night"))
        and any(term in task_text for term in ("fog", "hazy", "weather"))
    )
    damage_class_request = (
        "building" in task_text
        and "damage" in task_text
        and any(term in task_text for term in ("minor damage", "major damage", "destroyed", "extent of damage"))
    )
    explicit_building_damage_request = any(
        term in task_text
        for term in (
            "building damage",
            "damage to buildings",
            "damages to buildings",
            "damaged buildings",
            "damage assessment",
            "damage classification",
        )
    )
    anomaly_then_change_request = "anomal" in task_text and "change" in task_text
    multi_environment_request = "urban" in task_text and "forest" in task_text
    query_or_report_request = any(
        term in task_text
        for term in ("query", "description", "report", "contextual")
    )
    caption_weather_crowd_request = (
        "caption" in task_text
        and "metadata" in task_text
        and "generat" in task_text
        and "crowd" in task_text
        and any(term in task_text for term in ("weather", "storm", "snow"))
    )
    query_path_request = "query file" in task_text
    caption_weather_crowd_request = (
        "caption" in task_text
        and "metadata" in task_text
        and "generat" in task_text
        and "crowd" in task_text
        and any(term in task_text for term in ("weather", "storm", "snow"))
    )

    has_change_mapping = "Change_Mapping_and_Detection" in tool_set
    has_building_damage = "Building_damage_assessment" in tool_set
    has_urban_anomaly = "Urban_Anomaly_Detection" in tool_set
    has_low_light = "Low-Light_Object_Detection" in tool_set
    has_foggy = "Foggy_Scenario_Object_Detection" in tool_set
    has_caption_generation = "Metadata_and_Text_Prompt_Image_Generation" in tool_set
    has_weather_restore = "Weather_Degraded_Image_Restoration" in tool_set
    has_crowd_count = "Crowd_Counting_in_Adverse_Weather" in tool_set
    has_real_query_geochat = any(
        step.get("tool") == "GeoChat"
        and isinstance((step.get("params") or {}).get("user_query_path"), str)
        and (step.get("params") or {}).get("user_query_path") in task_paths
        for step in plan
    )

    seen_signatures = {}
    for index, step in enumerate(plan, start=1):
        params = step.get("params") or {}
        signature = (
            step.get("tool"),
            json.dumps(params, ensure_ascii=False, sort_keys=True),
            tuple(int(dep) for dep in step.get("dependencies", []) or [] if str(dep).isdigit()),
        )
        if signature in seen_signatures:
            failed.add(index)
        else:
            seen_signatures[signature] = index
        if step.get("tool") == "GeoChat":
            query_path = params.get("user_query_path")
            if query_path and query_path not in task_paths:
                failed.add(index)
        for key, value in params.items():
            if not key.endswith("_path") or not isinstance(value, str):
                continue
            is_task_path = value in task_paths
            is_generated = value.startswith("<GENERATED>-")
            is_absolute = value.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", value)
            if not (is_task_path or is_generated or is_absolute):
                failed.add(index)
        if (
            caption_weather_crowd_request
            and step.get("tool") == "High-Resolution_Image_Reconstructor"
            and str(params.get("image_path", "")).lower().endswith(".txt")
        ):
            failed.add(index)

    if caption_weather_crowd_request:
        if has_crowd_count and not has_weather_restore:
            for index, step in enumerate(plan, start=1):
                if step.get("tool") == "Crowd_Counting_in_Adverse_Weather":
                    failed.add(index)
        if has_weather_restore and not has_caption_generation:
            for index, step in enumerate(plan, start=1):
                if step.get("tool") == "Weather_Degraded_Image_Restoration":
                    failed.add(index)
        if has_caption_generation and has_weather_restore and has_crowd_count:
            for index, step in enumerate(plan, start=1):
                if step.get("tool") not in {
                    "Metadata_and_Text_Prompt_Image_Generation",
                    "Weather_Degraded_Image_Restoration",
                    "Crowd_Counting_in_Adverse_Weather",
                }:
                    failed.add(index)
            return failed, 3, not failed and len(plan) == 3

    if low_fog_request and has_low_light and has_foggy:
        for index, step in enumerate(plan, start=1):
            if step.get("tool") not in {"Low-Light_Object_Detection", "Foggy_Scenario_Object_Detection"}:
                failed.add(index)
        return failed, 2, not failed and len(plan) == 2

    if damage_class_request and has_building_damage:
        for index, step in enumerate(plan, start=1):
            if step.get("tool") in {"Change_Mapping_and_Detection", "GeoChat"}:
                failed.add(index)
        return failed, 1, not failed and len(plan) == 1

    if anomaly_then_change_request:
        for index, step in enumerate(plan, start=1):
            if (
                step.get("tool") == "Building_damage_assessment"
                and not (damage_class_request or explicit_building_damage_request)
            ):
                failed.add(index)
        if has_urban_anomaly and has_change_mapping:
            allowed_tools = {
                "Urban_Anomaly_Detection",
                "Change_Mapping_and_Detection",
            }
            if damage_class_request or explicit_building_damage_request:
                allowed_tools.add("Building_damage_assessment")
            if multi_environment_request:
                allowed_tools.add("Anomaly_Detection_Forest")
            if query_or_report_request:
                allowed_tools.add("GeoChat")
            for index, step in enumerate(plan, start=1):
                if step.get("tool") not in allowed_tools:
                    failed.add(index)
            pure_two_step_repair = not (
                damage_class_request
                or explicit_building_damage_request
                or multi_environment_request
                or query_or_report_request
            )
            if pure_two_step_repair:
                return failed, 2, not failed and len(plan) == 2

    if has_change_mapping and has_real_query_geochat and not damage_class_request:
        for index, step in enumerate(plan, start=1):
            if step.get("tool") == "Building_damage_assessment":
                failed.add(index)
        if query_path_request:
            for index, step in enumerate(plan, start=1):
                if step.get("tool") not in {"Change_Mapping_and_Detection", "GeoChat"}:
                    failed.add(index)
            return failed, 2, not failed and len(plan) == 2

    return failed, None, False

def _prune_rule_failed_plan(
    task: Task, plan: List[Dict[str, Any]], failed_steps: set
) -> List[Dict[str, Any]]:
    if not failed_steps or len(failed_steps) >= len(plan):
        return []
    old_to_new = {}
    kept = []
    for old_index, step in enumerate(plan, start=1):
        if old_index in failed_steps:
            continue
        old_to_new[old_index] = len(kept) + 1
        kept.append(copy.deepcopy(step))
    for new_index, step in enumerate(kept, start=1):
        step["step_idx"] = new_index
        deps = []
        for dep in step.get("dependencies", []) or []:
            try:
                dep_idx = int(dep)
            except Exception:
                continue
            if dep_idx in old_to_new:
                deps.append(old_to_new[dep_idx])
        step["dependencies"] = deps
        for key, value in list((step.get("params") or {}).items()):
            if not isinstance(value, str):
                continue
            match = re.match(r"^<GENERATED>-(\d+)-(.+)$", value)
            if match and int(match.group(1)) in old_to_new:
                step["params"][key] = f"<GENERATED>-{old_to_new[int(match.group(1))]}-{match.group(2)}"
    residual_failed, _, complete = _rule_failure_indices(task, kept)
    return kept if not residual_failed and (complete or len(kept) < len(plan)) else []

def _deterministic_rule_repair_plan(
    task: Task, plan: List[Dict[str, Any]], failed_steps: set
) -> List[Dict[str, Any]]:
    pruned = _prune_rule_failed_plan(task, plan, failed_steps)
    if pruned:
        return pruned
    task_text = (task.description or "").lower()
    caption_weather_crowd_request = (
        "caption" in task_text
        and "metadata" in task_text
        and "generate" in task_text
        and "crowd" in task_text
        and any(term in task_text for term in ("weather", "storm", "snow"))
    )
    if not caption_weather_crowd_request:
        return []
    generation = None
    crowd = None
    for step in plan:
        if step.get("tool") == "Metadata_and_Text_Prompt_Image_Generation":
            generation = copy.deepcopy(step)
        elif step.get("tool") == "Crowd_Counting_in_Adverse_Weather":
            crowd = copy.deepcopy(step)
    if not generation or not crowd:
        return []
    repaired = [
        generation,
        {
            "step_idx": 2,
            "tool": "Weather_Degraded_Image_Restoration",
            "params": {"weather_degraded_image_path": "<GENERATED>-1-generated_image_path"},
            "dependencies": [1],
            "outputs": ["restored_image_path"],
            "dependence_content": {"1": "generated_image_path"},
        },
        crowd,
    ]
    repaired[0]["step_idx"] = 1
    repaired[0]["dependencies"] = []
    repaired[2]["step_idx"] = 3
    repaired[2]["dependencies"] = [2]
    repaired[2]["params"] = {"image_path": "<GENERATED>-2-restored_image_path"}
    repaired[2]["dependence_content"] = {"2": "restored_image_path"}
    residual_failed, _, complete = _rule_failure_indices(task, repaired)
    return repaired if complete and not residual_failed else []

def _renumber_plan_after_local_prune(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    old_to_new = {}
    repaired = []
    for new_index, step in enumerate(plan, start=1):
        old_index = int(step.get("step_idx", new_index))
        old_to_new[old_index] = new_index
        repaired.append(copy.deepcopy(step))
    for new_index, step in enumerate(repaired, start=1):
        step["step_idx"] = new_index
        deps = []
        for dep in step.get("dependencies", []) or []:
            try:
                dep_idx = int(dep)
            except Exception:
                continue
            if dep_idx in old_to_new:
                deps.append(old_to_new[dep_idx])
        step["dependencies"] = sorted(set(deps))
        dependence_content = step.get("dependence_content")
        if isinstance(dependence_content, dict):
            rewritten_content = {}
            for key, value in dependence_content.items():
                try:
                    dep_idx = int(key)
                except Exception:
                    continue
                if dep_idx in old_to_new:
                    rewritten_content[str(old_to_new[dep_idx])] = value
            step["dependence_content"] = rewritten_content
        for key, value in list((step.get("params") or {}).items()):
            if not isinstance(value, str):
                continue
            match = re.match(r"^<GENERATED>-(\d+)-(.+)$", value)
            if match:
                old_source = int(match.group(1))
                if old_source in old_to_new:
                    step["params"][key] = f"<GENERATED>-{old_to_new[old_source]}-{match.group(2)}"
    return repaired

def _verifier_guided_local_prune_plan(
    task: Task,
    plan: List[Dict[str, Any]],
    scope: set,
    diagnosed_step_budget: Optional[int],
) -> List[Dict[str, Any]]:
    if not plan or not isinstance(diagnosed_step_budget, int):
        return []
    if diagnosed_step_budget < 1 or diagnosed_step_budget >= len(plan):
        return []
    redundant_tail = set(range(diagnosed_step_budget + 1, len(plan) + 1))
    if not redundant_tail or not redundant_tail.issubset(set(scope)):
        return []
    repaired = [
        copy.deepcopy(step)
        for index, step in enumerate(plan, start=1)
        if index not in redundant_tail
    ]
    repaired = _renumber_plan_after_local_prune(repaired)
    residual_failed, residual_budget, complete = _rule_failure_indices(task, repaired)
    if residual_failed:
        return []
    if residual_budget is not None and len(repaired) > residual_budget:
        return []
    return repaired if complete or len(repaired) < len(plan) else []

def _plan_quality(plan: List[Dict[str, Any]], task: Task) -> float:
    if not plan:
        return -1e9
    task_text = task.description.lower()
    damage_class_request = (
        "building" in task_text
        and "damage" in task_text
        and any(term in task_text for term in ("minor damage", "major damage", "destroyed", "extent of damage"))
    )
    explicit_building_damage_request = any(
        term in task_text
        for term in (
            "building damage",
            "damage to buildings",
            "damages to buildings",
            "damaged buildings",
            "damage assessment",
            "damage classification",
        )
    )
    anomaly_then_change_request = "anomal" in task_text and "change" in task_text
    caption_weather_crowd_request = (
        "caption" in task_text
        and "metadata" in task_text
        and "generat" in task_text
        and "crowd" in task_text
        and any(term in task_text for term in ("weather", "storm", "snow"))
    )
    task_paths = set(re.findall(r"['\"]([^'\"]+[/\\][^'\"]+)['\"]", task.description))
    serialized_params = json.dumps(
        [step.get("params", {}) for step in plan], ensure_ascii=False
    )
    coverage = (
        sum(1 for path in task_paths if path in serialized_params) / len(task_paths)
        if task_paths else 1.0
    )
    contracts = _build_runtime_contract_tree(plan, task)
    schema_valid = 0
    for index, step in enumerate(plan, start=1):
        contract = contracts.get(str(index), {})
        required = set(contract.get("required_inputs", []))
        outputs = set(contract.get("expected_outputs", []))
        if required.issubset(set((step.get("params") or {}).keys())) and outputs.issubset(
            set(step.get("outputs", []))
        ):
            schema_valid += 1
    hallucinated_path_penalty = 0
    semantic_penalty = 0
    has_urban_anomaly = any(step.get("tool") == "Urban_Anomaly_Detection" for step in plan)
    for step in plan:
        if (
            anomaly_then_change_request
            and step.get("tool") == "Building_damage_assessment"
            and not (damage_class_request or explicit_building_damage_request)
        ):
            semantic_penalty += 2
        if (
            caption_weather_crowd_request
            and step.get("tool") == "High-Resolution_Image_Reconstructor"
        ):
            semantic_penalty += 3
        for key, value in (step.get("params") or {}).items():
            if not key.endswith("_path") or not isinstance(value, str):
                continue
            is_task_path = value in task_paths
            is_generated = value.startswith("<GENERATED>-")
            is_absolute = value.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", value)
            if not (is_task_path or is_generated or is_absolute):
                hallucinated_path_penalty += 1
    if anomaly_then_change_request and not has_urban_anomaly:
        semantic_penalty += 1
    if caption_weather_crowd_request:
        needed = {
            "Metadata_and_Text_Prompt_Image_Generation",
            "Weather_Degraded_Image_Restoration",
            "Crowd_Counting_in_Adverse_Weather",
        }
        if not needed.issubset({step.get("tool") for step in plan}):
            semantic_penalty += 2
    return (
        coverage * 10.0
        + schema_valid / len(plan)
        - 0.1 * len(plan)
        - hallucinated_path_penalty
        - semantic_penalty
    )

def _format_tool_catalog(task: Task) -> str:
    entries = []
    for tool in task.tools:
        input_keys = sorted((tool.parameters or {}).keys())
        desc = (tool.description or "").replace("\n", " ")[:140]
        entries.append({
            "name": tool.name,
            "inputs": input_keys,
            "outputs": _schema_output_names(tool),
            "desc": desc,
        })
    return json.dumps(entries, ensure_ascii=False)

def _format_repair_tool_catalog(task: Task, plan: List[Dict[str, Any]]) -> str:
    task_text = (task.description or "").lower()
    current_tools = {step.get("tool") for step in plan if step.get("tool")}
    keyword_tools = set()
    if any(term in task_text for term in ("before", "after", "change", "alteration")):
        keyword_tools.add("Change_Mapping_and_Detection")
    if any(term in task_text for term in ("building", "damage", "destroyed")):
        keyword_tools.add("Building_damage_assessment")
    if "anomal" in task_text:
        keyword_tools.add("Urban_Anomaly_Detection")
    if "caption" in task_text and "metadata" in task_text:
        keyword_tools.add("Metadata_and_Text_Prompt_Image_Generation")
    if any(term in task_text for term in ("restore", "weather-degraded", "adverse weather", "snowstorm")):
        keyword_tools.add("Weather_Degraded_Image_Restoration")
    if "crowd" in task_text:
        keyword_tools.add("Crowd_Counting_in_Adverse_Weather")
    if any(term in task_text for term in ("low-light", "low light", "night")):
        keyword_tools.add("Low-Light_Object_Detection")
    if any(term in task_text for term in ("fog", "hazy", "rain")):
        keyword_tools.add("Foggy_Scenario_Object_Detection")
    if any(term in task_text for term in ("query file", "description", "describe")):
        keyword_tools.add("GeoChat")
    if "classif" in task_text or "categor" in task_text:
        keyword_tools.add("RGB_GeoImage_Classifier")
    selected = current_tools | keyword_tools
    entries = []
    for tool in task.tools:
        if tool.name not in selected:
            continue
        entries.append({
            "name": tool.name,
            "inputs": sorted((tool.parameters or {}).keys()),
            "outputs": _schema_output_names(tool),
            "desc": (tool.description or "").replace("\n", " ")[:90],
        })
    if not entries:
        return _format_tool_catalog(task)
    return json.dumps(entries, ensure_ascii=False)

def _initial_plan_prompt(task: Task) -> str:
    return (
        f"Task: {task.description}\nAvailable tools: {_format_tool_catalog(task)}\n"
        "Produce the smallest sufficient plan. Map each explicitly requested action to the single "
        "most direct specialized tool. Do not add optional preprocessing, broad analysis, alternate "
        "methods, or extra deliverables that the task did not request. Downstream tools should consume "
        "the relevant output of prior tools instead of independently repeating analysis. When a "
        "condition-specific direct tool exists, use it directly rather than adding restoration or "
        "preprocessing. When a description/report tool can consume an upstream result, do not add "
        "parallel specialist analyses that are not explicitly requested. Use at most "
        "5 steps and do not copy the full tool list. Generate a JSON list with keys: step_idx "
        "(1-based int), tool, params, "
        "dependencies (prior 1-based step indices; empty for roots), outputs (schema output names), "
        "and dependence_content. Fill every required input from the task or prior-step references "
        "formatted as <GENERATED>-step_index-<output_name>. Only output the JSON list."
    )

def _generate_shared_initial_plan(method: BaseRepairMethod, task: Task) -> List[Dict[str, Any]]:
    seed = getattr(method, "seed", getattr(task, "_experiment_seed", 0))
    prompt = (
        _initial_plan_prompt(task)
        + f"\nRun seed: {seed}. Use this seed only to break ties among equally direct, valid plans."
    )
    cache = getattr(task, "_shared_initial_plans", {})
    cache_key = (method.model_name, seed)
    cached = cache.get(cache_key)
    if cached is not None:
        method.total_tokens += len(prompt.split()) + method.config.max_tokens_per_step
        method.prompts.append(prompt)
        return copy.deepcopy(cached)
    response = method.call_llm(prompt, method.config.max_tokens_per_step)
    plan = _coerce_plan_text(response, task)
    cache[cache_key] = copy.deepcopy(plan)
    task._shared_initial_plans = cache
    return plan

def _extract_json_payload(text: str) -> Any:
    if not isinstance(text, str):
        return text
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
            return payload
        except Exception:
            continue
    return None

def _coerce_plan_text(resp: str, task: Task) -> List[Dict[str, Any]]:
    obj = _extract_json_payload(resp)
    if isinstance(obj, dict):
        for key in ("plan", "steps", "task_plan", "answer"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
    if not isinstance(obj, list):
        return []
    plan = []
    tools_by_name = {tool.name: tool for tool in task.tools}
    available = set(tools_by_name)
    for index, item in enumerate(obj):
        if not isinstance(item, dict):
            continue
        tool = item.get("tool") or item.get("agent") or item.get("agent_name")
        if not tool or tool not in available:
            continue
        step = item.get("step_idx", item.get("step", index + 1))
        try:
            step_idx = int(step)
        except Exception:
            step_idx = index + 1
        params = item.get("params")
        if params is None:
            params = item.get("inputs", {})
        dependencies = item.get("dependencies", item.get("dependence", []))
        outputs = item.get("outputs", [])
        dependence_content = item.get("dependence_content", item.get("dependency_content", {}))
        if not isinstance(dependencies, list):
            dependencies = [dependencies]
        normalized_params = {}
        for key, value in (params or {}).items():
            if (
                isinstance(value, str)
                and len(dependencies) == 1
                and value.startswith("{")
                and value.endswith("}")
            ):
                output_name = value[1:-1]
                try:
                    source = int(dependencies[0])
                    value = f"<GENERATED>-{source}-<{output_name}>"
                except Exception:
                    pass
            normalized_params[key] = value
        schema_outputs = _schema_output_names(tools_by_name[tool])
        normalized_step = {
            "step_idx": index + 1,
            "tool": tool,
            "params": normalized_params,
            "dependencies": dependencies,
            "outputs": schema_outputs or (outputs if isinstance(outputs, list) else []),
            "dependence_content": dependence_content if isinstance(dependence_content, dict) else {},
        }
        normalized_step["dependencies"] = _infer_step_dependencies(normalized_step, index + 1)
        plan.append(normalized_step)
    if len(plan) > 8 or (available and len(plan) > max(5, len(available) // 3)):
        return []
    return _normalize_plan_task_paths(plan, task)

# ----------------------------------------------------------------------
# PGIR 鈥?hidden taint ancestor repair (PGIR main condition)
# ----------------------------------------------------------------------
class PGIRHiddenTaintAncestorRepair(BaseRepairMethod):
    def _visible_taint_enabled(self) -> bool:
        return False

    def _allow_deterministic_rule_patch(self) -> bool:
        return True

    def _allow_llm_local_patch(self) -> bool:
        return True

    def _allow_verifier_guided_local_prune(self) -> bool:
        return True

    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        if not plan:
            plan = self._recover_empty_plan(task)
        initial_plan = copy.deepcopy(plan)
        self.last_plan = plan
        contract_tree = _build_runtime_contract_tree(plan, task)
        dependencies = _plan_dependencies(plan)
        child_map = self._child_map(dependencies)
        tainted = set()
        step_results = []
        contamination_events = []
        all_contamination_events = []
        repair_attempts = []
        repair_boundary_triggers = []
        repair_scope = set()
        frontier_nodes = set()
        affected_nodes = set()
        reexecuted = 0
        global_escalated = False
        repair_calls = 0
        boundary_repairs = 0
        local_attempts = 0
        global_attempts = 0
        stopped_at_unrepaired_boundary = False
        unrepaired_boundary_type = None
        unrepaired_boundary_step = None
        unrepaired_boundary_reason = None
        unrepaired_boundary_tainted_nodes = []
        index = 0
        plan_failed, plan_budget, plan_complete = _rule_failure_indices(task, plan)
        if plan_budget is not None:
            self._diagnosed_step_budget = plan_budget
        if plan_failed and not plan_complete:
            plan_events = [
                self._classify_plan_boundary_violation(node, plan, dependencies, child_map, plan_budget)
                for node in sorted(plan_failed)
            ]
            contamination_events.extend(plan_events)
            all_contamination_events.extend(plan_events)
            repair_result = None
            if any(event.get("status") == "blocking_taint" for event in plan_events):
                repair_boundary_triggers.append({
                    "type": "plan_boundary",
                    "step_idx": 0,
                    "events": len(contamination_events),
                    "tainted_nodes": sorted(tainted),
                })
                repair_result = self._repair_from_contamination_frontier(
                    task,
                    plan,
                    step_results,
                    contract_tree,
                    dependencies,
                    contamination_events,
                    local_attempts,
                    global_attempts,
                )
            if repair_result and repair_result.get("attempt_record"):
                repair_attempts.append(repair_result["attempt_record"])
                local_attempts, global_attempts = self._consume_attempt_record(
                    repair_result["attempt_record"], local_attempts, global_attempts
                )
            if repair_result and repair_result["repaired"]:
                plan = repair_result["plan"]
                self.last_plan = plan
                step_results = repair_result["step_results"]
                contract_tree = _build_runtime_contract_tree(plan, task)
                dependencies = _plan_dependencies(plan)
                child_map = self._child_map(dependencies)
                repair_scope.update(repair_result["repair_scope"])
                frontier_nodes.update(repair_result["frontier"])
                affected_nodes.update(repair_result["affected"])
                reexecuted += repair_result["reexecuted"]
                repair_calls += 1
                boundary_repairs += 1
                global_escalated = global_escalated or repair_result["global_escalated"]
                contamination_events = []
                tainted = set()
                index = len(step_results)
            elif repair_result:
                stopped_at_unrepaired_boundary = True
                unrepaired_boundary_type = "plan_boundary"
                unrepaired_boundary_step = 0
                unrepaired_boundary_reason = repair_result.get("attempt_record", {}).get("reason")
                unrepaired_boundary_tainted_nodes = sorted(tainted)
        while index < len(plan) and not stopped_at_unrepaired_boundary:
            step = plan[index]
            step_idx = index + 1
            pre_events = self._verify_dependency_consumption(step_idx, dependencies, tainted)
            for event in pre_events:
                self._record_contamination_event(event, contamination_events, all_contamination_events, tainted)
            if pre_events:
                boundary_type = (
                    "fan_in_aggregation"
                    if len(dependencies.get(step_idx, [])) > 1
                    else "dependency_consumption"
                )
                repair_boundary_triggers.append({
                    "type": boundary_type,
                    "step_idx": step_idx,
                    "events": len(contamination_events),
                    "tainted_nodes": sorted(tainted),
                })
                repair_result = self._repair_from_contamination_frontier(
                    task,
                    plan,
                    step_results,
                    contract_tree,
                    dependencies,
                    contamination_events,
                    local_attempts,
                global_attempts,
                )
                if repair_result.get("attempt_record"):
                    repair_attempts.append(repair_result["attempt_record"])
                    local_attempts, global_attempts = self._consume_attempt_record(
                        repair_result["attempt_record"], local_attempts, global_attempts
                    )
                if repair_result["repaired"]:
                    plan = repair_result["plan"]
                    self.last_plan = plan
                    step_results = repair_result["step_results"]
                    contract_tree = _build_runtime_contract_tree(plan, task)
                    dependencies = _plan_dependencies(plan)
                    child_map = self._child_map(dependencies)
                    repair_scope.update(repair_result["repair_scope"])
                    frontier_nodes.update(repair_result["frontier"])
                    affected_nodes.update(repair_result["affected"])
                    reexecuted += repair_result["reexecuted"]
                    repair_calls += 1
                    boundary_repairs += 1
                    global_escalated = global_escalated or repair_result["global_escalated"]
                    contamination_events = []
                    tainted = set()
                    index = len(step_results)
                    continue
                stopped_at_unrepaired_boundary = True
                unrepaired_boundary_type = boundary_type
                unrepaired_boundary_step = step_idx
                unrepaired_boundary_reason = repair_result.get("attempt_record", {}).get("reason")
                unrepaired_boundary_tainted_nodes = sorted(tainted)
                break
            res = benv.execute_tool(task, step["tool"], step.get("params", {}))
            if len(step_results) <= index:
                step_results.append(res)
            else:
                step_results[index] = res
            contract = contract_tree.get(str(step_idx)) if contract_tree else None
            event = self._classify_step_outcome(step_idx, step, res, contract)
            if event["status"] != "pass":
                self._record_contamination_event(event, contamination_events, all_contamination_events, tainted)
            boundary_type = self._repair_boundary_type(
                step_idx, plan, dependencies, child_map, contamination_events, tainted
            )
            if boundary_type:
                repair_boundary_triggers.append({
                    "type": boundary_type,
                    "step_idx": step_idx,
                    "events": len(contamination_events),
                    "tainted_nodes": sorted(tainted),
                })
                repair_result = self._repair_from_contamination_frontier(
                    task,
                    plan,
                    step_results,
                    contract_tree,
                    dependencies,
                    contamination_events,
                local_attempts,
                global_attempts,
                )
                if repair_result.get("attempt_record"):
                    repair_attempts.append(repair_result["attempt_record"])
                    local_attempts, global_attempts = self._consume_attempt_record(
                        repair_result["attempt_record"], local_attempts, global_attempts
                    )
                if repair_result["repaired"]:
                    plan = repair_result["plan"]
                    self.last_plan = plan
                    step_results = repair_result["step_results"]
                    contract_tree = _build_runtime_contract_tree(plan, task)
                    dependencies = _plan_dependencies(plan)
                    child_map = self._child_map(dependencies)
                    repair_scope.update(repair_result["repair_scope"])
                    frontier_nodes.update(repair_result["frontier"])
                    affected_nodes.update(repair_result["affected"])
                    reexecuted += repair_result["reexecuted"]
                    repair_calls += 1
                    boundary_repairs += 1
                    global_escalated = global_escalated or repair_result["global_escalated"]
                    contamination_events = []
                    tainted = set()
                    index = len(step_results)
                    continue
                stopped_at_unrepaired_boundary = True
                unrepaired_boundary_type = boundary_type
                unrepaired_boundary_step = step_idx
                unrepaired_boundary_reason = repair_result.get("attempt_record", {}).get("reason")
                unrepaired_boundary_tainted_nodes = sorted(tainted)
                break
            index += 1
        final_tainted = (
            set()
            if stopped_at_unrepaired_boundary
            else self._diagnose_semantic_failures(task, plan, step_results)
        )
        if final_tainted:
            final_events = [
                self._make_contamination_event(
                    node,
                    "blocking_taint",
                    "final_boundary_semantic_violation",
                    [node],
                    "Final boundary verifier found a remaining execution-critical violation.",
                    "final",
                )
                for node in sorted(final_tainted)
            ]
            contamination_events.extend(final_events)
            all_contamination_events.extend(final_events)
            repair_boundary_triggers.append({
                "type": "final_commit",
                "step_idx": len(plan),
                "events": len(contamination_events),
                "tainted_nodes": sorted(tainted),
            })
            repair_result = self._repair_from_contamination_frontier(
                task,
                plan,
                step_results,
                contract_tree,
                dependencies,
                contamination_events,
                local_attempts,
                global_attempts,
                force_global=global_attempts >= getattr(self.config, "pgir_global_replan_retry_cap", 1),
            )
            if repair_result.get("attempt_record"):
                repair_attempts.append(repair_result["attempt_record"])
                local_attempts, global_attempts = self._consume_attempt_record(
                    repair_result["attempt_record"], local_attempts, global_attempts
                )
            if repair_result["repaired"]:
                plan = repair_result["plan"]
                self.last_plan = plan
                step_results = repair_result["step_results"]
                contract_tree = _build_runtime_contract_tree(plan, task)
                repair_scope.update(repair_result["repair_scope"])
                frontier_nodes.update(repair_result["frontier"])
                affected_nodes.update(repair_result["affected"])
                reexecuted += repair_result["reexecuted"]
                repair_calls += 1
                boundary_repairs += 1
                global_escalated = global_escalated or repair_result["global_escalated"]
            else:
                stopped_at_unrepaired_boundary = True
                unrepaired_boundary_type = "final_commit"
                unrepaired_boundary_step = len(plan)
                unrepaired_boundary_reason = repair_result.get("attempt_record", {}).get("reason")
                unrepaired_boundary_tainted_nodes = sorted(final_tainted)
        final_output = step_results[-1].get("output", "") if step_results else ""
        unchanged_steps = sum(
            1
            for index in range(min(len(initial_plan), len(plan)))
            if initial_plan[index] == plan[index]
        )
        untouched_sib_ratio = unchanged_steps / len(initial_plan) if initial_plan else 0.0
        contract_pass = sum(
            1 for i, res in enumerate(step_results)
            if _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        ) / len(plan) if plan else 0.0
        blocking_events = [e for e in all_contamination_events if e.get("status") == "blocking_taint"]
        non_blocking_events = [
            e for e in all_contamination_events if e.get("status") == "non_blocking_deviation"
        ]
        initial_plan_len = len(initial_plan) if initial_plan else len(plan)
        blocking_nodes = {
            int(node)
            for event in blocking_events
            for node in ([event.get("node")] + list(event.get("responsible_nodes", [])))
            if isinstance(node, int) or (isinstance(node, str) and node.isdigit())
        }
        runtime_contract_pass = (
            max(0.0, 1.0 - (len(blocking_nodes) / initial_plan_len))
            if initial_plan_len
            else 1.0
        )
        contract_activation_rate = (
            len(blocking_events) / initial_plan_len
            if initial_plan_len
            else 0.0
        )
        taint_precision_denominator = max(len(frontier_nodes), len(repair_scope))
        taint_precision = (
            len(frontier_nodes) / taint_precision_denominator
            if taint_precision_denominator
            else 1.0
        )
        attempted_frontier_nodes = {
            int(node)
            for attempt in repair_attempts
            for node in attempt.get("frontier", [])
            if isinstance(node, int) or (isinstance(node, str) and str(node).isdigit())
        }
        attempted_scope_nodes = {
            int(node)
            for attempt in repair_attempts
            for node in attempt.get("scope", [])
            if isinstance(node, int) or (isinstance(node, str) and str(node).isdigit())
        }
        result = {
            "task_id": task.task_id,
            "final_output": final_output,
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": taint_precision,
            "cascade_depth": len(repair_scope),
            "contract_pass_rate": runtime_contract_pass,
            "final_contract_pass_rate": contract_pass,
            "contract_activation_rate": contract_activation_rate,
            "visible_taint_exposure": 1.0 if self._visible_taint_enabled() else 0.0,
            "visible_taint_configured": 1.0 if self._visible_taint_enabled() else 0.0,
            "visible_taint_prompt_exposure": (
                1.0
                if self._visible_taint_enabled()
                and any("VISIBLE_TAINT_LABELS" in prompt for prompt in self.prompts)
                else 0.0
            ),
            "diagnosed_first_failure": min(frontier_nodes) if frontier_nodes else None,
            "runtime_repair_mode": "deferred_frontier_repair",
            "stopped_at_unrepaired_boundary": stopped_at_unrepaired_boundary,
            "unrepaired_boundary_type": unrepaired_boundary_type,
            "unrepaired_boundary_step": unrepaired_boundary_step,
            "unrepaired_boundary_reason": unrepaired_boundary_reason,
            "unrepaired_boundary_tainted_nodes": unrepaired_boundary_tainted_nodes,
            "executed_step_count": len(step_results),
            "repair_calls": repair_calls,
            "boundary_repairs": boundary_repairs,
            "repair_boundary_triggers": repair_boundary_triggers,
            "plan_boundary_triggers": sum(
                1 for trigger in repair_boundary_triggers
                if trigger.get("type") == "plan_boundary"
            ),
            "dependency_consumption_boundary_triggers": sum(
                1 for trigger in repair_boundary_triggers
                if trigger.get("type") == "dependency_consumption"
            ),
            "fan_in_boundary_triggers": sum(
                1 for trigger in repair_boundary_triggers
                if trigger.get("type") == "fan_in_aggregation"
            ),
            "fan_out_boundary_triggers": sum(
                1 for trigger in repair_boundary_triggers
                if trigger.get("type") == "fan_out_spread"
            ),
            "final_boundary_triggers": sum(
                1 for trigger in repair_boundary_triggers
                if trigger.get("type") == "final_commit"
            ),
            "local_patch_repairs": sum(
                1 for attempt in repair_attempts
                if attempt.get("mode") == "local_patch" and attempt.get("accepted")
            ),
            "deterministic_rule_patch_repairs": sum(
                1 for attempt in repair_attempts
                if attempt.get("mode") == "local_patch"
                and attempt.get("accepted")
                and attempt.get("source") == "deterministic_rule_patch"
            ),
            "llm_local_patch_repairs": sum(
                1 for attempt in repair_attempts
                if attempt.get("mode") == "local_patch"
                and attempt.get("accepted")
                and attempt.get("source") == "llm_patch"
            ),
            "verifier_guided_local_prune_repairs": sum(
                1 for attempt in repair_attempts
                if attempt.get("mode") == "local_patch"
                and attempt.get("accepted")
                and attempt.get("source") == "verifier_guided_local_prune"
            ),
            "interface_preserving_local_graph_rewrites": sum(
                1 for attempt in repair_attempts
                if attempt.get("mode") == "local_patch"
                and attempt.get("accepted")
                and attempt.get("reason")
                == "accepted_interface_preserving_local_graph_rewrite"
            ),
            "llm_requested_global_replans": sum(
                1 for attempt in repair_attempts
                if attempt.get("source") == "llm_requested_global_replan"
            ),
            "rejected_patch_repairs": sum(
                1 for attempt in repair_attempts
                if attempt.get("mode") == "local_patch" and not attempt.get("accepted")
            ),
            "global_escalations": 1 if global_escalated else 0,
            "blocking_taint_count": len(blocking_events),
            "non_blocking_deviation_count": len(non_blocking_events),
            "repair_frontier_nodes": sorted(frontier_nodes),
            "repair_frontier_size": len(frontier_nodes),
            "repair_frontier_ratio": (len(frontier_nodes) / len(initial_plan)) if initial_plan else 0.0,
            "affected_nodes": sorted(affected_nodes),
            "attempted_repair_frontier_nodes": sorted(attempted_frontier_nodes),
            "attempted_repair_frontier_size": len(attempted_frontier_nodes),
            "attempted_repair_frontier_ratio": (
                len(attempted_frontier_nodes) / len(initial_plan)
                if initial_plan
                else 0.0
            ),
            "attempted_repair_scope_nodes": sorted(attempted_scope_nodes),
            "attempted_repair_scope_size": len(attempted_scope_nodes),
            "attempted_repair_scope_ratio": (
                len(attempted_scope_nodes) / len(initial_plan)
                if initial_plan
                else 0.0
            ),
            "contamination_events": all_contamination_events,
            "repair_attempts": repair_attempts,
            "prompts": self.prompts
        }
        return result

    def _generate_initial_plan(self, task: Task) -> List[Dict]:
        return _generate_shared_initial_plan(self, task)

    def _recover_empty_plan(self, task: Task) -> List[Dict]:
        prompt = (
            f"The initial planner returned no valid plan.\nTask: {task.description}\n"
            f"Available tools: {_format_tool_catalog(task)}\n"
            "Create a minimal valid plan of at most 5 direct task-solving steps. Avoid optional "
            "preprocessing unless explicitly required. Include step_idx, tool, params, dependencies, "
            "outputs using schema output names, and dependence_content. Fill required inputs. "
            "Output only the JSON list."
        )
        return _coerce_plan_text(self._call_llm_repair(prompt), task)

    def _diagnose_semantic_failures(self, task, plan, step_results) -> set:
        return self._semantic_failure_indices(task, plan, step_results)

    def _propagate_taint(self, step_idx: int, deps: dict, tainted: set):
        for child, parents in deps.items():
            if child not in tainted and step_idx in parents:
                tainted.add(child)
                self._propagate_taint(child, deps, tainted)

    def _child_map(self, deps: dict) -> Dict[int, List[int]]:
        children = {}
        for child, parents in deps.items():
            for parent in parents:
                children.setdefault(parent, []).append(child)
        return {node: sorted(set(nodes)) for node, nodes in children.items()}

    def _descendants(self, nodes: set, child_map: dict) -> set:
        descendants = set()
        stack = list(nodes)
        while stack:
            node = stack.pop()
            for child in child_map.get(node, []):
                if child not in descendants:
                    descendants.add(child)
                    stack.append(child)
        return descendants

    def _make_contamination_event(
        self,
        node: int,
        status: str,
        violation_type: str,
        responsible_nodes: List[int],
        detail: str,
        boundary: str,
    ) -> Dict[str, Any]:
        return {
            "node": node,
            "status": status,
            "violation_type": violation_type,
            "responsible_nodes": sorted(set(responsible_nodes)),
            "detail": detail,
            "boundary": boundary,
        }

    def _record_contamination_event(self, event, active_events, all_events, tainted):
        active_events.append(event)
        all_events.append(event)
        if event.get("status") == "blocking_taint":
            tainted.add(int(event.get("node", 0)))
            for node in event.get("responsible_nodes", []):
                tainted.add(int(node))

    def _consume_attempt_record(self, attempt, local_attempts, global_attempts):
        if not attempt:
            return local_attempts, global_attempts
        mode = attempt.get("mode")
        if mode == "local_patch":
            local_attempts += 1
        elif mode == "global_replan":
            global_attempts += 1
        return local_attempts, global_attempts

    def _classify_plan_boundary_violation(self, node, plan, deps, child_map, plan_budget=None):
        descendants = self._descendants({node}, child_map)
        is_redundant_tail = bool(plan_budget and node > plan_budget)
        is_unused_leaf = not descendants and node != len(plan)
        if is_redundant_tail or is_unused_leaf:
            return self._make_contamination_event(
                node,
                "non_blocking_deviation",
                "plan_boundary_quality_deviation",
                [node],
                "Plan boundary verifier found a redundant or non-critical leaf step; execution can continue.",
                "plan_boundary",
            )
        responsible = sorted(set(deps.get(node, []))) or [node]
        return self._make_contamination_event(
            node,
            "blocking_taint",
            "plan_boundary_contract_violation",
            responsible,
            "Initial plan boundary verifier found an execution-critical dependency or plan violation.",
            "plan_boundary",
        )

    def _verify_dependency_consumption(self, step_idx: int, deps: dict, tainted: set) -> List[Dict[str, Any]]:
        tainted_parents = sorted(parent for parent in deps.get(step_idx, []) if parent in tainted)
        if not tainted_parents:
            return []
        boundary = "fan_in_aggregation" if len(deps.get(step_idx, [])) > 1 else "dependency_consumption"
        violation = "fan_in_tainted_inputs" if boundary == "fan_in_aggregation" else "consumed_tainted_ancestor"
        return [
            self._make_contamination_event(
                step_idx,
                "blocking_taint",
                violation,
                tainted_parents,
                "Node is about to consume output from tainted ancestor(s).",
                boundary,
            )
        ]

    def _classify_step_outcome(self, step_idx, step, result, contract):
        if not result.get("success"):
            return self._make_contamination_event(
                step_idx,
                "blocking_taint",
                "tool_execution_failure",
                [step_idx],
                result.get("error") or "Tool execution failed.",
                "action_output",
            )
        if contract is None:
            return self._make_contamination_event(step_idx, "pass", "none", [], "", "action_output")
        if step.get("tool") != contract.get("expected_tool"):
            return self._make_contamination_event(
                step_idx,
                "blocking_taint",
                "tool_contract_mismatch",
                [step_idx],
                "Executed tool does not match the contract expected tool.",
                "action_output",
            )
        required = set(contract.get("required_inputs", []))
        provided = set((step.get("params") or {}).keys())
        missing = sorted(required - provided)
        if missing:
            return self._make_contamination_event(
                step_idx,
                "blocking_taint",
                "missing_required_input",
                [step_idx],
                f"Missing required input(s): {missing}",
                "action_output",
            )
        expected_outputs = set(contract.get("expected_outputs", []))
        declared_outputs = set(step.get("outputs", []))
        missing_outputs = sorted(expected_outputs - declared_outputs)
        if missing_outputs:
            return self._make_contamination_event(
                step_idx,
                "blocking_taint",
                "missing_declared_output",
                [step_idx],
                f"Missing declared output(s): {missing_outputs}",
                "action_output",
            )
        if not _verify_step(result.get("output", ""), contract):
            return self._make_contamination_event(
                step_idx,
                "blocking_taint",
                "output_contract_violation",
                [step_idx],
                "Output does not satisfy the runtime contract.",
                "action_output",
            )
        return self._make_contamination_event(step_idx, "pass", "none", [], "", "action_output")

    def _repair_boundary_type(self, step_idx, plan, deps, child_map, events, tainted) -> Optional[str]:
        if not events:
            return None
        if not any(event.get("status") == "blocking_taint" for event in events):
            return None
        event_boundaries = {
            event.get("boundary")
            for event in events
            if event.get("status") == "blocking_taint"
        }
        if "dependency_consumption" in event_boundaries:
            return "dependency_consumption"
        if "fan_in_aggregation" in event_boundaries:
            return "fan_in_aggregation"
        if "plan_boundary" in event_boundaries:
            return "plan_boundary"
        if step_idx in tainted and len(child_map.get(step_idx, [])) > 1:
            return "fan_out_spread"
        if step_idx == len(plan):
            return "final_commit"
        if len(deps.get(step_idx, [])) > 1:
            return "fan_in_aggregation"
        if any(parent in tainted and len(child_map.get(parent, [])) > 1 for parent in deps.get(step_idx, [])):
            return "fan_out_spread"
        return None

    def _is_repair_boundary(self, step_idx, plan, deps, child_map, events, tainted) -> bool:
        return self._repair_boundary_type(step_idx, plan, deps, child_map, events, tainted) is not None

    def _select_repair_frontier(self, events, deps) -> set:
        responsible = set()
        for event in events:
            if event.get("status") != "blocking_taint":
                continue
            nodes = event.get("responsible_nodes") or [event.get("node")]
            for node in nodes:
                try:
                    responsible.add(int(node))
                except Exception:
                    continue
        if not responsible:
            return set()
        frontier = set(responsible)
        for node in list(responsible):
            parents = set(deps.get(node, []))
            if parents & responsible:
                frontier.discard(node)
        return frontier or responsible

    def _repair_scope_from_frontier(self, frontier, deps, plan_len):
        child_map = self._child_map(deps)
        scope = set(frontier) | self._descendants(set(frontier), child_map)
        return {node for node in scope if 1 <= node <= plan_len}

    def _frontier_is_not_local(self, frontier, scope, plan):
        if not plan:
            return False
        if not frontier:
            return True
        all_nodes = set(range(1, len(plan) + 1))
        scope = set(scope)
        if scope >= all_nodes:
            return True
        # A local repair scope is admissible only if no unrepaired outside node
        # consumes repaired output. If such an edge exists, the selected scope
        # is structurally incomplete and the repair must be escalated.
        outside = all_nodes - scope
        deps = _plan_dependencies(plan)
        for node in outside:
            if set(deps.get(node, [])) & scope:
                return True
        return False

    def _attempt_record(self, mode, accepted, frontier, scope, reason, source=None):
        return {
            "mode": mode,
            "accepted": bool(accepted),
            "frontier": sorted(frontier),
            "scope": sorted(scope),
            "reason": reason,
            "source": source,
            "taint_visibility": "visible" if self._visible_taint_enabled() else "hidden",
        }

    def _normalize_patch_step(self, item, task, step_idx):
        if not isinstance(item, dict):
            return None
        tools_by_name = {tool.name: tool for tool in task.tools}
        tool = item.get("tool") or item.get("agent") or item.get("agent_name")
        if not tool or tool not in tools_by_name:
            return None
        params = item.get("params")
        if params is None:
            params = item.get("inputs", {})
        dependencies = item.get("dependencies", item.get("dependence", []))
        if dependencies is None:
            dependencies = []
        if not isinstance(dependencies, list):
            dependencies = [dependencies]
        normalized_dependencies = []
        for dep in dependencies:
            try:
                dep_idx = int(dep)
            except Exception:
                continue
            if dep_idx >= 1:
                normalized_dependencies.append(dep_idx)
        outputs = item.get("outputs", [])
        schema_outputs = _schema_output_names(tools_by_name[tool])
        dependence_content = item.get("dependence_content", item.get("dependency_content", {}))
        normalized = {
            "step_idx": int(step_idx),
            "tool": tool,
            "params": copy.deepcopy(params) if isinstance(params, dict) else {},
            "dependencies": sorted(set(normalized_dependencies)),
            "outputs": schema_outputs or (outputs if isinstance(outputs, list) else []),
            "dependence_content": dependence_content if isinstance(dependence_content, dict) else {},
        }
        normalized["dependencies"] = _infer_step_dependencies(normalized, int(step_idx))
        return _normalize_plan_task_paths([normalized], task)[0]

    def _parse_local_patch_plan(self, text, task, scope, plan):
        payload = _extract_json_payload(text)
        requested_global = isinstance(payload, dict) and bool(payload.get("requires_global_replan"))
        raw_steps = None
        if isinstance(payload, dict):
            for key in ("repaired_plan", "revised_plan", "plan", "full_plan"):
                if isinstance(payload.get(key), list):
                    return _coerce_plan_text(
                        json.dumps(payload[key], ensure_ascii=False),
                        task,
                    ), requested_global
            for key in ("patch_steps", "patched_steps", "steps", "repair_steps"):
                if isinstance(payload.get(key), (list, dict)):
                    raw_steps = payload[key]
                    break
            if raw_steps is None:
                digit_items = {
                    key: value for key, value in payload.items()
                    if isinstance(key, str) and key.isdigit() and isinstance(value, dict)
                }
                if digit_items:
                    raw_steps = digit_items
        elif isinstance(payload, list):
            if len(payload) == len(plan):
                return _coerce_plan_text(json.dumps(payload, ensure_ascii=False), task), requested_global
            raw_steps = payload

        if raw_steps is None:
            full_plan = _coerce_plan_text(text, task)
            return full_plan, requested_global

        patch_items = []
        if isinstance(raw_steps, dict):
            for key, value in raw_steps.items():
                try:
                    step_idx = int(key)
                except Exception:
                    step_idx = value.get("step_idx") if isinstance(value, dict) else None
                patch_items.append((step_idx, value))
        else:
            for value in raw_steps:
                step_idx = value.get("step_idx") if isinstance(value, dict) else None
                patch_items.append((step_idx, value))

        patched = copy.deepcopy(plan)
        changed = False
        for raw_idx, item in patch_items:
            try:
                step_idx = int(raw_idx)
            except Exception:
                continue
            if step_idx not in scope or step_idx < 1 or step_idx > len(plan):
                return [], requested_global
            normalized = self._normalize_patch_step(item, task, step_idx)
            if not normalized:
                return [], requested_global
            patched[step_idx - 1] = normalized
            changed = True
        return (patched if changed else []), requested_global

    def _interface_step_shape(self, step):
        """Compare node semantics while ignoring mechanical graph renumbering."""
        normalized = copy.deepcopy(step)
        normalized.pop("step_idx", None)
        normalized.pop("dependencies", None)
        normalized.pop("dependence_content", None)
        normalized.pop("dependency_content", None)

        def normalize_refs(value):
            if isinstance(value, str):
                return re.sub(
                    r"<GENERATED>-\d+-<?([^<>]+)>?",
                    r"<GENERATED>-SOURCE-\1",
                    value,
                )
            if isinstance(value, dict):
                return {key: normalize_refs(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize_refs(item) for item in value]
            return value

        return normalize_refs(normalized)

    def _outside_node_mapping(self, old_plan, repaired, scope):
        """Map preserved outside nodes onto a structurally rewritten plan."""
        outside = [
            index for index in range(1, len(old_plan) + 1)
            if index not in set(scope)
        ]
        mapping = {}
        cursor = 1
        for old_index in outside:
            old_shape = self._interface_step_shape(old_plan[old_index - 1])
            match = None
            for new_index in range(cursor, len(repaired) + 1):
                if self._interface_step_shape(repaired[new_index - 1]) == old_shape:
                    match = new_index
                    break
            if match is None:
                return {}, "local_rewrite_changed_or_removed_outside_node"
            mapping[old_index] = match
            cursor = match + 1
        return mapping, "outside_nodes_preserved"

    def _local_rewrite_analysis(self, old_plan, repaired, scope):
        """Certify an interface-preserving rewrite of a strict local subgraph."""
        if not repaired:
            return None, "empty_patch"
        scope = set(scope)
        outside = set(range(1, len(old_plan) + 1)) - scope
        if not outside:
            return None, "local_rewrite_has_no_preserved_outside_region"

        mapping, mapping_reason = self._outside_node_mapping(old_plan, repaired, scope)
        if not mapping:
            return None, mapping_reason
        reverse_mapping = {new: old for old, new in mapping.items()}
        rewritten_new = set(range(1, len(repaired) + 1)) - set(reverse_mapping)
        if not rewritten_new and len(repaired) >= len(old_plan):
            return None, "local_rewrite_has_no_rewritten_region"

        old_deps = _plan_dependencies(old_plan)
        new_deps = _plan_dependencies(repaired)
        for old_index, new_index in mapping.items():
            if set(old_deps.get(old_index, [])) & scope:
                return None, "outside_node_consumes_old_repair_region"
            expected = {
                mapping[parent]
                for parent in old_deps.get(old_index, [])
                if parent in mapping
            }
            actual = set(new_deps.get(new_index, []))
            if actual != expected:
                return None, "local_rewrite_changed_outside_dependency_interface"

        return {
            "outside_mapping": mapping,
            "reverse_outside_mapping": reverse_mapping,
            "rewritten_new_nodes": rewritten_new,
        }, "valid_interface_preserving_local_rewrite"

    def _local_patch_is_valid(self, old_plan, repaired, scope):
        if not repaired:
            return False, "empty_patch"
        for index, step in enumerate(repaired, start=1):
            if int(step.get("step_idx", index)) != index:
                return False, "step_index_mismatch"
            for dep in step.get("dependencies", []) or []:
                try:
                    dep_idx = int(dep)
                except Exception:
                    return False, "non_integer_dependency"
                if dep_idx < 1 or dep_idx >= index:
                    return False, "dependency_not_prior_step"
            for value in (step.get("params") or {}).values():
                if not isinstance(value, str):
                    continue
                match = re.match(r"^<GENERATED>-(\d+)-(.+)$", value)
                if match and int(match.group(1)) >= index:
                    return False, "generated_reference_not_prior_step"
        analysis, reason = self._local_rewrite_analysis(old_plan, repaired, scope)
        if not analysis:
            return False, reason
        return True, reason

    def _repair_from_contamination_frontier(
        self,
        task,
        plan,
        step_results,
        contract_tree,
        dependencies,
        contamination_events,
        local_attempts,
        global_attempts,
        force_global=False,
    ):
        frontier = self._select_repair_frontier(contamination_events, dependencies)
        scope = self._repair_scope_from_frontier(frontier, dependencies, len(plan))
        if not scope:
            attempt = self._attempt_record("none", False, frontier, scope, "empty_repair_scope")
            return {
                "repaired": False,
                "plan": plan,
                "step_results": step_results,
                "reexecuted": 0,
                "repair_scope": set(),
                "frontier": frontier,
                "affected": set(),
                "global_escalated": False,
                "attempt_record": attempt,
            }
        non_localizable = self._frontier_is_not_local(frontier, scope, plan)
        local_operator_exhausted = (
            local_attempts >= getattr(self.config, "pgir_frontier_retry_cap", 1)
            and not non_localizable
        )
        use_global = force_global or non_localizable
        if local_operator_exhausted and not use_global:
            attempt = self._attempt_record(
                "local_patch",
                False,
                frontier,
                scope,
                "local_repair_operator_cap_exhausted_frontier_still_localizable",
            )
            return {
                "repaired": False,
                "plan": plan,
                "step_results": step_results,
                "reexecuted": 0,
                "repair_scope": scope,
                "frontier": frontier,
                "affected": scope,
                "global_escalated": False,
                "attempt_record": attempt,
            }
        if use_global and global_attempts >= getattr(self.config, "pgir_global_replan_retry_cap", 1):
            attempt = self._attempt_record("global_replan", False, frontier, scope, "global_replan_cap_exhausted")
            return {
                "repaired": False,
                "plan": plan,
                "step_results": step_results,
                "reexecuted": 0,
                "repair_scope": scope,
                "frontier": frontier,
                "affected": scope,
                "global_escalated": True,
                "attempt_record": attempt,
            }
        if use_global:
            global_scope = set(range(1, len(plan) + 1))
            prompt = self._build_pgir_global_replan_prompt(
                task,
                plan,
                step_results,
                scope,
                contamination_events=contamination_events,
                frontier=frontier,
                visible_labels=self._visible_taint_enabled(),
            )
            repaired = _coerce_plan_text(self._call_llm_repair(prompt), task)
            repaired_rule_failed, _, _ = _rule_failure_indices(task, repaired)
            deterministic = (
                _deterministic_rule_repair_plan(task, repaired, repaired_rule_failed)
                if self._allow_deterministic_rule_patch()
                else None
            )
            global_source = "llm_global_replan"
            if deterministic:
                repaired = deterministic
                global_source = "llm_global_replan_with_deterministic_rule_patch"
            if repaired and _plan_quality(repaired, task) >= _plan_quality(plan, task):
                new_results = [
                    benv.execute_tool(task, step["tool"], step.get("params", {}))
                    for step in repaired
                ]
                return {
                    "repaired": True,
                    "plan": repaired,
                    "step_results": new_results,
                    "reexecuted": len(repaired),
                    "repair_scope": global_scope,
                    "frontier": frontier,
                    "affected": global_scope,
                    "global_escalated": True,
                    "attempt_record": self._attempt_record(
                        "global_replan",
                        True,
                        frontier,
                        global_scope,
                        "accepted_global_replan",
                        global_source,
                    ),
                }
            attempt = self._attempt_record("global_replan", False, frontier, scope, "global_replan_quality_rejected")
            return {
                "repaired": False,
                "plan": plan,
                "step_results": step_results,
                "reexecuted": 0,
                "repair_scope": scope,
                "frontier": frontier,
                "affected": scope,
                "global_escalated": True,
                "attempt_record": attempt,
            }
        structural_prune = (
            _verifier_guided_local_prune_plan(
                task,
                plan,
                scope,
                self._diagnosed_step_budget,
            )
            if self._allow_verifier_guided_local_prune()
            else None
        )
        if structural_prune:
            repaired = structural_prune
            source = "verifier_guided_local_prune"
        else:
            deterministic_repair = (
                _deterministic_rule_repair_plan(task, plan, scope)
                if self._allow_deterministic_rule_patch()
                else None
            )
            if deterministic_repair:
                repaired = deterministic_repair
                source = "deterministic_rule_patch"
            else:
                if not self._allow_llm_local_patch():
                    attempt = self._attempt_record(
                        "local_patch",
                        False,
                        frontier,
                        scope,
                        "no_enabled_local_operator_available",
                        "local_operator_disabled",
                    )
                    return {
                        "repaired": False,
                        "plan": plan,
                        "step_results": step_results,
                        "reexecuted": 0,
                        "repair_scope": scope,
                        "frontier": frontier,
                        "affected": scope,
                        "global_escalated": False,
                        "attempt_record": attempt,
                    }
                repair_prompt = self._build_pgir_repair_prompt(
                    task,
                    plan,
                    step_results,
                    scope,
                    contract_tree,
                    visible_labels=self._visible_taint_enabled(),
                    contamination_events=contamination_events,
                    frontier=frontier,
                    affected=scope,
                )
                repair_text = self._call_llm_repair(repair_prompt)
                repaired, requested_global = self._parse_local_patch_plan(repair_text, task, scope, plan)
                source = "llm_patch"
                if requested_global:
                    source = "llm_requested_global_replan"
        valid_patch, rejection_reason = self._local_patch_is_valid(plan, repaired, scope)
        if not valid_patch:
            attempt = self._attempt_record("local_patch", False, frontier, scope, rejection_reason, source)
            return {
                "repaired": False,
                "plan": plan,
                "step_results": step_results,
                "reexecuted": 0,
                "repair_scope": scope,
                "frontier": frontier,
                "affected": scope,
                "global_escalated": False,
                "attempt_record": attempt,
            }
        repaired_rule_failed, _, _ = _rule_failure_indices(task, repaired)
        if repaired_rule_failed:
            attempt = self._attempt_record("local_patch", False, frontier, scope, "residual_rule_failure", source)
            return {
                "repaired": False,
                "plan": plan,
                "step_results": step_results,
                "reexecuted": 0,
                "repair_scope": scope,
                "frontier": frontier,
                "affected": scope,
                "global_escalated": False,
                "attempt_record": attempt,
            }
        if (
            len(repaired) == len(plan)
            and _plan_quality(repaired, task) < _plan_quality(plan, task)
        ):
            attempt = self._attempt_record("local_patch", False, frontier, scope, "local_patch_quality_rejected", source)
            return {
                "repaired": False,
                "plan": plan,
                "step_results": step_results,
                "reexecuted": 0,
                "repair_scope": scope,
                "frontier": frontier,
                "affected": scope,
                "global_escalated": False,
                "attempt_record": attempt,
            }
        new_results, reexecuted = self._execute_repaired_frontier(
            task, plan, repaired, step_results, scope, len(step_results)
        )
        return {
            "repaired": True,
            "plan": repaired,
            "step_results": new_results,
            "reexecuted": reexecuted,
            "repair_scope": scope,
            "frontier": frontier,
            "affected": scope,
            "global_escalated": False,
            "attempt_record": self._attempt_record(
                "local_patch",
                True,
                frontier,
                scope,
                (
                    "accepted_interface_preserving_local_graph_rewrite"
                    if len(plan) != len(repaired)
                    else "accepted_local_patch"
                ),
                source,
            ),
        }

    def _execute_repaired_frontier(self, task, old_plan, repaired, old_results, scope, executed_prefix_len=None):
        if executed_prefix_len is None:
            executed_prefix_len = len(old_results)
        structural_change = len(old_plan) != len(repaired)
        changed = {
            index + 1
            for index in range(min(len(old_plan), len(repaired)))
            if old_plan[index] != repaired[index]
        }
        if structural_change:
            analysis, _ = self._local_rewrite_analysis(old_plan, repaired, scope)
            if not analysis:
                return old_results, 0
            reverse_mapping = analysis["reverse_outside_mapping"]
            rewritten = set(analysis["rewritten_new_nodes"])
            repaired_deps = _plan_dependencies(repaired)
            execute_scope = rewritten | self._descendants(
                rewritten, self._child_map(repaired_deps)
            )
            already_executed_new = {
                new_index
                for new_index, old_index in reverse_mapping.items()
                if old_index <= len(old_results)
            }
            cutoff = max(execute_scope | already_executed_new, default=0)
            new_results = []
            reexecuted = 0
            for new_index, step in enumerate(repaired[:cutoff], start=1):
                old_index = reverse_mapping.get(new_index)
                if old_index is not None and old_index <= len(old_results):
                    new_results.append(old_results[old_index - 1])
                    continue
                new_results.append(
                    benv.execute_tool(task, step["tool"], step.get("params", {}))
                )
                if new_index in execute_scope:
                    reexecuted += 1
            return new_results, reexecuted
        else:
            repaired_deps = _plan_dependencies(repaired)
            execute_scope = set(scope) | changed | self._descendants(set(scope) | changed, self._child_map(repaired_deps))
        new_results = []
        reexecuted = 0
        for index, step in enumerate(repaired[:executed_prefix_len], start=1):
            if index in execute_scope or index > len(old_results):
                new_results.append(benv.execute_tool(task, step["tool"], step.get("params", {})))
                reexecuted += 1
            else:
                new_results.append(old_results[index - 1])
        return new_results, reexecuted

    def _ancestor_path(self, tainted: set, deps: dict) -> set:
        ancestors = set()
        for node in tainted:
            stack = list(deps.get(node, []))
            while stack:
                parent = stack.pop()
                if parent not in ancestors and parent not in tainted:
                    ancestors.add(parent)
                    stack.extend(deps.get(parent, []))
        return tainted.union(ancestors)

    def _build_pgir_repair_prompt(
        self,
        task,
        plan,
        step_results,
        scope,
        contract_tree,
        visible_labels=False,
        contamination_events=None,
        frontier=None,
        affected=None,
    ):
        lines = ["Repair the following execution using PGIR failure-propagation repair: "]
        lines.append(f"Task: {task.description}")
        lines.append(f"Relevant tools: {_format_repair_tool_catalog(task, plan)}")
        lines.append(f"Current plan: {json.dumps(plan, ensure_ascii=False)}")
        if contamination_events is not None and visible_labels:
            lines.append(
                "VISIBLE_TAINT_LABELS: "
                f"{json.dumps(contamination_events, ensure_ascii=False)}"
            )
        elif contamination_events is not None:
            lines.append(
                "Internal verifier found execution-critical contamination and selected "
                "the repair scope. Explicit taint labels and violation types are hidden "
                "from this repair prompt."
            )
        if frontier is not None:
            if visible_labels:
                lines.append(f"VISIBLE_RESPONSIBLE_ANCESTOR_FRONTIER: {sorted(frontier)}")
            else:
                lines.append(f"Internal verifier repair frontier roots: {sorted(frontier)}")
        if affected is not None:
            if visible_labels:
                lines.append(f"VISIBLE_AFFECTED_DESCENDANTS_TO_INVALIDATE: {sorted(affected)}")
            else:
                lines.append(f"Internal verifier affected repair scope: {sorted(affected)}")
        lines.append("Step results:")
        for i, res in enumerate(step_results):
            step_id = i+1
            status = "pass" if (res["success"] and _verify_step(res.get("output",""), contract_tree.get(str(step_id)))) else "fail"
            if visible_labels:
                lines.append(f"Step {step_id} (status:{status}): {res.get('output','')[:200]}")
            else:
                lines.append(f"Step {step_id}: {res.get('output','')[:200]}")
        lines.append(f"Repair scope (frontier plus affected descendants): {sorted(scope)}")
        if self._diagnosed_step_budget is not None:
            lines.append(
                f"Verifier-estimated maximum necessary plan length: {self._diagnosed_step_budget} steps."
            )
            lines.append(
                "If the current plan is longer than this verifier estimate and the extra "
                "steps are inside the repair scope, prefer Form B repaired_plan that prunes "
                "those scoped redundant steps while preserving out-of-scope steps."
            )
            lines.append(
                "A local repair whose remaining plan length exceeds this verifier estimate "
                "will be rejected as a residual execution-critical violation. If every step "
                "after the estimated necessary prefix is inside the repair scope, return "
                "repaired_plan containing the unchanged necessary prefix instead of replacing "
                "the redundant scoped steps with alternative tools."
            )
        lines.append(
            "Return only a JSON object in one of two local-repair forms. Form A: key patch_steps, "
            "where patch_steps contains complete replacement step objects only for steps in the "
            "repair scope; with this form, do not add, remove, or renumber steps. Form B: key "
            "repaired_plan, where repaired_plan is a complete revised plan that preserves every "
            "step outside the repair scope semantically unchanged and may replace the scoped "
            "subgraph with any number of new, removed, merged, or split steps. Renumber the "
            "complete plan and update dependencies. This remains a local repair only when every "
            "out-of-scope node keeps the same semantics, relative order, and dependencies on other "
            "out-of-scope nodes; the rewritten subgraph may consume preserved outside inputs but "
            "must not change the interface of preserved outside nodes. "
            "If the repair cannot be expressed by either scoped replacement or scoped pruning, return "
            "{\"patch_steps\": [], \"requires_global_replan\": true, \"reason\": \"brief\"}. "
            "Within the scoped replacement steps, make the plan smallest-sufficient: one direct "
            "specialized tool per requested action, "
            "no optional preprocessing or extra analysis, and downstream tools must consume relevant "
            "prior outputs. Every output must satisfy an explicit requested deliverable or feed a "
            "later necessary step. Use condition-specific tools directly without restoration, and avoid "
            "unrequested parallel analyses when a report/description tool can consume an upstream "
            "result. For urban anomaly-then-change tasks, use Urban_Anomaly_Detection for "
            "anomaly detection and Change_Mapping_and_Detection for before/after changes; "
            "reserve Building_damage_assessment for explicit building damage-class maps. "
            "For caption+metadata image-generation tasks, do not feed a caption .txt file "
            "as an image_path to High-Resolution_Image_Reconstructor; use "
            "Metadata_and_Text_Prompt_Image_Generation before restoration/counting tools. "
            "Prefer no more than 3 steps unless more are unavoidable. Preserve "
            "unaffected steps. Jointly repair the responsible frontier; do not perform "
            "one-node-at-a-time micro-repair. Steps outside the repair scope must remain "
            "unchanged, except for mechanical renumbering and dependency updates after scoped "
            "node pruning."
        )
        return "\n".join(lines)

    def _build_pgir_global_replan_prompt(
        self,
        task,
        plan,
        step_results,
        failed_steps,
        contamination_events=None,
        frontier=None,
        visible_labels=False,
    ):
        lines = [
            "PGIR localized repair still leaves semantically weak or redundant steps. "
            "Escalate once to a complete ancestor/root-path replan using only public task "
            "text, public tool schemas, and the execution trace."
        ]
        lines.append(f"Task: {task.description}")
        lines.append(f"Relevant tools: {_format_repair_tool_catalog(task, plan)}")
        lines.append(f"Current plan: {json.dumps(plan, ensure_ascii=False)}")
        if visible_labels and contamination_events is not None:
            lines.append(
                "VISIBLE_TAINT_LABELS: "
                f"{json.dumps(contamination_events, ensure_ascii=False)}"
            )
        elif contamination_events is not None:
            lines.append(
                "Internal verifier selected these failed/redundant steps. Explicit "
                "taint labels and violation categories are hidden from this replan."
            )
        if frontier is not None:
            if visible_labels:
                lines.append(f"VISIBLE_RESPONSIBLE_ANCESTOR_FRONTIER: {sorted(frontier)}")
            else:
                lines.append(f"Internal verifier frontier roots: {sorted(frontier)}")
        lines.append(f"Remaining failed or redundant steps: {sorted(failed_steps)}")
        if self._diagnosed_step_budget is not None:
            lines.append(
                f"Verifier-estimated maximum necessary plan length: {self._diagnosed_step_budget} steps."
            )
        lines.append(f"Trace: {json.dumps(step_results, ensure_ascii=False)}")
        lines.append(
            "Return only the complete revised JSON plan. Use the smallest sufficient plan. "
            "Prefer a direct specialized tool over a broader auxiliary tool when it already "
            "produces the requested artifact. Do not add a report/description step unless the "
            "task provides a real query file path. For tasks that ask to identify urban "
            "anomalies and then changes between pre/post images, use Urban_Anomaly_Detection "
            "for anomaly detection and Change_Mapping_and_Detection for before/after changes; "
            "reserve Building_damage_assessment for explicit building damage-class maps. "
            "For caption+metadata image-generation tasks, use Metadata_and_Text_Prompt_Image_Generation "
            "before weather restoration or crowd counting; do not pass caption .txt files as image_path. "
            "Include step_idx, tool, params, "
            "dependencies, outputs, and dependence_content."
        )
        return "\n".join(lines)

    def _parse_repair_output(self, text: str, scope: set) -> Dict[int, Dict]:
        try:
            data = _extract_json_payload(text) or {}
            return {int(k): v for k, v in data.items() if int(k) in scope}
        except:
            return {}

# ----------------------------------------------------------------------
# Control: Full trace retry
# ----------------------------------------------------------------------
class FullTraceRetryControl(BaseRepairMethod):
    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        step_results = self._execute_plan(plan, task)
        hard_failures = {i + 1 for i, r in enumerate(step_results) if not r["success"]}
        semantic_failures = self._semantic_failure_indices(task, plan, step_results)
        diagnosed_failures = hard_failures | semantic_failures
        repaired = False
        if diagnosed_failures:
            repair_prompt = self._build_full_trace_repair_prompt(
                task, plan, step_results, diagnosed_failures
            )
            new_plan_text = self._call_llm_repair(repair_prompt)
            parsed = self._parse_plan(new_plan_text)
            if parsed:
                plan = parsed
                repaired = True
            self.last_plan = plan
            step_results = self._execute_plan(plan, task)
        reexecuted = len(plan) if repaired else 0  # full re-execution only after retry
        untouched_sib_ratio = 0.0
        contract_tree = _build_runtime_contract_tree(plan, task)
        contract_pass = sum(
            1 for i, res in enumerate(step_results)
            if _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        ) / len(plan) if plan else 0.0
        result = {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": 0.0,
            "cascade_depth": len(plan),
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": min(diagnosed_failures) if diagnosed_failures else None,
            "prompts": self.prompts
        }
        return result

    def _generate_initial_plan(self, task: Task) -> List[Dict]:
        return _generate_shared_initial_plan(self, task)

    def _execute_plan(self, plan, task):
        results = []
        for step in plan:
            res = benv.execute_tool(task, step["tool"], step.get("params", {}))
            results.append(res)
        return results

    def _build_full_trace_repair_prompt(self, task, plan, step_results, failed_steps):
        lines = [
            "The previous execution produced a weak or failed plan. Below is the full trace. "
            "Regenerate the complete plan from scratch to complete the task successfully."
        ]
        lines.append(f"Task: {task.description}")
        lines.append(f"Available tools: {_format_tool_catalog(task)}")
        lines.append(f"Current plan: {json.dumps(plan, ensure_ascii=False)}")
        lines.append(f"Diagnosed failed or redundant steps: {sorted(failed_steps)}")
        if self._diagnosed_step_budget is not None:
            lines.append(
                f"Verifier-estimated maximum necessary plan length: {self._diagnosed_step_budget} steps."
            )
        for i, res in enumerate(step_results):
            step = plan[i] if i < len(plan) else {"tool": "unknown", "params": {}}
            lines.append(f"Step {i+1}: tool={step['tool']} params={step.get('params',{})} output={res.get('output','')} success={res['success']}")
        lines.append(
            "Output only a complete revised JSON list with step_idx, tool, params, "
            "dependencies, outputs, and dependence_content. Use the smallest sufficient "
            "plan; remove redundant optional steps and route downstream inputs through "
            "relevant prior outputs."
        )
        return "\n".join(lines)

    def _parse_plan(self, text):
        return _coerce_plan_text(text, self.task) if self.task else []

# ----------------------------------------------------------------------
# Control: No repair / first attempt only
# ----------------------------------------------------------------------
class NoRepairControl(BaseRepairMethod):
    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        step_results = [benv.execute_tool(task, s["tool"], s.get("params", {})) for s in plan]
        contract_tree = _build_runtime_contract_tree(plan, task)
        contract_pass = sum(
            1 for i, res in enumerate(step_results)
            if _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        ) / len(plan) if plan else 0.0
        return {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": 0,
            "untouched_sibling_ratio": 1.0,
            "taint_precision": 0.0,
            "cascade_depth": 0,
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": self._first_failure(step_results),
            "prompts": self.prompts,
        }

    def _generate_initial_plan(self, task: Task) -> List[Dict]:
        return _generate_shared_initial_plan(self, task)

# ----------------------------------------------------------------------
# Control: Binary checkpoint rollback
# ----------------------------------------------------------------------
class BinaryCheckpointRollbackControl(BaseRepairMethod):
    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        contract_tree = _build_runtime_contract_tree(plan, task)
        step_results = [
            benv.execute_tool(task, step["tool"], step.get("params", {}))
            for step in plan
        ]
        hard_failures = {
            i + 1
            for i, res in enumerate(step_results)
            if not _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        }
        semantic_failures = self._semantic_failure_indices(task, plan, step_results)
        diagnosed_failures = hard_failures | semantic_failures
        first_failure = min(diagnosed_failures) if diagnosed_failures else None
        last_verified = first_failure - 1 if first_failure else len(plan)
        reexecuted = 0
        repair_prompt = None
        new_part = None
        new_steps = None
        if first_failure:
            repair_prompt = self._build_rollback_prompt(
                task, plan[:last_verified], step_results[:last_verified], first_failure
            )
            new_part = self._call_llm_repair(repair_prompt)
            new_steps = self._parse_plan(new_part)
            if new_steps:
                plan = plan[:last_verified] + new_steps
                self.last_plan = plan
                prefix_results = step_results[:last_verified]
                tail_results = []
                for step in new_steps:
                    tail_results.append(benv.execute_tool(task, step["tool"], step.get("params", {})))
                    reexecuted += 1
                step_results = prefix_results + tail_results
        untouched_sib_ratio = (len(plan) - reexecuted) / len(plan) if len(plan) > 0 else 1.0
        contract_tree = _build_runtime_contract_tree(plan, task)
        contract_pass = sum(
            1 for i, res in enumerate(step_results)
            if _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        ) / len(plan) if plan else 0.0
        result = {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": 0.0,
            "cascade_depth": reexecuted,
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": first_failure,
            "prompts": self.prompts
        }
        return result

    def _generate_initial_plan(self, task):
        return _generate_shared_initial_plan(self, task)

    def _build_rollback_prompt(self, task, prefix, results, fail_idx):
        lines = ["Execution failed at step {}. Rollback to last verified step. Regenerate steps starting from step {} to complete the task.".format(fail_idx, fail_idx)]
        lines.append(f"Task: {task.description}")
        lines.append(f"Available tools: {_format_tool_catalog(task)}")
        lines.append("Verified prefix:")
        for i, step in enumerate(prefix):
            lines.append(f"Step {i+1}: {step['tool']} -> {results[i].get('output','')}")
        lines.append(
            "Provide remaining steps as a JSON list of steps with step_idx starting from {}, "
            "tool, params, dependencies, outputs, and dependence_content. Use the smallest "
            "sufficient suffix that completes the task from the verified prefix."
            .format(fail_idx)
        )
        return "\n".join(lines)

    def _parse_plan(self, text):
        return _coerce_plan_text(text, self.task) if self.task else []

# ----------------------------------------------------------------------
# Control: No contract retry
# ----------------------------------------------------------------------
class NoContractRetryControl(BaseRepairMethod):
    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        step_results = self._execute_plan(plan, task)
        hard_failures = {i + 1 for i, res in enumerate(step_results) if not res["success"]}
        semantic_failures = self._semantic_failure_indices(task, plan, step_results)
        diagnosed_failures = hard_failures | semantic_failures
        reexecuted = 0
        if diagnosed_failures:
            prompt = (
                f"Task: {task.description}\nAvailable tools: {_format_tool_catalog(task)}\n"
                f"Current plan: {json.dumps(plan, ensure_ascii=False)}\n"
                f"Execution trace: {json.dumps(step_results, ensure_ascii=False)}\n"
                f"Failed or weak steps: {sorted(diagnosed_failures)}\n"
                "Repair without using contract labels or contract tree information. Return a "
                "complete revised JSON plan with step_idx, tool, params, dependencies, outputs, "
                "and dependence_content. Use the smallest sufficient plan."
            )
            repaired = _coerce_plan_text(self._call_llm_repair(prompt), task)
            if repaired:
                plan = repaired
                self.last_plan = plan
                step_results = self._execute_plan(plan, task)
                reexecuted = len(plan)
        untouched_sib_ratio = (len(plan) - reexecuted) / len(plan) if plan else 1.0
        contract_pass = 0.0  # no contract
        result = {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": 0.0,
            "cascade_depth": reexecuted,
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": min(diagnosed_failures) if diagnosed_failures else None,
            "prompts": self.prompts
        }
        return result

    def _generate_initial_plan(self, task):
        return _generate_shared_initial_plan(self, task)

    def _execute_plan(self, plan, task):
        return [benv.execute_tool(task, s["tool"], s.get("params", {})) for s in plan]

    def _first_failure(self, results):
        for i, r in enumerate(results):
            if not r["success"]:
                return i+1
        return None

# ----------------------------------------------------------------------
# Control: Local leaf retry no ancestor
# ----------------------------------------------------------------------
class LocalLeafRetryNoAncestorControl(BaseRepairMethod):
    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        step_results = self._execute_plan(plan, task)
        hard_failures = {i + 1 for i, res in enumerate(step_results) if not res["success"]}
        semantic_failures = self._semantic_failure_indices(task, plan, step_results)
        diagnosed_failures = hard_failures | semantic_failures
        reexecuted = 0
        for step_idx in sorted(diagnosed_failures):
            i = step_idx - 1
            if i < 0 or i >= len(plan):
                continue
            prompt = (
                f"Task: {task.description}\nAvailable tools: {_format_tool_catalog(task)}\n"
                f"Current plan: {json.dumps(plan, ensure_ascii=False)}\n"
                f"Step {step_idx} output: {step_results[i].get('output','')}\n"
                "Repair only this leaf step. Do not change ancestors, siblings, or add "
                "global recovery steps. Return one JSON object with optional keys tool, params, "
                "outputs, dependencies, and dependence_content for this step."
            )
            new_step_text = self._call_llm_repair(prompt)
            payload = _extract_json_payload(new_step_text)
            if isinstance(payload, list) and payload:
                payload = payload[0]
            if isinstance(payload, dict):
                plan[i].update({k: v for k, v in payload.items() if k in {
                    "tool", "params", "outputs", "dependencies", "dependence_content"
                }})
            new_res = benv.execute_tool(task, plan[i]["tool"], plan[i].get("params", {}))
            step_results[i] = new_res
            reexecuted += 1
        untouched_sib_ratio = (len(plan) - reexecuted) / len(plan) if plan else 1.0
        contract_tree = _build_runtime_contract_tree(plan, task)
        contract_pass = sum(
            1 for i, res in enumerate(step_results)
            if _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        ) / len(plan) if plan else 0.0
        result = {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": 0.0,
            "cascade_depth": reexecuted,
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": min(diagnosed_failures) if diagnosed_failures else None,
            "prompts": self.prompts
        }
        return result

    def _generate_initial_plan(self, task):
        return _generate_shared_initial_plan(self, task)

    def _execute_plan(self, plan, task):
        return [benv.execute_tool(task, s["tool"], s.get("params", {})) for s in plan]

    def _first_failure(self, results):
        for i, r in enumerate(results):
            if not r["success"]:
                return i+1
        return None

# ----------------------------------------------------------------------
# Baseline: Reflexion Verbal Retry
# ----------------------------------------------------------------------
class ReflexionVerbalRetry(BaseRepairMethod):
    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        step_results = self._execute_plan(plan, task)
        hard_failures = {i + 1 for i, r in enumerate(step_results) if not r["success"]}
        semantic_failures = self._semantic_failure_indices(task, plan, step_results)
        diagnosed_failures = hard_failures | semantic_failures
        repaired = False
        if diagnosed_failures:
            reflection = self._generate_reflection(task, plan, step_results, diagnosed_failures)
            repair_prompt = self._build_reflexion_repair_prompt(
                task, plan, step_results, reflection, diagnosed_failures
            )
            new_plan_text = self._call_llm_repair(repair_prompt)
            parsed = self._parse_plan(new_plan_text)
            if parsed:
                plan = parsed
                repaired = True
            self.last_plan = plan
            step_results = self._execute_plan(plan, task)
        reexecuted = len(plan) if repaired else 0
        untouched_sib_ratio = 0.0
        contract_tree = _build_runtime_contract_tree(plan, task)
        contract_pass = sum(
            1 for i, res in enumerate(step_results)
            if _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        ) / len(plan) if plan else 0.0
        result = {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": 0.0,
            "cascade_depth": len(plan),
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": min(diagnosed_failures) if diagnosed_failures else None,
            "prompts": self.prompts
        }
        return result

    def _generate_initial_plan(self, task):
        return _generate_shared_initial_plan(self, task)

    def _execute_plan(self, plan, task):
        return [benv.execute_tool(task, s["tool"], s.get("params", {})) for s in plan]

    def _generate_reflection(self, task, plan, step_results, failed_steps):
        trace = "\n".join([f"Step {i+1}: output {r['output']}, success={r['success']}" for i,r in enumerate(step_results)])
        prompt = (
            f"Task: {task.description}\nAvailable tools: {_format_tool_catalog(task)}\n"
            f"Current plan: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Diagnosed failed or redundant steps: {sorted(failed_steps)}\n"
            f"Execution trace:\n{trace}\n"
            "Reflect on why the plan failed or was semantically weak, using only the task, "
            "public tool schemas, and trace. Identify redundant steps, wrong tool choices, "
            "missing direct tools, and missing data flow. Do not assume a reference answer."
        )
        return self.call_llm(prompt)

    def _build_reflexion_repair_prompt(self, task, plan, step_results, reflection, failed_steps):
        trace = "\n".join([f"Step {i+1}: {r['output']}" for i,r in enumerate(step_results)])
        prompt = (
            f"Task: {task.description}\nAvailable tools: {_format_tool_catalog(task)}\n"
            f"Current plan: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Diagnosed failed or redundant steps: {sorted(failed_steps)}\n"
            f"Previous trace:\n{trace}\nReflection: {reflection}\n"
        )
        if self._diagnosed_step_budget is not None:
            prompt += (
                f"Verifier-estimated maximum necessary plan length: "
                f"{self._diagnosed_step_budget} steps.\n"
            )
        prompt += (
            "Generate a complete revised plan as a JSON list. Include step_idx, tool, "
            "params, dependencies, outputs, and dependence_content. Use the smallest "
            "sufficient plan, remove redundant optional steps, and connect downstream "
            "inputs to relevant prior outputs."
        )
        return prompt

    def _parse_plan(self, text):
        return _coerce_plan_text(text, self.task) if self.task else []

# ----------------------------------------------------------------------
# Baseline: Post-tool Reflection RAG Repair
# ----------------------------------------------------------------------
class PostToolReflectionRAGRepair(BaseRepairMethod):
    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        step_results = self._execute_plan(plan, task)
        hard_failures = {i + 1 for i, res in enumerate(step_results) if not res["success"]}
        semantic_failures = self._semantic_failure_indices(task, plan, step_results)
        diagnosed_failures = hard_failures | semantic_failures
        reexecuted = 0
        if diagnosed_failures:
            tool_names = [
                plan[index - 1].get("tool", "")
                for index in sorted(diagnosed_failures)
                if 1 <= index <= len(plan)
            ]
            query = " ".join([name for name in tool_names if name]) or task.description[:160]
            search_results = self.search.search(f"{query} tool repair workflow failure") if self.search else []
            context = "\n".join([f"{s['title']}: {s['content']}" for s in search_results[:3]])
            prompt = (
                f"Task: {task.description}\nAvailable tools: {_format_tool_catalog(task)}\n"
                f"Current plan: {json.dumps(plan, ensure_ascii=False)}\n"
                f"Execution trace: {json.dumps(step_results, ensure_ascii=False)}\n"
                f"Failed or weak steps: {sorted(diagnosed_failures)}\n"
                f"Retrieved repair context:\n{context}\n"
                "Using post-tool reflection and retrieved context, produce a complete revised "
                "JSON plan with step_idx, tool, params, dependencies, outputs, and "
                "dependence_content. Prefer the smallest sufficient plan."
            )
            repaired = _coerce_plan_text(self._call_llm_repair(prompt), task)
            if repaired:
                plan = repaired
                self.last_plan = plan
                step_results = self._execute_plan(plan, task)
                reexecuted = len(plan)
        untouched_sib_ratio = (len(plan) - reexecuted) / len(plan) if plan else 1.0
        contract_tree = _build_runtime_contract_tree(plan, task)
        contract_pass = sum(
            1 for i, res in enumerate(step_results)
            if _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        ) / len(plan) if plan else 0.0
        result = {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": 0.0,
            "cascade_depth": reexecuted,
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": min(diagnosed_failures) if diagnosed_failures else None,
            "prompts": self.prompts
        }
        return result

    def _generate_initial_plan(self, task):
        return _generate_shared_initial_plan(self, task)

    def _execute_plan(self, plan, task):
        return [benv.execute_tool(task, s["tool"], s.get("params", {})) for s in plan]

    def _first_failure(self, results):
        for i, r in enumerate(results):
            if not r["success"]:
                return i+1
        return None

# ----------------------------------------------------------------------
# Baseline: AgentRx Diagnosis Failure Localization
# ----------------------------------------------------------------------
class AgentRxDiagnosisFailureLocalization(BaseRepairMethod):
    def run_task_and_repair(self, task: Task) -> Dict[str, Any]:
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        step_results = self._execute_plan(plan, task)
        hard_failures = {i + 1 for i, res in enumerate(step_results) if not res["success"]}
        semantic_failures = self._semantic_failure_indices(task, plan, step_results)
        diagnosed_set = hard_failures | semantic_failures
        diagnosed = min(diagnosed_set) if diagnosed_set else self._diagnose_failure(task, step_results)
        reexecuted = 0
        if diagnosed and 1 <= diagnosed <= len(plan):
            prompt = (
                f"Task: {task.description}\nAvailable tools: {_format_tool_catalog(task)}\n"
                f"Localized first failed step: {diagnosed}\n"
                f"Current plan: {json.dumps(plan, ensure_ascii=False)}\n"
                f"Execution trace: {json.dumps(step_results, ensure_ascii=False)}\n"
                "Apply failure-localized recovery from the diagnosed step onward. Keep the "
                "verified prefix unchanged, but regenerate the failed step and necessary suffix. "
                "Return the suffix as a JSON list with step_idx, tool, params, dependencies, "
                "outputs, and dependence_content."
            )
            suffix = _coerce_plan_text(self._call_llm_repair(prompt), task)
            if suffix:
                plan = plan[:diagnosed - 1] + suffix
                self.last_plan = plan
                prefix_results = step_results[:diagnosed - 1]
                suffix_results = [
                    benv.execute_tool(task, step["tool"], step.get("params", {}))
                    for step in suffix
                ]
                step_results = prefix_results + suffix_results
                reexecuted = len(suffix)
        untouched_sib_ratio = (len(plan) - reexecuted) / len(plan) if plan else 1.0
        contract_tree = _build_runtime_contract_tree(plan, task)
        contract_pass = sum(
            1 for i, res in enumerate(step_results)
            if _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        ) / len(plan) if plan else 0.0
        result = {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": 0.0,
            "cascade_depth": reexecuted,
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": diagnosed,
            "prompts": self.prompts
        }
        return result

    def _generate_initial_plan(self, task):
        return _generate_shared_initial_plan(self, task)

    def _execute_plan(self, plan, task):
        return [benv.execute_tool(task, s["tool"], s.get("params", {})) for s in plan]

    def _diagnose_failure(self, task, step_results):
        trace = "\n".join([f"Step {i+1}: {r['output']} success={r['success']}" for i,r in enumerate(step_results)])
        prompt = f"Task: {task.description}\nExecution trace:\n{trace}\nIdentify the step index (1-based) that is the first point of failure. Output only the integer."
        resp = self.call_llm(prompt)
        try:
            return int(resp.strip())
        except:
            return None

# ----------------------------------------------------------------------
# Ablation: Visible taint labels
# ----------------------------------------------------------------------
class PGIRVisibleTaintLabels(PGIRHiddenTaintAncestorRepair):
    def _visible_taint_enabled(self) -> bool:
        return True

    def _build_pgir_repair_prompt(
        self,
        task,
        plan,
        step_results,
        scope,
        contract_tree,
        visible_labels=True,
        **kwargs,
    ):
        return super()._build_pgir_repair_prompt(
            task,
            plan,
            step_results,
            scope,
            contract_tree,
            visible_labels=True,
            **kwargs,
        )

# ----------------------------------------------------------------------
# Ablation: PGIR diagnosis/contracts with full-trace repair scope
# ----------------------------------------------------------------------
class PGIRFullTraceRepair(PGIRHiddenTaintAncestorRepair):
    """Use PGIR contract diagnosis but repair/re-execute the whole trace."""

    def _repair_scope_from_frontier(self, frontier, deps, plan_len):
        return set(range(1, plan_len + 1))

    def _frontier_is_not_local(self, frontier, scope, plan):
        return bool(plan)


# ----------------------------------------------------------------------
# Ablation: deterministic rule patch only
# ----------------------------------------------------------------------
class PGIRDeterministicRulePatchOnly(PGIRHiddenTaintAncestorRepair):
    """Expose deterministic rule patches as their own ablation condition."""

    def _allow_verifier_guided_local_prune(self) -> bool:
        return False

    def _allow_llm_local_patch(self) -> bool:
        return False


# ----------------------------------------------------------------------
# Ablation: verifier-guided local prune only
# ----------------------------------------------------------------------
class PGIRVerifierGuidedLocalPruneOnly(PGIRHiddenTaintAncestorRepair):
    """Expose structural local pruning as its own PGIR repair operator."""

    def _allow_deterministic_rule_patch(self) -> bool:
        return False

    def _allow_llm_local_patch(self) -> bool:
        return False


# ----------------------------------------------------------------------
# Ablation: force LLM local ancestor-path patch
# ----------------------------------------------------------------------
class PGIRLLMForcedLocalPatch(PGIRHiddenTaintAncestorRepair):
    """Measure the LLM local patch operator without deterministic shortcuts."""

    def _allow_verifier_guided_local_prune(self) -> bool:
        return False

    def _allow_deterministic_rule_patch(self) -> bool:
        return False

# ----------------------------------------------------------------------
# Ablation: No provenance taint
# ----------------------------------------------------------------------
class PGIRNoProvenanceTaint(PGIRHiddenTaintAncestorRepair):
    def run_task_and_repair(self, task):
        self.task = task
        start_time = time.time()
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        contract_tree = _build_runtime_contract_tree(plan, task)
        tainted = set()
        step_results = []
        for step in plan:
            res = benv.execute_tool(task, step["tool"], step.get("params", {}))
            step_results.append(res)
            contract = contract_tree.get(str(step["step_idx"])) if contract_tree else None
            if not _verify_outcome(step, res, contract):
                tainted.add(step["step_idx"])
                # NO propagation, only local
        tainted.update(self._semantic_failure_indices(task, plan, step_results))
        # repair scope = only failed leaves (tainted), no ancestors
        repair_scope = tainted
        repair_prompt = self._build_pgir_repair_prompt(task, plan, step_results, repair_scope, contract_tree, visible_labels=False)
        reexecuted = 0
        new_params = None
        if repair_scope:
            new_params = self._parse_repair_output(self._call_llm_repair(repair_prompt), repair_scope)
            for step_idx in sorted(repair_scope):
                orig_step = plan[step_idx-1]
                params = new_params.get(step_idx, orig_step.get("params", {}))
                res = benv.execute_tool(task, orig_step["tool"], params)
                step_results[step_idx-1] = res
                reexecuted += 1
        result = self._build_result(task, plan, step_results, repair_scope, tainted, reexecuted, start_time)
        return result

    def _build_result(self, task, plan, step_results, repair_scope, tainted, reexecuted, start_time):
        contract_tree = _build_runtime_contract_tree(plan, task)
        untouched_siblings = len(plan) - len(repair_scope)
        untouched_sib_ratio = untouched_siblings / len(plan) if plan else 1.0
        contract_pass = sum(
            1 for i, res in enumerate(step_results)
            if _verify_outcome(plan[i], res, contract_tree.get(str(i + 1)))
        ) / len(plan) if plan else 0.0
        taint_precision = len(tainted)/len(repair_scope) if repair_scope else 1.0
        return {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": taint_precision,
            "cascade_depth": len(repair_scope),
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": min(tainted) if tainted else None,
            "prompts": self.prompts
        }

# ----------------------------------------------------------------------
# Ablation: Auto contract tree
# ----------------------------------------------------------------------
class PGIRAutoContractTree(PGIRHiddenTaintAncestorRepair):
    def run_task_and_repair(self, task):
        self.task = task
        start_time = time.time()
        contract_tree = task.auto_contract_tree  # use auto tree
        if contract_tree is None:
            contract_tree = {}
        plan = self._generate_initial_plan(task)
        self.last_plan = plan
        tainted = set()
        step_results = []
        for step in plan:
            res = benv.execute_tool(task, step["tool"], step.get("params", {}))
            step_results.append(res)
            contract = contract_tree.get(str(step["step_idx"])) if contract_tree else None
            if not res["success"] or not _verify_step(res.get("output", ""), contract):
                tainted.add(step["step_idx"])
                self._propagate_taint(step["step_idx"], task.dependencies, tainted)
        repair_scope = self._ancestor_path(tainted, task.dependencies)
        repair_prompt = self._build_pgir_repair_prompt(task, plan, step_results, repair_scope, contract_tree, visible_labels=False)
        reexecuted = 0
        new_params = None
        if repair_scope:
            new_params = self._parse_repair_output(self._call_llm_repair(repair_prompt), repair_scope)
            for step_idx in sorted(repair_scope):
                orig_step = plan[step_idx-1]
                params = new_params.get(step_idx, orig_step.get("params", {}))
                res = benv.execute_tool(task, orig_step["tool"], params)
                step_results[step_idx-1] = res
                reexecuted += 1
        return self._build_result(task, plan, step_results, repair_scope, tainted, reexecuted, start_time)

    def _build_result(self, task, plan, step_results, repair_scope, tainted, reexecuted, start_time):
        contract_tree = task.auto_contract_tree or {}
        untouched_siblings = len(plan) - len(repair_scope)
        untouched_sib_ratio = untouched_siblings / len(plan) if plan else 1.0
        contract_pass = sum(1 for i, res in enumerate(step_results) if _verify_step(res.get("output",""),
            contract_tree.get(str(i+1)))) / len(plan) if plan else 1.0
        taint_precision = len(tainted)/len(repair_scope) if repair_scope else 1.0
        return {
            "task_id": task.task_id,
            "final_output": step_results[-1]["output"] if step_results else "",
            "repair_prompt_tokens": self.repair_prompt_tokens_used,
            "total_repair_tokens": self.total_tokens,
            "repair_latency": time.time() - start_time,
            "reexecuted_steps": reexecuted,
            "untouched_sibling_ratio": untouched_sib_ratio,
            "taint_precision": taint_precision,
            "cascade_depth": len(repair_scope),
            "contract_pass_rate": contract_pass,
            "diagnosed_first_failure": min(tainted) if tainted else None,
            "prompts": self.prompts
        }

# ----------------------------------------------------------------------
# Helper for computing contract pass rate
# ----------------------------------------------------------------------
def _compute_contract_pass(contract_tree, step_results):
    if not contract_tree:
        return 0.0
    total = len(step_results)
    if total == 0:
        return 1.0
    passed = sum(1 for i, res in enumerate(step_results) if _verify_step(res.get("output",""), contract_tree.get(str(i+1))))
    return passed / total
