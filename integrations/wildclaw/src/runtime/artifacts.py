from __future__ import annotations

import json
import hashlib
import posixpath
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


WORKSPACE_PREFIXES = ("/tmp_workspace/", "/root/.openclaw/workspace/")
PATH_KEYS = ("path", "file_path", "filePath", "target", "destination", "output")
CONCRETE_ARTIFACT_PREFIXES = ("workspace:", "url:", "home:")


def normalize_artifact(value: str) -> str:
    value = value.strip().strip("\"'")
    if value.startswith(("workspace:", "url:", "home:", "tool-result:")):
        return value
    if value.startswith(("http://", "https://")):
        parsed = urlsplit(value)
        return "url:" + urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.query, "")
        )
    for prefix in WORKSPACE_PREFIXES:
        if value.startswith(prefix):
            rel = posixpath.normpath(value[len(prefix) :]).lstrip("/")
            return f"workspace:{rel}"
    if value.startswith("~/"):
        return f"home:{posixpath.normpath(value[2:])}"
    return value


def is_concrete_artifact(value: str) -> bool:
    """Return whether an artifact can be verified directly from a tool call/workspace."""
    return value.startswith(CONCRETE_ARTIFACT_PREFIXES)


def tool_artifacts(
    tool_name: str,
    params: dict[str, Any],
    *,
    result: Any = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tool = tool_name.lower()
    refs = set(_extract_refs(params))
    if tool in {"write", "edit", "apply_patch"}:
        # Text bodies may mention URLs and workspace paths without consuming or
        # producing them. Only explicit path-like fields describe file effects.
        refs = set(_extract_path_refs(params))
    consumes: set[str] = set()
    produces: set[str] = set()
    if tool in {"read", "image", "web_fetch"}:
        consumes.update(refs)
    elif tool in {"write", "edit", "apply_patch"}:
        consumes.update(refs if tool != "write" else ())
        produces.update(refs)
    elif tool == "exec":
        command = _command_text(params)
        outputs = {normalize_artifact(path) for path in _redirect_targets(command)}
        produces.update(outputs)
        consumes.update(refs - outputs)
    else:
        consumes.update(refs)
    if result is not None:
        produces.add(f"tool-result:{_stable_result_key(tool, params)}")
    return tuple(sorted(consumes)), tuple(sorted(produces))


def _extract_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PATH_KEYS and isinstance(item, str):
                refs.add(normalize_artifact(item))
            refs.update(_extract_refs(item))
        return refs
    if isinstance(value, list):
        for item in value:
            refs.update(_extract_refs(item))
        return refs
    if not isinstance(value, str):
        return refs
    for match in re.findall(
        r"(?:https?://[^\s\"']+|/tmp_workspace/[^\s\"']+|/root/\.openclaw/workspace/[^\s\"']+)",
        value,
    ):
        refs.add(normalize_artifact(match.rstrip(";|&,.)")))
    return refs


def _extract_path_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PATH_KEYS and isinstance(item, str):
                refs.add(normalize_artifact(item))
            elif isinstance(item, (dict, list)):
                refs.update(_extract_path_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_extract_path_refs(item))
    return refs


def _command_text(params: dict[str, Any]) -> str:
    for key in ("command", "cmd", "script"):
        value = params.get(key)
        if isinstance(value, str):
            return value
    return ""


def _redirect_targets(command: str) -> tuple[str, ...]:
    return tuple(
        path.rstrip(";|&")
        for path in re.findall(
            r"(?:>>?|(?:-o))\s*(/tmp_workspace/[^\s\"']+)",
            command,
            flags=re.IGNORECASE,
        )
    )


def _stable_result_key(tool: str, params: dict[str, Any]) -> str:
    canonical = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{tool}:{digest}"
