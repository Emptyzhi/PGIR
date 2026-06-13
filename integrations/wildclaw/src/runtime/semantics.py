from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
from typing import Any, Callable

from .artifacts import normalize_artifact, tool_artifacts


@dataclass(frozen=True)
class ToolEffect:
    tool_name: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    mutates: tuple[str, ...] = ()
    required_params: tuple[str, ...] = ()
    replayable: bool = False
    irreversible: bool = False
    confidence: str = "structural"


SemanticAnalyzer = Callable[[str, dict[str, Any]], ToolEffect]


class ToolSemanticRegistry:
    """Deterministically infer execution effects from tool calls."""

    def __init__(self) -> None:
        self._analyzers: dict[str, SemanticAnalyzer] = {}
        for tool in ("read", "image", "web_fetch"):
            self.register(tool, self._read_effect)
        self.register("write", self._write_effect)
        self.register("edit", self._edit_effect)
        self.register("exec", self._exec_effect)

    def register(self, tool_name: str, analyzer: SemanticAnalyzer) -> None:
        self._analyzers[tool_name.lower()] = analyzer

    def analyze(self, tool_name: str, params: dict[str, Any]) -> ToolEffect:
        tool = tool_name.lower()
        analyzer = self._analyzers.get(tool, self._generic_effect)
        return analyzer(tool, params)

    @staticmethod
    def _read_effect(tool: str, params: dict[str, Any]) -> ToolEffect:
        consumes, _ = tool_artifacts(tool, params)
        required = ("url",) if tool == "web_fetch" else ("path",)
        return ToolEffect(
            tool_name=tool,
            consumes=consumes,
            required_params=required,
            replayable=tool in {"read", "image"},
        )

    @staticmethod
    def _write_effect(tool: str, params: dict[str, Any]) -> ToolEffect:
        _, produces = tool_artifacts(tool, params)
        return ToolEffect(
            tool_name=tool,
            produces=produces,
            mutates=produces,
            required_params=("path", "content"),
            replayable=True,
        )

    @staticmethod
    def _edit_effect(tool: str, params: dict[str, Any]) -> ToolEffect:
        consumes, produces = tool_artifacts(tool, params)
        required = ("path",)
        if tool == "edit":
            required += ("old_text", "new_text")
        return ToolEffect(
            tool_name=tool,
            consumes=consumes,
            produces=produces,
            mutates=produces,
            required_params=required,
            replayable=tool == "edit",
        )

    @staticmethod
    def _exec_effect(tool: str, params: dict[str, Any]) -> ToolEffect:
        command = _command_text(params)
        consumes, produces = tool_artifacts(tool, params)
        consumed = set(consumes)
        produced = set(produces)
        mutated = set(produced)
        irreversible = _has_external_side_effect(command)

        for source, target in _command_file_pairs(command):
            if source:
                consumed.add(normalize_artifact(source))
            if target:
                artifact = normalize_artifact(target)
                produced.add(artifact)
                mutated.add(artifact)
        for target in _removed_paths(command):
            mutated.add(normalize_artifact(target))

        absolute_paths = re.findall(r"(?<![\w:])(/[^\s\"';|&]+)", command)
        replayable = bool(command) and not irreversible and all(
            path.startswith("/tmp_workspace/") for path in absolute_paths
        )
        return ToolEffect(
            tool_name=tool,
            consumes=tuple(sorted(consumed - produced)),
            produces=tuple(sorted(produced)),
            mutates=tuple(sorted(mutated)),
            required_params=("command",),
            replayable=replayable,
            irreversible=irreversible,
            confidence="parsed_shell" if command else "unknown",
        )

    @staticmethod
    def _generic_effect(tool: str, params: dict[str, Any]) -> ToolEffect:
        consumes, produces = tool_artifacts(tool, params)
        return ToolEffect(
            tool_name=tool,
            consumes=consumes,
            produces=produces,
            mutates=produces,
            replayable=False,
            irreversible=True,
            confidence="conservative_unknown",
        )


def _command_text(params: dict[str, Any]) -> str:
    for key in ("command", "cmd", "script"):
        value = params.get(key)
        if isinstance(value, str):
            return value
    return ""


def _command_file_pairs(command: str) -> tuple[tuple[str | None, str | None], ...]:
    pairs: list[tuple[str | None, str | None]] = []
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        base = token.rsplit("/", 1)[-1].lower()
        if base in {"cp", "mv", "pdftotext"} and index + 2 < len(tokens):
            source, target = tokens[index + 1], tokens[index + 2]
            if _artifact_arg(source) and _artifact_arg(target):
                pairs.append((source, target))
        if base in {"curl", "wget"}:
            for flag in ("-o", "--output"):
                if flag in tokens[index + 1 :]:
                    output_index = tokens.index(flag, index + 1) + 1
                    if output_index < len(tokens):
                        url = next(
                            (
                                item
                                for item in tokens[index + 1 :]
                                if item.startswith(("http://", "https://"))
                            ),
                            None,
                        )
                        pairs.append((url, tokens[output_index]))
    return tuple(pairs)


def _artifact_arg(value: str) -> bool:
    return value.startswith(("/", "~/", "http://", "https://"))


def _removed_paths(command: str) -> tuple[str, ...]:
    return tuple(
        match
        for match in re.findall(
            r"\b(?:rm|unlink)\s+(?:-[^\s]+\s+)*(/[^\s\"';|&]+)",
            command,
        )
    )


def _has_external_side_effect(command: str) -> bool:
    lowered = command.lower()
    return bool(
        re.search(
            r"\b(?:send|upload|publish|post|email|mail|git\s+push|curl\s+[^|]*(?:-x|--request)\s*(?:post|put|delete))\b",
            lowered,
        )
    )
