from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .adapter import AgentExecutorAdapter
from .artifacts import is_concrete_artifact, normalize_artifact
from .model import BoundaryKind, ContractViolation, EventType, ExecutionEvent
from .semantics import ToolEffect, ToolSemanticRegistry


@dataclass(frozen=True)
class GoalArtifactContract:
    artifact: str
    kind: str = "exists_nonempty"
    blocking: bool = True


class AutoContractMonitor:
    """Synthesize verifiable contracts from task intent and runtime tool semantics."""

    name = "automatic_contract_synthesis"

    def __init__(self, registry: ToolSemanticRegistry | None = None) -> None:
        self.registry = registry or ToolSemanticRegistry()
        self.effects: dict[str, ToolEffect] = {}
        self.goal_contracts: tuple[GoalArtifactContract, ...] = ()
        self.synthesized_count = 0

    def verify(
        self,
        event: ExecutionEvent,
        adapter: AgentExecutorAdapter,
    ) -> tuple[ContractViolation, ...]:
        self._ensure_goal_contracts(adapter)
        if event.type in {EventType.BEFORE_CONSUME, EventType.FAN_IN}:
            return self._verify_before(event)
        if event.type == EventType.NODE_FINISH:
            return self._verify_after(event)
        if event.boundary == BoundaryKind.FINAL_COMMIT:
            return self._verify_goals(event, adapter)
        return ()

    def report(self) -> dict[str, Any]:
        return {
            "synthesized_tool_contracts": self.synthesized_count,
            "goal_contracts": [asdict(contract) for contract in self.goal_contracts],
            "tool_effects": {
                node: asdict(effect) for node, effect in self.effects.items()
            },
        }

    def _verify_before(self, event: ExecutionEvent) -> tuple[ContractViolation, ...]:
        tool = str(event.payload.get("tool_name") or "")
        if not tool or tool == "pgir_declare_plan":
            return ()
        params = dict(event.payload.get("params") or {})
        effect = self.registry.analyze(tool, params)
        self.effects[event.node_id] = effect
        self.synthesized_count += 1
        missing = tuple(
            name for name in effect.required_params if not _has_param(params, name)
        )
        if not missing:
            return ()
        return (
            ContractViolation(
                contract_id="auto_required_tool_params",
                node_id=event.node_id,
                boundary=event.boundary,
                responsible_nodes=(event.node_id,),
                blocking=True,
                details={"tool": tool, "missing": list(missing), "source": effect.confidence},
            ),
        )

    def _verify_after(self, event: ExecutionEvent) -> tuple[ContractViolation, ...]:
        effect = self.effects.get(event.node_id)
        if effect is None or event.payload.get("error"):
            return ()
        expected = {
            artifact for artifact in effect.produces if is_concrete_artifact(artifact)
        }
        missing = tuple(sorted(expected - set(event.produces)))
        if not missing:
            return ()
        return (
            ContractViolation(
                contract_id="auto_expected_tool_outputs",
                node_id=event.node_id,
                boundary=event.boundary,
                responsible_nodes=(event.node_id,),
                blocking=True,
                details={
                    "tool": effect.tool_name,
                    "missing": list(missing),
                    "source": effect.confidence,
                },
            ),
        )

    def _verify_goals(
        self,
        event: ExecutionEvent,
        adapter: AgentExecutorAdapter,
    ) -> tuple[ContractViolation, ...]:
        inspect = getattr(adapter, "inspect_artifact", None)
        if not callable(inspect):
            return ()
        violations: list[ContractViolation] = []
        for contract in self.goal_contracts:
            observation = inspect(contract.artifact)
            if _goal_satisfied(contract, observation):
                continue
            violations.append(
                ContractViolation(
                    contract_id=f"auto_goal_{contract.kind}",
                    node_id=event.node_id,
                    boundary=event.boundary,
                    responsible_nodes=("task_root",),
                    blocking=contract.blocking,
                    details={
                        "artifact": contract.artifact,
                        "observation": observation,
                    },
                )
            )
        return tuple(violations)

    def _ensure_goal_contracts(self, adapter: AgentExecutorAdapter) -> None:
        if self.goal_contracts:
            return
        spec = getattr(adapter, "spec", None)
        prompt = str(getattr(spec, "prompt", "") or "")
        self.goal_contracts = synthesize_goal_contracts(prompt)


def synthesize_goal_contracts(prompt: str) -> tuple[GoalArtifactContract, ...]:
    contracts: dict[str, GoalArtifactContract] = {}
    for match in re.finditer(r"/tmp_workspace/[^\s,;:'\"`?]+", prompt):
        raw = match.group(0).rstrip(".)]")
        prefix = prompt[: match.start()]
        context = re.split(r"[.!?\n;]", prefix)[-1].lower()
        if not re.search(
            r"\b(?:create|write|save|download|produce|generate|export|summarize|summarise|store|put)\b",
            context,
        ):
            continue
        artifact = normalize_artifact(raw)
        kind = "valid_pdf" if artifact.lower().endswith(".pdf") else "exists_nonempty"
        contracts[artifact] = GoalArtifactContract(artifact=artifact, kind=kind)
    return tuple(contracts.values())


def _has_param(params: dict[str, Any], name: str) -> bool:
    aliases = {
        "path": ("path", "file_path", "filePath"),
        "content": ("content", "text"),
        "command": ("command", "cmd", "script"),
        "old_text": ("oldText", "old_string"),
        "new_text": ("newText", "new_string"),
        "url": ("url",),
    }
    return any(key in params and params[key] not in (None, "") for key in aliases.get(name, (name,)))


def _goal_satisfied(contract: GoalArtifactContract, observation: dict[str, Any]) -> bool:
    if not observation.get("exists") or not observation.get("is_file"):
        return False
    if observation.get("size", 0) <= 0:
        return False
    if contract.kind == "valid_pdf":
        return observation.get("magic") == "%PDF-"
    return True
