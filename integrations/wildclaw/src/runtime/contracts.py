from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .adapter import AgentExecutorAdapter
from .auto_contracts import AutoContractMonitor
from .model import ContractViolation, ExecutionEvent
from .plan import PlanContractMonitor


ContractPredicate = Callable[[ExecutionEvent], bool]


class ContractMonitor(Protocol):
    name: str

    def verify(
        self,
        event: ExecutionEvent,
        adapter: AgentExecutorAdapter,
    ) -> tuple[ContractViolation | dict[str, Any], ...]: ...


@dataclass(frozen=True)
class RuntimeContract:
    contract_id: str
    node_id: str
    predicate: ContractPredicate
    responsible_nodes: tuple[str, ...] = ()
    blocking: bool = True
    description: str = ""


class ContractTree:
    def __init__(self, monitors: tuple[ContractMonitor, ...] = ()) -> None:
        self._contracts: dict[str, list[RuntimeContract]] = {}
        self._monitors: list[ContractMonitor] = list(monitors)

    def add(self, contract: RuntimeContract) -> None:
        self._contracts.setdefault(contract.node_id, []).append(contract)

    def add_monitor(self, monitor: ContractMonitor) -> None:
        self._monitors.append(monitor)

    @property
    def monitors(self) -> tuple[ContractMonitor, ...]:
        return tuple(self._monitors)

    @property
    def monitor_names(self) -> tuple[str, ...]:
        return tuple(monitor.name for monitor in self._monitors)

    def verify(
        self,
        event: ExecutionEvent,
        adapter: AgentExecutorAdapter,
    ) -> tuple[ContractViolation, ...]:
        violations: list[ContractViolation] = []
        for contract in self._contracts.get(event.node_id, ()):
            if contract.predicate(event):
                continue
            responsible = contract.responsible_nodes or (event.node_id,)
            violations.append(
                ContractViolation(
                    contract_id=contract.contract_id,
                    node_id=event.node_id,
                    boundary=event.boundary,
                    responsible_nodes=responsible,
                    blocking=contract.blocking,
                    details={"description": contract.description},
                )
            )
        for monitor in self._monitors:
            for raw in monitor.verify(event, adapter):
                if isinstance(raw, ContractViolation):
                    violations.append(raw)
                elif isinstance(raw, dict):
                    violations.append(self._from_dict(event, raw))
        return tuple(violations)

    @staticmethod
    def _from_dict(event: ExecutionEvent, raw: dict[str, Any]) -> ContractViolation:
        responsible = tuple(raw.get("responsible_nodes") or (event.node_id,))
        return ContractViolation(
            contract_id=str(raw.get("contract_id", "adapter_boundary_violation")),
            node_id=str(raw.get("node_id", event.node_id)),
            boundary=event.boundary,
            responsible_nodes=responsible,
            blocking=bool(raw.get("blocking", True)),
            details=dict(raw.get("details") or {}),
        )


class AdapterBoundaryContract:
    """Compatibility monitor for executor-defined validators."""

    name = "adapter_boundary_contract"

    def verify(
        self,
        event: ExecutionEvent,
        adapter: AgentExecutorAdapter,
    ) -> tuple[ContractViolation | dict[str, Any], ...]:
        return tuple(adapter.boundary_violations(event))


class SnapshotGuardContract:
    """Protect pre-existing workspace artifacts without embedding policy in an executor."""

    name = "snapshot_guard_contract"

    def verify(
        self,
        event: ExecutionEvent,
        adapter: AgentExecutorAdapter,
    ) -> tuple[ContractViolation, ...]:
        mutation_paths = getattr(adapter, "protected_mutation_paths", None)
        if callable(mutation_paths):
            paths = tuple(mutation_paths(event))
            if paths:
                return (
                    ContractViolation(
                        contract_id="preserve_preexisting_workspace_files",
                        node_id=event.node_id,
                        boundary=event.boundary,
                        responsible_nodes=(event.node_id,),
                        blocking=True,
                        details={
                            "paths": list(paths),
                            "reason": "tool call would mutate a protected pre-existing workspace file",
                        },
                    ),
                )

        workspace_changes = getattr(adapter, "workspace_changes", None)
        if callable(workspace_changes):
            changes = list(workspace_changes(event))
            if changes:
                return (
                    ContractViolation(
                        contract_id="preserve_preexisting_workspace_files",
                        node_id=event.node_id,
                        boundary=event.boundary,
                        responsible_nodes=("workspace_mutation",),
                        blocking=True,
                        details={"changes": changes},
                    ),
                )
        return ()


class ToolExecutionContract:
    """Defer failed-tool repair until the next controllable consumption boundary."""

    name = "tool_execution_contract"

    def verify(
        self,
        event: ExecutionEvent,
        adapter: AgentExecutorAdapter,
    ) -> tuple[ContractViolation, ...]:
        pending = getattr(adapter, "pending_execution_failures", None)
        return tuple(pending(event)) if callable(pending) else ()


def default_contract_tree() -> ContractTree:
    return ContractTree(
        monitors=(
            AutoContractMonitor(),
            PlanContractMonitor(),
            SnapshotGuardContract(),
            ToolExecutionContract(),
            AdapterBoundaryContract(),
        )
    )


def execution_failure_contract_tree() -> ContractTree:
    """Retain only concrete executor failures for the no-contract ablation."""
    return ContractTree(monitors=(ToolExecutionContract(),))
