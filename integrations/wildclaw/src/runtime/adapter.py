from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .model import ExecutionEvent


class AgentExecutorAdapter(ABC):
    """Control surface required by executor-agnostic repair controllers."""

    @abstractmethod
    def pause(self) -> None:
        """Pause execution at the current boundary."""

    @abstractmethod
    def resume(self) -> None:
        """Resume execution after a successful intervention."""

    @abstractmethod
    def snapshot(self, label: str) -> Any:
        """Capture restorable executor state."""

    @abstractmethod
    def restore(self, snapshot: Any) -> None:
        """Restore a previously captured executor state."""

    @abstractmethod
    def repair_nodes(self, node_ids: tuple[str, ...], violations: tuple[Any, ...]) -> bool:
        """Repair the selected frontier nodes and return whether repair succeeded."""

    @abstractmethod
    def replay_nodes(self, node_ids: tuple[str, ...]) -> bool:
        """Replay an already-executed affected forward subgraph."""

    @abstractmethod
    def global_replan(self, reason: str) -> bool:
        """Replace the current plan when no strict local frontier exists."""

    def boundary_violations(self, event: ExecutionEvent) -> tuple[Any, ...]:
        """Return adapter-observed violations for a boundary event."""
        _ = event
        return ()

    def install_plan(self, plan: Any) -> None:
        """Install a controller-validated execution plan when supported."""
        _ = plan

    def prepare_repair(
        self,
        snapshot: Any,
        node_ids: tuple[str, ...],
        affected_nodes: tuple[str, ...] = (),
    ) -> bool:
        """Restore executor state before repairing a frontier when required.

        Stateless/in-memory executors may keep the current state and return
        True. Side-effecting executors should restore the supplied checkpoint.
        """
        _ = (snapshot, node_ids, affected_nodes)
        return True

    def replay_snapshots(self) -> dict[str, Any]:
        """Return refreshed pre-node snapshots captured during replay."""
        return {}

    def select_repair_checkpoint(
        self,
        snapshots: dict[str, Any],
        node_ids: tuple[str, ...],
    ) -> Any:
        """Choose the checkpoint preceding the earliest repaired frontier node."""
        return next((snapshots[node] for node in node_ids if node in snapshots), None)
