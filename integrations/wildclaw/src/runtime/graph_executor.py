from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .adapter import AgentExecutorAdapter
from .controller import RuntimeController
from .model import (
    BoundaryKind,
    ControlAction,
    ExecutionEvent,
    EventType,
)


NodeRunner = Callable[[dict[str, Any]], Any]
NodeRepairer = Callable[[Any, tuple[Any, ...]], Any]
BoundaryValidator = Callable[[dict[str, Any]], tuple[Any, ...]]


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    run: NodeRunner
    parents: tuple[str, ...] = ()
    repair: NodeRepairer | None = None
    validate_before_consume: BoundaryValidator | None = None


class GraphExecutorAdapter(AgentExecutorAdapter):
    """Reference executor implementing the complete PGIR control interface."""

    def __init__(
        self,
        nodes: list[GraphNode],
        *,
        final_validator: BoundaryValidator | None = None,
        global_replanner: Callable[[str], bool] | None = None,
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.order = self._topological_order(nodes)
        self.outputs: dict[str, Any] = {}
        self.executed: list[str] = []
        self.paused = False
        self.pending_event: ExecutionEvent | None = None
        self.final_validator = final_validator
        self.global_replanner = global_replanner
        self.operations: list[dict[str, Any]] = []

    def run(self, controller: RuntimeController) -> dict[str, Any]:
        controller.handle(ExecutionEvent(EventType.TASK_START, "task_root"), self)
        for node_id in self.order:
            node = self.nodes[node_id]
            if node.parents:
                boundary = (
                    BoundaryKind.FAN_IN
                    if len(node.parents) > 1
                    else BoundaryKind.DEPENDENCY_CONSUMPTION
                )
                event_type = EventType.FAN_IN if len(node.parents) > 1 else EventType.BEFORE_CONSUME
                boundary_event = ExecutionEvent(
                    event_type,
                    node_id,
                    boundary=boundary,
                    parents=node.parents,
                    consumes=tuple(f"{parent}.out" for parent in node.parents),
                )
                self.pending_event = boundary_event
                decision = controller.handle(boundary_event, self)
                if decision.action == ControlAction.ABORT:
                    raise RuntimeError(decision.reason)
                if decision.action == ControlAction.GLOBAL_REPLAN:
                    raise RuntimeError("Graph changed by global replan; start a new execution run")

            controller.handle(
                ExecutionEvent(EventType.NODE_START, node_id, parents=node.parents),
                self,
            )
            self._execute_node(node_id)
            controller.handle(
                ExecutionEvent(
                    EventType.NODE_FINISH,
                    node_id,
                    parents=node.parents,
                    consumes=tuple(f"{parent}.out" for parent in node.parents),
                    produces=(f"{node_id}.out",),
                ),
                self,
            )

        final_event = ExecutionEvent(
            EventType.FINAL_COMMIT,
            "final_commit",
            boundary=BoundaryKind.FINAL_COMMIT,
            parents=(self.order[-1],) if self.order else (),
        )
        self.pending_event = final_event
        decision = controller.handle(final_event, self)
        if decision.action == ControlAction.ABORT:
            raise RuntimeError(decision.reason)
        return dict(self.outputs)

    def pause(self) -> None:
        self.paused = True
        self.operations.append({"operation": "pause"})

    def resume(self) -> None:
        self.paused = False
        self.operations.append({"operation": "resume"})

    def snapshot(self, label: str) -> Any:
        return {
            "label": label,
            "outputs": dict(self.outputs),
            "executed": list(self.executed),
        }

    def restore(self, snapshot: Any) -> None:
        self.outputs = dict(snapshot["outputs"])
        self.executed = list(snapshot["executed"])
        self.operations.append({"operation": "restore", "label": snapshot["label"]})

    def repair_nodes(self, node_ids: tuple[str, ...], violations: tuple[Any, ...]) -> bool:
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if node is None or node.repair is None or node_id not in self.outputs:
                return False
            self.outputs[node_id] = node.repair(self.outputs[node_id], violations)
        self.operations.append({"operation": "repair_nodes", "nodes": list(node_ids)})
        return True

    def replay_nodes(self, node_ids: tuple[str, ...]) -> bool:
        selected = set(node_ids)
        for node_id in self.order:
            if node_id in selected:
                self._execute_node(node_id, replay=True)
        self.operations.append({"operation": "replay_nodes", "nodes": list(node_ids)})
        return True

    def global_replan(self, reason: str) -> bool:
        self.operations.append({"operation": "global_replan", "reason": reason})
        return self.global_replanner(reason) if self.global_replanner else False

    def boundary_violations(self, event: ExecutionEvent) -> tuple[Any, ...]:
        if event.boundary == BoundaryKind.FINAL_COMMIT:
            return self.final_validator(dict(self.outputs)) if self.final_validator else ()
        node = self.nodes.get(event.node_id)
        if node is None or node.validate_before_consume is None:
            return ()
        parent_outputs = {parent: self.outputs[parent] for parent in node.parents}
        return node.validate_before_consume(parent_outputs)

    def _execute_node(self, node_id: str, *, replay: bool = False) -> None:
        node = self.nodes[node_id]
        inputs = {parent: self.outputs[parent] for parent in node.parents}
        self.outputs[node_id] = node.run(inputs)
        if node_id not in self.executed:
            self.executed.append(node_id)
        self.operations.append(
            {"operation": "replay_node" if replay else "execute_node", "node": node_id}
        )

    @staticmethod
    def _topological_order(nodes: list[GraphNode]) -> list[str]:
        remaining = {node.node_id: set(node.parents) for node in nodes}
        order: list[str] = []
        while remaining:
            ready = sorted(node_id for node_id, parents in remaining.items() if not parents)
            if not ready:
                raise ValueError("Graph contains a cycle or missing parent")
            for node_id in ready:
                order.append(node_id)
                remaining.pop(node_id)
                for parents in remaining.values():
                    parents.discard(node_id)
        return order
