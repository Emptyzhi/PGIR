from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapter import AgentExecutorAdapter
from .artifacts import normalize_artifact
from .model import BoundaryKind, ContractViolation, EventType, ExecutionEvent


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    description: str = ""
    tool: str | None = None
    parents: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    required_params: tuple[str, ...] = ()
    replayable: bool = False


@dataclass
class ExecutionPlan:
    nodes: dict[str, PlanNode] = field(default_factory=dict)
    bindings: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExecutionPlan":
        nodes: dict[str, PlanNode] = {}
        for item in raw.get("nodes") or ():
            node = PlanNode(
                node_id=str(item["id"]),
                description=str(item.get("description") or ""),
                tool=str(item["tool"]) if item.get("tool") else None,
                parents=tuple(str(value) for value in item.get("parents") or ()),
                consumes=tuple(
                    normalize_artifact(str(value)) for value in item.get("consumes") or ()
                ),
                produces=tuple(
                    normalize_artifact(str(value)) for value in item.get("produces") or ()
                ),
                required_params=tuple(str(value) for value in item.get("required_params") or ()),
                replayable=bool(item.get("replayable", False)),
            )
            nodes[node.node_id] = node
        plan = cls(nodes=nodes)
        plan.validate()
        return plan

    def validate(self) -> None:
        for node in self.nodes.values():
            missing = set(node.parents) - self.nodes.keys()
            if missing:
                raise ValueError(f"Plan node {node.node_id} has missing parents: {sorted(missing)}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("Execution plan contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for parent in self.nodes[node_id].parents:
                visit(parent)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id)

    def bind(
        self,
        runtime_node: str,
        *,
        tool: str | None,
        consumes: tuple[str, ...],
        produces: tuple[str, ...],
    ) -> PlanNode | None:
        if runtime_node in self.bindings:
            return self.nodes[self.bindings[runtime_node]]
        consumed = set(consumes)
        produced = set(produces)
        bound = set(self.bindings.values())
        candidates = [
            node
            for node in self.nodes.values()
            if node.node_id not in bound
            and (node.tool is None or node.tool == tool)
            and (not node.consumes or bool(set(node.consumes) & consumed))
            and (not node.produces or bool(set(node.produces) & produced))
        ]
        if not candidates:
            candidates = [
                node
                for node in self.nodes.values()
                if node.node_id not in bound and (node.tool is None or node.tool == tool)
            ]
        if not candidates:
            return None
        declaration_order = {node_id: index for index, node_id in enumerate(self.nodes)}
        candidate = min(
            candidates,
            key=lambda node: (
                sum(parent not in bound for parent in node.parents),
                declaration_order[node.node_id],
            ),
        )
        self.bindings[runtime_node] = candidate.node_id
        return candidate

    def runtime_parents(self, runtime_node: str) -> tuple[str, ...]:
        plan_id = self.bindings.get(runtime_node)
        if plan_id is None:
            return ()
        reverse = {plan_node: runtime for runtime, plan_node in self.bindings.items()}
        return tuple(
            reverse[parent]
            for parent in self.nodes[plan_id].parents
            if parent in reverse
        )

    def bound_node(self, runtime_node: str) -> PlanNode | None:
        plan_id = self.bindings.get(runtime_node)
        return self.nodes.get(plan_id) if plan_id else None


class PlanContractMonitor:
    """Contracts compiled from the agent-declared execution DAG."""

    name = "plan_derived_contract_tree"

    def __init__(self) -> None:
        self.plan: ExecutionPlan | None = None

    def install(self, plan: ExecutionPlan) -> None:
        self.plan = plan

    def verify(
        self,
        event: ExecutionEvent,
        adapter: AgentExecutorAdapter,
    ) -> tuple[ContractViolation, ...]:
        _ = adapter
        if self.plan is None or event.boundary == BoundaryKind.NONE:
            return ()
        plan_id = self.plan.bindings.get(event.node_id)
        if plan_id is None:
            return ()
        node = self.plan.nodes[plan_id]
        violations: list[ContractViolation] = []
        tool = event.payload.get("tool_name")
        if node.tool and tool and node.tool != tool:
            violations.append(
                self._violation(event, node, "plan_tool_type", {"expected": node.tool, "actual": tool})
            )
        params = event.payload.get("params") or {}
        missing_params = [key for key in node.required_params if key not in params]
        if missing_params:
            violations.append(
                self._violation(event, node, "plan_required_params", {"missing": missing_params})
            )
        missing_inputs = sorted(set(node.consumes) - set(event.consumes))
        if missing_inputs:
            violations.append(
                self._violation(event, node, "plan_required_inputs", {"missing": missing_inputs})
            )
        if event.type == EventType.NODE_FINISH:
            missing_outputs = sorted(set(node.produces) - set(event.produces))
            if missing_outputs:
                violations.append(
                    self._violation(
                        event,
                        node,
                        "plan_required_outputs",
                        {"missing": missing_outputs},
                    )
                )
        return tuple(violations)

    @staticmethod
    def _violation(
        event: ExecutionEvent,
        node: PlanNode,
        contract_id: str,
        details: dict[str, Any],
    ) -> ContractViolation:
        return ContractViolation(
            contract_id=contract_id,
            node_id=event.node_id,
            boundary=event.boundary,
            responsible_nodes=(event.node_id,),
            # Agent-declared plans are useful provenance hypotheses, but they
            # are not deterministic evidence. Auto-synthesized tool/workspace
            # contracts remain the blocking execution authority.
            blocking=False,
            details={"plan_node": node.node_id, **details},
        )


class PlanContractCompiler:
    @staticmethod
    def compile(raw: dict[str, Any]) -> tuple[ExecutionPlan, PlanContractMonitor]:
        plan = ExecutionPlan.from_dict(raw)
        monitor = PlanContractMonitor()
        monitor.install(plan)
        return plan, monitor
