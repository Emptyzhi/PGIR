from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    TASK_START = "task_start"
    PLAN_DECLARED = "plan_declared"
    NODE_START = "node_start"
    NODE_FINISH = "node_finish"
    BEFORE_CONSUME = "before_consume"
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    FINAL_COMMIT = "final_commit"
    TASK_FINISH = "task_finish"


class BoundaryKind(str, Enum):
    NONE = "none"
    DEPENDENCY_CONSUMPTION = "dependency_consumption"
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    FINAL_COMMIT = "final_commit"


class ControlAction(str, Enum):
    CONTINUE = "continue"
    REPAIR_CONTINUE = "repair_continue"
    REPAIR_REPLAY = "repair_replay"
    GLOBAL_REPLAN = "global_replan"
    ABORT = "abort"


@dataclass(frozen=True)
class ExecutionEvent:
    type: EventType
    node_id: str
    boundary: BoundaryKind = BoundaryKind.NONE
    parents: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContractViolation:
    contract_id: str
    node_id: str
    boundary: BoundaryKind
    responsible_nodes: tuple[str, ...]
    blocking: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairPlan:
    frontier: tuple[str, ...]
    affected_subgraph: tuple[str, ...]
    replay_nodes: tuple[str, ...]
    global_replan: bool
    reason: str


@dataclass(frozen=True)
class ControlDecision:
    action: ControlAction
    violations: tuple[ContractViolation, ...] = ()
    plan: RepairPlan | None = None
    reason: str = ""
