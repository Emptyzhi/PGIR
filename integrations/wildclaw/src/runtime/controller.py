from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .adapter import AgentExecutorAdapter
from .contracts import ContractTree, default_contract_tree
from .model import (
    BoundaryKind,
    ContractViolation,
    ControlAction,
    ControlDecision,
    ExecutionEvent,
    RepairPlan,
    EventType,
)
from .provenance import ProvenanceGraph
from .plan import ExecutionPlan, PlanContractMonitor


class RuntimeController:
    name = "runtime_controller"

    def handle(self, event: ExecutionEvent, adapter: AgentExecutorAdapter) -> ControlDecision:
        _ = (event, adapter)
        return ControlDecision(ControlAction.CONTINUE)


class NoRepairController(RuntimeController):
    name = "no_repair_control"


class PGIRController(RuntimeController):
    name = "pgir"

    def __init__(
        self,
        contracts: ContractTree | None = None,
        *,
        max_interventions: int = 8,
        use_provenance: bool = True,
        repair_policy: str = "ancestor_frontier",
        variant: str = "pgir_hidden_taint_ancestor_repair",
    ) -> None:
        if repair_policy not in {"ancestor_frontier", "local_leaf"}:
            raise ValueError(f"Unknown repair policy: {repair_policy}")
        self.contracts = contracts or default_contract_tree()
        self.provenance = ProvenanceGraph()
        self.max_interventions = max_interventions
        self.use_provenance = use_provenance
        self.repair_policy = repair_policy
        self.variant = variant
        self.interventions = 0
        self.snapshots: dict[str, Any] = {}
        self.decisions: list[ControlDecision] = []

    def handle(self, event: ExecutionEvent, adapter: AgentExecutorAdapter) -> ControlDecision:
        if event.type == EventType.PLAN_DECLARED:
            self._install_plan(event.payload.get("plan") or {}, adapter)
        self.provenance.record(event)
        if plan_node := event.payload.get("plan_node"):
            self.provenance.bind_plan_node(event.node_id, str(plan_node))
        if event.node_id not in self.snapshots:
            self.snapshots[event.node_id] = adapter.snapshot(event.node_id)
        if event.boundary == BoundaryKind.NONE:
            return self._record(ControlDecision(ControlAction.CONTINUE))

        violations = self.contracts.verify(event, adapter)
        blocking = tuple(violation for violation in violations if violation.blocking)
        if not blocking:
            return self._record(
                ControlDecision(
                    ControlAction.CONTINUE,
                    violations=violations,
                    reason="no execution-critical violation",
                )
            )
        if self.interventions >= self.max_interventions:
            return self._record(
                ControlDecision(
                    ControlAction.ABORT,
                    violations=blocking,
                    reason="engineering safeguard: intervention budget exhausted",
                )
            )

        responsible = {
            node
            for violation in blocking
            for node in violation.responsible_nodes
        }
        self.provenance.mark_tainted(responsible)
        observed = {violation.node_id for violation in blocking}
        if self.repair_policy == "local_leaf":
            frontier = observed
            affected = frontier & self.provenance.executed
            replay: set[str] = set()
        elif not self.use_provenance:
            frontier = responsible
            affected = frontier & self.provenance.executed
            replay = set()
        else:
            frontier = self.provenance.minimal_frontier(responsible)
            affected = self.provenance.affected_executed_subgraph(frontier)
            replay = affected - frontier
        is_local = self.provenance.is_strict_local_frontier(frontier)
        plan = RepairPlan(
            frontier=tuple(sorted(frontier)),
            affected_subgraph=tuple(sorted(affected)),
            replay_nodes=tuple(sorted(replay)),
            global_replan=not is_local,
            reason=(
                "strict local repair frontier exists"
                if is_local
                else "no strict local repair frontier exists"
            ),
        )

        adapter.pause()
        self.interventions += 1
        if plan.global_replan:
            ok = adapter.global_replan(plan.reason)
            adapter.resume()
            action = ControlAction.GLOBAL_REPLAN if ok else ControlAction.ABORT
            return self._record(
                ControlDecision(action, violations=blocking, plan=plan, reason=plan.reason)
            )

        checkpoint = adapter.select_repair_checkpoint(self.snapshots, plan.frontier)
        if not adapter.prepare_repair(checkpoint, plan.frontier, plan.affected_subgraph):
            adapter.resume()
            return self._record(
                ControlDecision(
                    ControlAction.ABORT,
                    violations=blocking,
                    plan=plan,
                    reason="frontier checkpoint could not be restored",
                )
            )
        repaired = adapter.repair_nodes(plan.frontier, blocking)
        if not repaired:
            adapter.resume()
            return self._record(
                ControlDecision(
                    ControlAction.ABORT,
                    violations=blocking,
                    plan=plan,
                    reason="frontier repair failed",
                )
            )
        self.snapshots.update(adapter.replay_snapshots())
        if plan.replay_nodes:
            replayed = adapter.replay_nodes(plan.replay_nodes)
            self.snapshots.update(adapter.replay_snapshots())
            if not replayed:
                escalated = adapter.global_replan(
                    "affected forward subgraph contains an action that cannot be deterministically replayed"
                )
                adapter.resume()
                return self._record(
                    ControlDecision(
                        ControlAction.GLOBAL_REPLAN if escalated else ControlAction.ABORT,
                        violations=blocking,
                        plan=plan,
                        reason=(
                            "deterministic replay was inadmissible; escalated to global replan"
                            if escalated
                            else "deterministic replay was inadmissible"
                        ),
                    )
                )
            adapter.resume()
            return self._record(
                ControlDecision(
                    ControlAction.REPAIR_REPLAY,
                    violations=blocking,
                    plan=plan,
                    reason="repaired frontier and replayed affected executed descendants",
                )
            )
        adapter.resume()
        return self._record(
            ControlDecision(
                ControlAction.REPAIR_CONTINUE,
                violations=blocking,
                plan=plan,
                reason="repaired before tainted output was consumed; no replay required",
            )
        )

    def report(self) -> dict[str, Any]:
        monitor_reports = {
            monitor.name: monitor.report()
            for monitor in self.contracts.monitors
            if callable(getattr(monitor, "report", None))
        }
        return {
            "controller": self.name,
            "variant": self.variant,
            "use_provenance": self.use_provenance,
            "repair_policy": self.repair_policy,
            "interventions": self.interventions,
            "decisions": [asdict(decision) for decision in self.decisions],
            "roots": sorted(self.provenance.roots),
            "executed": sorted(self.provenance.executed),
            "tainted": sorted(self.provenance.tainted),
            "contract_monitors": list(self.contracts.monitor_names),
            "contract_synthesis": monitor_reports,
        }

    def _record(self, decision: ControlDecision) -> ControlDecision:
        self.decisions.append(decision)
        return decision

    def _install_plan(self, raw: dict[str, Any], adapter: AgentExecutorAdapter) -> None:
        plan = ExecutionPlan.from_dict(raw)
        adapter.install_plan(plan)
        for monitor in self.contracts.monitors:
            if isinstance(monitor, PlanContractMonitor):
                monitor.install(plan)
        self.provenance.seed_plan(plan)
