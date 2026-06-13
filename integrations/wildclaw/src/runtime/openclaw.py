from __future__ import annotations

import base64
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from src.agents.base import AgentTaskSpec

from .controller import RuntimeController
from .artifacts import is_concrete_artifact, tool_artifacts
from .model import (
    BoundaryKind,
    ContractViolation,
    ControlAction,
    ControlDecision,
    EventType,
    ExecutionEvent,
)
from .plan import ExecutionPlan
from .semantics import ToolSemanticRegistry
from .workspace import WorkspaceExecutorAdapter


class OpenClawExecutorAdapter(WorkspaceExecutorAdapter):
    """Translate OpenClaw plugin hooks into executor-independent runtime events."""

    def __init__(self, spec: AgentTaskSpec, bridge_dir: Path) -> None:
        super().__init__(spec)
        self.bridge_dir = bridge_dir
        self.require_declared_plan = bool(spec.task.get("pgir_require_plan", False))
        self.active_bridge_event: dict[str, Any] | None = None
        self.tool_events: list[dict[str, Any]] = []
        self.tool_nodes: dict[str, str] = {}
        self.tool_parents: dict[str, tuple[str, ...]] = {}
        self.plan: ExecutionPlan | None = None
        self.artifact_producers: dict[str, str] = {}
        self.recorded_actions: dict[str, dict[str, Any]] = {}
        self.pre_action_manifests: dict[str, dict[str, str] | None] = {}
        self.execution_order: list[str] = []
        self.finished_nodes: list[str] = []
        self._replay_snapshots: dict[str, Any] = {}
        self.pending_tool_failures: list[ContractViolation] = []
        self.controller_blocked_nodes: set[str] = set()
        self.current_boundary_node: str | None = None
        self.semantic_registry = ToolSemanticRegistry()

    def event_from_bridge(self, raw: dict[str, Any]) -> ExecutionEvent | None:
        kind = str(raw.get("kind", ""))
        event = dict(raw.get("event") or {})
        context = dict(raw.get("context") or {})
        self.active_bridge_event = raw
        self.tool_events.append(
            {
                "event_id": raw.get("event_id"),
                "kind": kind,
                "tool_name": event.get("toolName") or context.get("toolName"),
                "tool_call_id": event.get("toolCallId") or context.get("toolCallId"),
            }
        )

        if kind == "before_tool_call":
            node = self._tool_node(event, context)
            self.current_boundary_node = node
            tool_name = str(event.get("toolName") or context.get("toolName") or "")
            params = dict(event.get("params") or {})
            effect = self.semantic_registry.analyze(tool_name, params)
            consumes, produces = effect.consumes, effect.produces
            plan_node = (
                self.plan.bind(
                    node,
                    tool=tool_name,
                    consumes=consumes,
                    produces=produces,
                )
                if self.plan is not None and tool_name != "pgir_declare_plan"
                else None
            )
            if plan_node is not None:
                consumes = tuple(
                    sorted(
                        set(consumes)
                        | {
                            artifact
                            for artifact in plan_node.consumes
                            if not is_concrete_artifact(artifact)
                        }
                    )
                )
            parents = self._dependency_parents(node, consumes)
            self.tool_parents[node] = parents
            self.pre_action_manifests[node] = self.capture_workspace_manifest()
            self.recorded_actions[node] = {
                "tool_name": tool_name,
                "params": params,
                "consumes": consumes,
                "produces": produces,
                "expected_produces": produces,
                "replayable": self._is_replayable(tool_name, params, plan_node),
                "semantic_effect": effect,
                "plan_node": plan_node.node_id if plan_node else None,
                "finished": False,
            }
            boundary = BoundaryKind.FAN_IN if len(parents) > 1 else BoundaryKind.DEPENDENCY_CONSUMPTION
            return ExecutionEvent(
                EventType.FAN_IN if boundary == BoundaryKind.FAN_IN else EventType.BEFORE_CONSUME,
                node,
                boundary=boundary,
                parents=parents,
                consumes=consumes,
                payload={
                    "tool_name": tool_name,
                    "params": params,
                    "plan_node": plan_node.node_id if plan_node else None,
                    "expected_produces": produces,
                    "bridge_event_id": raw.get("event_id"),
                },
            )
        if kind == "after_tool_call":
            node = self._tool_node(event, context)
            if node in self.controller_blocked_nodes:
                self.controller_blocked_nodes.discard(node)
                self.operations.append(
                    {
                        "operation": "observe_controller_block_result",
                        "node": node,
                    }
                )
                return None
            tool_name = str(event.get("toolName") or context.get("toolName") or "")
            params = dict(event.get("params") or {})
            if tool_name == "pgir_declare_plan":
                if self._tool_failed(event):
                    self.operations.append(
                        {
                            "operation": "plan_declaration_failed",
                            "error": event.get("error"),
                        }
                    )
                    return None
                plan = self._declared_plan(params, event.get("result"))
                return ExecutionEvent(
                    EventType.PLAN_DECLARED,
                    node,
                    payload={"plan": plan, "tool_name": tool_name, "params": params},
                )
            parents = self.tool_parents.get(node, ("task_root",))
            effect = self.semantic_registry.analyze(tool_name, params)
            consumes = effect.consumes
            _, result_produces = tool_artifacts(tool_name, params, result=event.get("result"))
            inferred_produces = tuple(sorted(set(effect.produces) | set(result_produces)))
            bound_plan_node = (
                self.plan.bound_node(node) if self.plan is not None else None
            )
            if bound_plan_node is not None:
                consumes = tuple(
                    sorted(
                        set(consumes)
                        | {
                            artifact
                            for artifact in bound_plan_node.consumes
                            if not is_concrete_artifact(artifact)
                        }
                    )
                )
            produces = self._observed_produces(node, inferred_produces)
            action = self.recorded_actions.setdefault(
                node,
                {
                    "tool_name": tool_name,
                    "params": params,
                    "consumes": consumes,
                    "produces": produces,
                    "replayable": self._is_replayable(tool_name, params, None),
                    "plan_node": None,
                },
            )
            failed = self._tool_failed(event)
            action.update(
                {
                    "consumes": consumes,
                    "produces": produces,
                    "finished": True,
                    "failed": failed,
                }
            )
            if bound_plan_node is not None and not failed:
                logical_outputs = {
                    artifact
                    for artifact in bound_plan_node.produces
                    if not is_concrete_artifact(artifact)
                }
                produces = tuple(sorted(set(produces) | logical_outputs))
                action.update({"produces": produces})
            for artifact in produces:
                self.artifact_producers[artifact] = node
            if node not in self.execution_order:
                self.execution_order.append(node)
            if node not in self.finished_nodes:
                self.finished_nodes.append(node)
            if failed:
                violation = ContractViolation(
                    contract_id="tool_execution_success",
                    node_id=node,
                    boundary=BoundaryKind.DEPENDENCY_CONSUMPTION,
                    responsible_nodes=(node,),
                    blocking=True,
                    details={
                        "tool_name": tool_name,
                        "error": event.get("error"),
                        "plan_node": action.get("plan_node"),
                        "expected_produces": list(
                            self.recorded_actions.get(node, {}).get("expected_produces", ())
                        ),
                    },
                )
                if violation not in self.pending_tool_failures:
                    self.pending_tool_failures.append(violation)
            else:
                self._clear_resolved_failures(node)
            return ExecutionEvent(
                EventType.NODE_FINISH,
                node,
                boundary=BoundaryKind.FAN_OUT,
                parents=parents,
                consumes=consumes,
                produces=produces,
                payload={
                    "tool_name": tool_name,
                    "params": params,
                    "plan_node": action.get("plan_node"),
                    "error": event.get("error"),
                    "duration_ms": event.get("durationMs"),
                },
            )
        if kind == "agent_end":
            return ExecutionEvent(
                EventType.TASK_FINISH,
                "task_execution",
                boundary=BoundaryKind.FINAL_COMMIT,
                parents=tuple(self.finished_nodes[-1:]) or ("task_root",),
                payload={"success": event.get("success"), "error": event.get("error")},
            )
        return None

    def install_plan(self, plan: Any) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("OpenClaw runtime requires an ExecutionPlan")
        self.plan = plan
        self.pending_tool_failures.clear()
        self.controller_blocked_nodes.clear()
        self.operations.append(
            {"operation": "install_plan", "nodes": sorted(plan.nodes)}
        )

    def protected_mutation_paths(self, event: ExecutionEvent) -> tuple[str, ...]:
        if event.boundary not in {BoundaryKind.DEPENDENCY_CONSUMPTION, BoundaryKind.FAN_IN}:
            return ()
        return self._protected_mutation_paths(event)

    def boundary_violations(self, event: ExecutionEvent) -> tuple[ContractViolation, ...]:
        if event.type not in {EventType.BEFORE_CONSUME, EventType.FAN_IN}:
            return ()
        tool_name = str(event.payload.get("tool_name") or "")
        if tool_name == "pgir_declare_plan":
            return ()
        if self.plan is None and self.require_declared_plan:
            return (
                ContractViolation(
                    contract_id="execution_plan_declared",
                    node_id=event.node_id,
                    boundary=event.boundary,
                    responsible_nodes=(event.node_id,),
                    blocking=True,
                    details={"reason": "declare the execution DAG before using task tools"},
                ),
            )
        if self.plan is not None and event.payload.get("plan_node") is None:
            return (
                ContractViolation(
                    contract_id="plan_node_binding",
                    node_id=event.node_id,
                    boundary=event.boundary,
                    responsible_nodes=(event.node_id,),
                    blocking=False,
                    details={
                        "reason": (
                            "tool action is not represented in the declared execution DAG; "
                            "recorded as a runtime graph extension"
                        )
                    },
                ),
            )
        if self.plan is not None:
            plan_node = self.plan.bound_node(event.node_id)
            missing_logical = sorted(
                artifact
                for artifact in (plan_node.consumes if plan_node else ())
                if not is_concrete_artifact(artifact)
                and (
                    (producer := self.artifact_producers.get(artifact)) is None
                    or producer not in self.finished_nodes
                )
            )
            if missing_logical:
                return (
                    ContractViolation(
                        contract_id="logical_artifact_availability",
                        node_id=event.node_id,
                        boundary=event.boundary,
                        responsible_nodes=tuple(
                            sorted(
                                {
                                    self.artifact_producers.get(artifact, event.node_id)
                                    for artifact in missing_logical
                                }
                            )
                        ),
                        blocking=False,
                        details={"missing": missing_logical},
                    ),
                )
        manifest = self.pre_action_manifests.get(event.node_id)
        if manifest is not None:
            missing = sorted(
                artifact
                for artifact in event.consumes
                if artifact.startswith("workspace:")
                and artifact.removeprefix("workspace:") not in manifest
            )
            if missing:
                responsible = tuple(
                    sorted(
                        {
                            self.artifact_producers.get(artifact, event.node_id)
                            for artifact in missing
                        }
                    )
                )
                return (
                    ContractViolation(
                        contract_id="artifact_input_availability",
                        node_id=event.node_id,
                        boundary=event.boundary,
                        responsible_nodes=responsible,
                        blocking=True,
                        details={"missing": missing},
                    ),
                )
        return ()

    def pending_execution_failures(self, event: ExecutionEvent) -> tuple[ContractViolation, ...]:
        if event.boundary not in {BoundaryKind.DEPENDENCY_CONSUMPTION, BoundaryKind.FAN_IN}:
            return ()
        if event.payload.get("tool_name") == "pgir_declare_plan":
            return ()
        return tuple(
            violation
            for violation in self.pending_tool_failures
            if self._failure_relevant_to_event(violation, event)
        )

    def repair_nodes(self, node_ids: tuple[str, ...], violations: tuple[Any, ...]) -> bool:
        if any(
            getattr(violation, "contract_id", "") == "preserve_preexisting_workspace_files"
            and getattr(violation, "boundary", BoundaryKind.NONE) == BoundaryKind.FINAL_COMMIT
            for violation in violations
        ):
            return super().repair_nodes(node_ids, violations)
        failed_nodes = tuple(
            node for node in node_ids if self.recorded_actions.get(node, {}).get("failed")
        )
        past_nodes = tuple(
            node
            for node in node_ids
            if self.recorded_actions.get(node, {}).get("finished")
            and node not in failed_nodes
        )
        if past_nodes:
            repaired = self._replay_actions(past_nodes, operation="repair_frontier")
            if not repaired:
                return False
        self.operations.append(
            {
                "operation": (
                    "request_model_mediated_failed_tool_repair"
                    if failed_nodes
                    else "block_tool_call_for_revision"
                ),
                "nodes": list(node_ids),
                "contracts": [
                    getattr(violation, "contract_id", "unknown") for violation in violations
                ],
            }
        )
        self.controller_blocked_nodes.update(
            node for node in node_ids if node.startswith("tool:")
        )
        if self.current_boundary_node:
            self.controller_blocked_nodes.add(self.current_boundary_node)
        repaired_ids = set(node_ids)
        self.pending_tool_failures = [
            violation
            for violation in self.pending_tool_failures
            if not (
                set(violation.responsible_nodes) & repaired_ids
                and not (
                    set(violation.responsible_nodes) & set(failed_nodes)
                )
            )
        ]
        return True

    def _failure_relevant_to_event(
        self,
        violation: ContractViolation,
        event: ExecutionEvent,
    ) -> bool:
        if violation.contract_id != "tool_execution_success":
            return True
        failed_nodes = set(violation.responsible_nodes)
        if event.node_id in failed_nodes:
            return False
        if self._is_retry_for_failure(event.node_id, violation):
            return False
        if failed_nodes & set(event.parents):
            return True
        expected = set(violation.details.get("expected_produces") or ())
        return bool(expected & set(event.consumes))

    def _is_retry_for_failure(
        self,
        candidate_node: str,
        violation: ContractViolation,
    ) -> bool:
        candidate = self.recorded_actions.get(candidate_node, {})
        failed_node = next(iter(violation.responsible_nodes), "")
        failed = self.recorded_actions.get(failed_node, {})
        if not candidate or not failed:
            return False
        candidate_plan = candidate.get("plan_node")
        failed_plan = failed.get("plan_node")
        if candidate_plan and candidate_plan == failed_plan:
            return True
        candidate_outputs = set(candidate.get("expected_produces") or candidate.get("produces") or ())
        failed_outputs = set(failed.get("expected_produces") or failed.get("produces") or ())
        if candidate_outputs and failed_outputs and candidate_outputs & failed_outputs:
            return True
        return (
            candidate.get("tool_name") == failed.get("tool_name")
            and candidate.get("params") == failed.get("params")
        )

    def _clear_resolved_failures(self, successful_node: str) -> None:
        self.pending_tool_failures = [
            violation
            for violation in self.pending_tool_failures
            if not self._is_retry_for_failure(successful_node, violation)
        ]

    def replay_nodes(self, node_ids: tuple[str, ...]) -> bool:
        return self._replay_actions(node_ids, operation="deterministic_replay")

    def global_replan(self, reason: str) -> bool:
        self.operations.append(
            {
                "operation": "request_model_mediated_global_replan",
                "reason": reason,
            }
        )
        if self.current_boundary_node:
            self.controller_blocked_nodes.add(self.current_boundary_node)
        return True

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["adapter"] = "openclaw"
        report["capabilities"].update(
            {
                "tool_events": True,
                "dependency_consumption_boundaries": True,
                "live_tool_blocking": True,
                "selective_replay": True,
                "selective_replay_mode": "deterministic_recorded_actions",
                "global_replan": True,
                "global_replan_mode": "model_mediated",
                "plan_derived_contracts": True,
                "artifact_level_provenance": True,
            }
        )
        report["tool_event_count"] = len(self.tool_events)
        report["tool_events"] = self.tool_events
        report["plan_nodes"] = sorted(self.plan.nodes) if self.plan else []
        report["plan_bindings"] = dict(self.plan.bindings) if self.plan else {}
        report["artifact_producers"] = dict(self.artifact_producers)
        report["recorded_actions"] = {
            node: {
                "tool_name": action["tool_name"],
                "consumes": list(action.get("consumes", ())),
                "produces": list(action.get("produces", ())),
                "replayable": action.get("replayable", False),
                "plan_node": action.get("plan_node"),
                "finished": action.get("finished", False),
            }
            for node, action in self.recorded_actions.items()
        }
        return report

    def prepare_repair(
        self,
        snapshot: Any,
        node_ids: tuple[str, ...],
        affected_nodes: tuple[str, ...] = (),
    ) -> bool:
        if not any(self.recorded_actions.get(node, {}).get("finished") for node in node_ids):
            return True
        if not snapshot or not snapshot.get("checkpoint"):
            self.operations.append(
                {
                    "operation": "prepare_repair",
                    "nodes": list(node_ids),
                    "restored": False,
                    "reason": "checkpoint unavailable for executed frontier",
                }
            )
            return False
        artifacts = sorted(
            {
                artifact.removeprefix("workspace:")
                for node in affected_nodes
                for artifact in self.recorded_actions.get(node, {}).get("produces", ())
                if artifact.startswith("workspace:")
            }
        )
        if not artifacts:
            self.operations.append(
                {
                    "operation": "prepare_repair",
                    "nodes": list(node_ids),
                    "affected_nodes": list(affected_nodes),
                    "restored": True,
                    "artifacts": [],
                }
            )
            return True
        checkpoint = snapshot["checkpoint"]
        script = (
            "from pathlib import Path; import json,shutil,sys; "
            "src=Path(sys.argv[1]); dst=Path('/tmp_workspace'); "
            "items=json.loads(sys.argv[2]); "
            "\nfor rel in items:\n"
            " s=src/rel; d=dst/rel\n"
            " if d.exists() or d.is_symlink():\n"
            "  shutil.rmtree(d) if d.is_dir() and not d.is_symlink() else d.unlink()\n"
            " if s.exists() or s.is_symlink():\n"
            "  d.parent.mkdir(parents=True,exist_ok=True)\n"
            "  shutil.copytree(s,d,symlinks=True) if s.is_dir() else shutil.copy2(s,d)\n"
        )
        proc = subprocess.run(
            [
                "docker",
                "exec",
                self.spec.task_id,
                "python3",
                "-c",
                script,
                checkpoint,
                json.dumps(artifacts),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.operations.append(
            {
                "operation": "prepare_repair",
                "nodes": list(node_ids),
                "affected_nodes": list(affected_nodes),
                "restored": proc.returncode == 0,
                "artifacts": artifacts,
                "stderr": proc.stderr[-500:],
            }
        )
        return proc.returncode == 0

    def replay_snapshots(self) -> dict[str, Any]:
        snapshots = dict(self._replay_snapshots)
        self._replay_snapshots.clear()
        return snapshots

    def select_repair_checkpoint(
        self,
        snapshots: dict[str, Any],
        node_ids: tuple[str, ...],
    ) -> Any:
        positions = {node: index for index, node in enumerate(self.execution_order)}
        candidates = [
            (positions.get(node, len(positions)), snapshots[node])
            for node in node_ids
            if node in snapshots
        ]
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _dependency_parents(
        self,
        runtime_node: str,
        consumes: tuple[str, ...],
    ) -> tuple[str, ...]:
        parents = {
            producer
            for artifact in consumes
            if (producer := self.artifact_producers.get(artifact)) is not None
        }
        if self.plan is not None:
            parents.update(self.plan.runtime_parents(runtime_node))
        return tuple(sorted(parents)) or ("task_root",)

    def _observed_produces(
        self,
        node: str,
        inferred: tuple[str, ...],
    ) -> tuple[str, ...]:
        before = self.pre_action_manifests.get(node)
        after = self.capture_workspace_manifest()
        if before is None or after is None:
            return inferred
        changed = {
            f"workspace:{path}"
            for path, digest in after.items()
            if before.get(path) != digest
        }
        verified_inferred = {
            artifact
            for artifact in inferred
            if not artifact.startswith("workspace:")
            or artifact.removeprefix("workspace:") in after
        }
        return tuple(sorted(changed | verified_inferred))

    def _replay_actions(self, node_ids: tuple[str, ...], *, operation: str) -> bool:
        selected = set(node_ids)
        ordered = [node for node in self.execution_order if node in selected]
        missing = selected - set(ordered)
        inadmissible = [
            node
            for node in ordered
            if not self.recorded_actions.get(node, {}).get("replayable", False)
        ]
        if missing or inadmissible:
            self.operations.append(
                {
                    "operation": operation,
                    "nodes": ordered,
                    "supported": False,
                    "missing": sorted(missing),
                    "unreplayable": inadmissible,
                }
            )
            return False
        results: list[dict[str, Any]] = []
        for node in ordered:
            action = self.recorded_actions[node]
            self._replay_snapshots[node] = self.snapshot(node)
            result = self._execute_recorded_action(action)
            results.append({"node": node, **result})
            if not result["ok"]:
                self.operations.append(
                    {
                        "operation": operation,
                        "nodes": ordered,
                        "supported": True,
                        "results": results,
                    }
                )
                return False
        self.operations.append(
            {
                "operation": operation,
                "nodes": ordered,
                "supported": True,
                "results": results,
            }
        )
        return True

    def _execute_recorded_action(self, action: dict[str, Any]) -> dict[str, Any]:
        tool = str(action["tool_name"]).lower()
        params = dict(action["params"])
        if tool in {"read", "image"}:
            return {"ok": True, "mode": "read_noop"}
        if tool == "write":
            path = _path_param(params)
            content = params.get("content", params.get("text", ""))
            if not path or not isinstance(content, str):
                return {"ok": False, "mode": "write", "error": "missing path/content"}
            encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
            script = (
                "from pathlib import Path; import base64,sys; "
                "p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); "
                "p.write_bytes(base64.b64decode(sys.argv[2]))"
            )
            return self._docker_python(script, path, encoded, mode="write")
        if tool == "edit":
            path = _path_param(params)
            old = params.get("oldText", params.get("old_string"))
            new = params.get("newText", params.get("new_string"))
            if not path or not isinstance(old, str) or not isinstance(new, str):
                return {"ok": False, "mode": "edit", "error": "missing exact edit fields"}
            script = (
                "from pathlib import Path; import base64,sys; "
                "p=Path(sys.argv[1]); old=base64.b64decode(sys.argv[2]).decode(); "
                "new=base64.b64decode(sys.argv[3]).decode(); text=p.read_text(); "
                "assert old in text, 'old text absent'; p.write_text(text.replace(old,new,1))"
            )
            return self._docker_python(
                script,
                path,
                base64.b64encode(old.encode()).decode(),
                base64.b64encode(new.encode()).decode(),
                mode="edit",
            )
        if tool == "exec":
            command = _command_text(params)
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.spec.task_id,
                    "/bin/bash",
                    "-lc",
                    f"cd /tmp_workspace && {command}",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return {"ok": proc.returncode == 0, "mode": "exec", "stderr": proc.stderr[-500:]}
        return {"ok": False, "mode": tool, "error": "unsupported replay tool"}

    def _docker_python(self, script: str, *args: str, mode: str) -> dict[str, Any]:
        proc = subprocess.run(
            ["docker", "exec", self.spec.task_id, "python3", "-c", script, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return {"ok": proc.returncode == 0, "mode": mode, "stderr": proc.stderr[-500:]}

    @staticmethod
    def _is_replayable(tool_name: str, params: dict[str, Any], plan_node: Any) -> bool:
        if plan_node is not None and not plan_node.replayable:
            return False
        return ToolSemanticRegistry().analyze(tool_name, params).replayable

    @staticmethod
    def _declared_plan(params: dict[str, Any], result: Any) -> dict[str, Any]:
        if isinstance(params.get("plan"), dict):
            return dict(params["plan"])
        if isinstance(params.get("nodes"), list):
            return {"nodes": params["nodes"]}
        if isinstance(result, dict):
            details = result.get("details")
            if isinstance(details, dict) and isinstance(details.get("plan"), dict):
                return dict(details["plan"])
        raise ValueError("pgir_declare_plan did not contain a structured plan")

    def _tool_node(self, event: dict[str, Any], context: dict[str, Any]) -> str:
        tool_call_id = str(event.get("toolCallId") or context.get("toolCallId") or "")
        if tool_call_id:
            return self.tool_nodes.setdefault(tool_call_id, f"tool:{tool_call_id}")
        event_id = str(self.active_bridge_event.get("event_id")) if self.active_bridge_event else "unknown"
        return f"tool:{event_id}"

    def _protected_mutation_paths(self, event: ExecutionEvent) -> tuple[str, ...]:
        if not self.guard_enabled:
            return ()
        params = event.payload.get("params") or {}
        tool_name = str(event.payload.get("tool_name") or "").lower()
        command = _command_text(params)
        referenced_artifacts = self._referenced_artifacts(params)
        mutates_directly = tool_name in {"write", "edit", "apply_patch"}
        if not mutates_directly and (not command or not _looks_mutating(command)):
            return ()
        referenced = [
            rel
            for rel in self.seed_snapshot
            if (
                rel.lower() in command.lower()
                or f"/tmp_workspace/{rel}".lower() in command.lower()
                or any(path.lower().endswith(f"/{rel.lower()}") for path in referenced_artifacts)
            )
        ]
        if not referenced:
            return ()
        return tuple(referenced)

    @staticmethod
    def _tool_failed(event: dict[str, Any]) -> bool:
        if event.get("error"):
            return True
        result = event.get("result")
        if not isinstance(result, dict):
            return False
        if result.get("isError") is True:
            return True
        details = result.get("details")
        return isinstance(details, dict) and details.get("exitCode") not in (None, 0)

    @classmethod
    def _produced_artifacts(cls, event: dict[str, Any]) -> tuple[str, ...]:
        tool_name = str(event.get("toolName") or "").lower()
        params = event.get("params") or {}
        if tool_name in {"write", "edit", "apply_patch"}:
            return cls._referenced_artifacts(params)
        command = _command_text(params)
        if tool_name == "exec" and command:
            return tuple(sorted(set(_redirect_targets(command))))
        return ()

    @staticmethod
    def _referenced_artifacts(value: Any) -> tuple[str, ...]:
        text = json.dumps(value, ensure_ascii=False, default=str)
        paths = re.findall(r"(?:/tmp_workspace/|/root/.openclaw/workspace/)[^\s\"']+", text)
        return tuple(sorted(set(paths)))


class OpenClawRuntimeBridge:
    """Host-side event loop for the file-backed OpenClaw plugin bridge."""

    def __init__(
        self,
        bridge_dir: Path,
        controller: RuntimeController,
        adapter: OpenClawExecutorAdapter,
    ) -> None:
        self.bridge_dir = bridge_dir
        self.controller = controller
        self.adapter = adapter
        self.events_path = bridge_dir / "events.jsonl"
        self.decisions_dir = bridge_dir / "decisions"
        self._offset = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []

    def start(self) -> None:
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="openclaw-runtime-bridge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.process_pending()

    def process_pending(self) -> None:
        if not self.events_path.exists():
            return
        with self.events_path.open("r", encoding="utf-8") as stream:
            stream.seek(self._offset)
            while line := stream.readline():
                self._offset = stream.tell()
                try:
                    self._process(json.loads(line))
                except Exception as exc:  # bridge failures must not strand the executor
                    self.errors.append(str(exc))

    def report(self) -> dict[str, Any]:
        return {"bridge": "file_jsonl_v1", "errors": self.errors}

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            self.process_pending()

    def _process(self, raw: dict[str, Any]) -> None:
        event = self.adapter.event_from_bridge(raw)
        decision = (
            self.controller.handle(event, self.adapter)
            if event is not None
            else ControlDecision(ControlAction.CONTINUE)
        )
        if raw.get("kind") == "before_tool_call":
            self._write_decision(str(raw["event_id"]), decision)

    def _write_decision(self, event_id: str, decision: ControlDecision) -> None:
        block = decision.action in {
            ControlAction.ABORT,
            ControlAction.REPAIR_CONTINUE,
            ControlAction.REPAIR_REPLAY,
            ControlAction.GLOBAL_REPLAN,
        }
        reason = decision.reason
        if decision.violations:
            contracts = ", ".join(v.contract_id for v in decision.violations)
            reason = (
                f"Runtime contract violation ({contracts}). "
                f"{self._repair_guidance(decision)} {reason}"
            )
        payload = {
            "action": "block" if block else "continue",
            "controller_action": decision.action.value,
            "reason": reason,
        }
        target = self.decisions_dir / f"{event_id}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    @staticmethod
    def _repair_guidance(decision: ControlDecision) -> str:
        protected = sorted(
            {
                path
                for violation in decision.violations
                if violation.contract_id == "preserve_preexisting_workspace_files"
                for path in violation.details.get("paths", ())
            }
        )
        if protected:
            return (
                f"Do not modify the pre-existing path(s): {', '.join(protected)}. "
                "Use a distinct new output path and continue."
            )
        if decision.action == ControlAction.REPAIR_REPLAY and decision.plan is not None:
            return (
                "Repair the responsible frontier, then re-execute only the affected "
                f"forward nodes: {', '.join(decision.plan.replay_nodes)}."
            )
        if decision.action == ControlAction.GLOBAL_REPLAN:
            return "The failure is not strictly localizable. Replace the root plan before continuing."
        if any(v.contract_id == "tool_execution_success" for v in decision.violations):
            return "Repair or retry the failed upstream action before consuming its result."
        return "Revise this tool call before continuing."


def _command_text(params: dict[str, Any]) -> str:
    for key in ("command", "cmd", "script"):
        value = params.get(key)
        if isinstance(value, str):
            return value
    return ""


def _looks_mutating(command: str) -> bool:
    lowered = command.lower()
    patterns = (
        r"(?:^|[;&|]\s*)(?:rm|mv|cp|sed\s+-i|truncate)\b",
        r"(?:^|\s)(?:>|>>)\s*\S+",
        r"\b(?:write_text|write_bytes|unlink|rename|replace)\s*\(",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _redirect_targets(command: str) -> tuple[str, ...]:
    return tuple(
        match.rstrip(";|&")
        for match in re.findall(
            r"(?:>|>>|-o\s+)(/tmp_workspace/[^\s\"']+)",
            command,
            flags=re.IGNORECASE,
        )
    )


def _path_param(params: dict[str, Any]) -> str:
    for key in ("path", "file_path", "filePath", "target", "destination", "output"):
        value = params.get(key)
        if isinstance(value, str):
            return value
    return ""


def _has_external_side_effect(command: str) -> bool:
    lowered = command.lower()
    patterns = (
        r"\b(?:curl|wget|ssh|scp|rsync|git\s+(?:push|pull|clone|fetch)|pip\s+install|npm\s+install|apt(?:-get)?|docker)\b",
        r"\b(?:shutdown|reboot|kill|pkill|taskkill)\b",
        r"(?:^|\s)(?:https?|ftp)://",
        r"(?:^|\s)/(?!tmp_workspace/)",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)
