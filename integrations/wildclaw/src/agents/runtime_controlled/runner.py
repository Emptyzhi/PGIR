from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent
from src.runtime import (
    BoundaryKind,
    EventType,
    ExecutionEvent,
    NoRepairController,
    PGIRController,
    RuntimeController,
    execution_failure_contract_tree,
)
from src.runtime.workspace import WorkspaceExecutorAdapter


CONDITIONS = {
    "pgir_hidden_taint_ancestor_repair",
    "pgir_no_provenance_taint",
    "local_leaf_retry_no_ancestor_control",
    "no_contract_retry_control",
    "full_trace_retry_control",
    "reflexion_verbal_retry",
    "no_repair_control",
}


class RuntimeControlledAgent(BaseAgent):
    """Run every repair condition through the same executor/control surface."""

    def __init__(self, delegate: BaseAgent, condition: str) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown runtime condition: {condition}")
        self.delegate = delegate
        self.condition = condition
        self._runtime: dict[str, tuple[RuntimeController, WorkspaceExecutorAdapter]] = {}

    @property
    def expects_gateway(self) -> bool:
        return self.delegate.expects_gateway

    @property
    def transcript_container_path(self) -> str:
        return self.delegate.transcript_container_path

    def prepare_grading_transcript(self, task_id: str) -> str:
        return self.delegate.prepare_grading_transcript(task_id)

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        controller = self._make_controller()
        patched = AgentTaskSpec(
            **{
                **spec.__dict__,
                "prompt": self._protocol_prompt(spec.prompt),
            }
        )
        bridge_report: dict[str, Any] = {}
        runtime_runner = getattr(self.delegate, "run_task_with_runtime", None)
        if callable(runtime_runner):
            execution, adapter, bridge_report = runtime_runner(patched, controller)
        else:
            adapter = WorkspaceExecutorAdapter(spec)
            controller.handle(
                ExecutionEvent(EventType.TASK_START, "task_root"),
                adapter,
            )
            execution = self.delegate.run_task(patched)
        self._runtime[spec.task_id] = (controller, adapter)
        controller.handle(
            ExecutionEvent(
                EventType.NODE_FINISH,
                "task_execution",
                parents=("task_root",),
            ),
            adapter,
        )
        decision = controller.handle(
            ExecutionEvent(
                EventType.FINAL_COMMIT,
                "final_commit",
                boundary=BoundaryKind.FINAL_COMMIT,
                parents=("task_execution",),
            ),
            adapter,
        )
        report = {
            "condition": self.condition,
            "decision": decision.action.value,
            "controller": (
                controller.report()
                if isinstance(controller, PGIRController)
                else {"controller": controller.name}
            ),
            "adapter": adapter.report(),
            "bridge": bridge_report,
        }
        (spec.output_dir / "runtime_control_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return execution

    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict[str, Any]:
        usage = self.delegate.collect_usage(task_id, output_dir, elapsed_time)
        usage["runtime_condition"] = self.condition
        controller = self._runtime.get(task_id, (None, None))[0]
        usage["runtime_controller"] = (
            controller.name if isinstance(controller, RuntimeController) else "unknown"
        )
        usage["runtime_backend_type"] = "executor_adapter_v1"
        return usage

    def _make_controller(self) -> RuntimeController:
        if self.condition == "pgir_hidden_taint_ancestor_repair":
            return PGIRController(variant=self.condition)
        if self.condition == "pgir_no_provenance_taint":
            return PGIRController(use_provenance=False, variant=self.condition)
        if self.condition == "local_leaf_retry_no_ancestor_control":
            return PGIRController(repair_policy="local_leaf", variant=self.condition)
        if self.condition == "no_contract_retry_control":
            return PGIRController(
                contracts=execution_failure_contract_tree(),
                variant=self.condition,
            )
        return NoRepairController()

    def _protocol_prompt(self, prompt: str) -> str:
        prefix = PROTOCOLS[self.condition].strip()
        return f"{prefix}\n\n{prompt}" if prefix else prompt


PROTOCOLS = {
    "no_repair_control": (
        ""
    ),
    "full_trace_retry_control": (
        "Method condition: full_trace_retry_control. Execute the task, inspect "
        "the complete trace and artifacts before finishing, then revise the full "
        "solution if a requirement is not satisfied."
    ),
    "reflexion_verbal_retry": (
        "Method condition: reflexion_verbal_retry. Attempt the task, privately "
        "reflect on likely failures, then apply the reflection to final artifacts."
    ),
    "pgir_hidden_taint_ancestor_repair": (
        ""
    ),
    "pgir_no_provenance_taint": (
        ""
    ),
    "local_leaf_retry_no_ancestor_control": (
        ""
    ),
    "no_contract_retry_control": (
        ""
    ),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
