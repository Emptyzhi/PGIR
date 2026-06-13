from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.agents.base import AgentTaskSpec
from src.runtime import (
    BoundaryKind,
    ContractViolation,
    EventType,
    ExecutionEvent,
    NoRepairController,
    OpenClawExecutorAdapter,
    OpenClawRuntimeBridge,
    PGIRController,
)


class OpenClawRuntimeTests(unittest.TestCase):
    def test_pgir_blocks_protected_mutation_at_before_tool_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter, bridge = self._runtime(root, PGIRController())
            self._emit_before_mutation(bridge.bridge_dir, "event-pgir")
            decision = self._wait_decision(bridge.decisions_dir / "event-pgir.json")
            bridge.stop()

            self.assertEqual(decision["action"], "block")
            self.assertEqual(decision["controller_action"], "repair_continue")
            self.assertEqual(adapter.report()["capabilities"]["live_tool_blocking"], True)
            self.assertTrue(
                any(op["operation"] == "block_tool_call_for_revision" for op in adapter.operations)
            )

    def test_pgir_blocks_direct_write_tool_to_protected_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _adapter, bridge = self._runtime(root, PGIRController())
            self._emit(
                bridge.bridge_dir,
                "event-write",
                "before_tool_call",
                "call-write",
                {"path": "/tmp_workspace/summary.md", "content": "replacement"},
                tool_name="write",
            )
            decision = self._wait_decision(bridge.decisions_dir / "event-write.json")
            self._emit(
                bridge.bridge_dir,
                "after-write",
                "after_tool_call",
                "call-write",
                {"path": "/tmp_workspace/summary.md", "content": "replacement"},
                error=decision["reason"],
                tool_name="write",
            )
            self._emit(
                bridge.bridge_dir,
                "event-safe",
                "before_tool_call",
                "call-safe",
                {"path": "/tmp_workspace/mae_summary.md", "content": "replacement"},
                tool_name="write",
            )
            safe_decision = self._wait_decision(bridge.decisions_dir / "event-safe.json")
            bridge.stop()
            self.assertEqual(decision["action"], "block")
            self.assertIn("distinct new output path", decision["reason"])
            self.assertEqual(safe_decision["action"], "continue")

    def test_no_repair_observes_same_boundary_but_allows_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter, bridge = self._runtime(root, NoRepairController())
            self._emit_before_mutation(bridge.bridge_dir, "event-control")
            decision = self._wait_decision(bridge.decisions_dir / "event-control.json")
            bridge.stop()

            self.assertEqual(decision["action"], "continue")
            self.assertEqual(adapter.tool_events[0]["kind"], "before_tool_call")
            self.assertFalse(adapter.operations)

    def test_parallel_tool_calls_remain_siblings_and_failure_repairs_at_next_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter, bridge = self._runtime(root, PGIRController())
            self._emit(
                bridge.bridge_dir,
                "before-a",
                "before_tool_call",
                "call-a",
                {"command": "echo a > /tmp_workspace/a.txt"},
            )
            self._wait_decision(bridge.decisions_dir / "before-a.json")
            self._emit(
                bridge.bridge_dir,
                "before-b",
                "before_tool_call",
                "call-b",
                {"command": "echo b"},
            )
            self._wait_decision(bridge.decisions_dir / "before-b.json")
            self._emit(
                bridge.bridge_dir,
                "after-a",
                "after_tool_call",
                "call-a",
                {"command": "echo a > /tmp_workspace/a.txt"},
                error="command failed",
            )
            self._emit(
                bridge.bridge_dir,
                "after-b",
                "after_tool_call",
                "call-b",
                {"command": "echo b"},
            )
            self._emit(
                bridge.bridge_dir,
                "before-c",
                "before_tool_call",
                "call-c",
                {"command": "cat /tmp_workspace/a.txt"},
            )
            decision = self._wait_decision(bridge.decisions_dir / "before-c.json")
            bridge.stop()

            self.assertEqual(
                adapter.tool_parents["tool:call-a"],
                adapter.tool_parents["tool:call-b"],
            )
            self.assertEqual(decision["action"], "block")
            self.assertIn("tool_execution_success", decision["reason"])

    def test_unrelated_tool_is_not_blocked_by_failed_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter, bridge = self._runtime(root, PGIRController())
            self._emit(
                bridge.bridge_dir,
                "before-a",
                "before_tool_call",
                "call-a",
                {"command": "echo a > /tmp_workspace/a.txt"},
            )
            self._wait_decision(bridge.decisions_dir / "before-a.json")
            self._emit(
                bridge.bridge_dir,
                "after-a",
                "after_tool_call",
                "call-a",
                {"command": "echo a > /tmp_workspace/a.txt"},
                error="command failed",
            )
            self._emit(
                bridge.bridge_dir,
                "before-unrelated",
                "before_tool_call",
                "call-unrelated",
                {"command": "echo unrelated"},
            )
            decision = self._wait_decision(bridge.decisions_dir / "before-unrelated.json")
            bridge.stop()

            self.assertEqual(decision["action"], "continue")
            self.assertEqual(len(adapter.pending_tool_failures), 1)

    def test_successful_semantic_retry_clears_pending_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter, bridge = self._runtime(root, PGIRController())
            failed_params = {"command": "echo a > /tmp_workspace/a.txt"}
            self._emit(
                bridge.bridge_dir,
                "before-a",
                "before_tool_call",
                "call-a",
                failed_params,
            )
            self._wait_decision(bridge.decisions_dir / "before-a.json")
            self._emit(
                bridge.bridge_dir,
                "after-a",
                "after_tool_call",
                "call-a",
                failed_params,
                error="command failed",
            )
            self._emit(
                bridge.bridge_dir,
                "before-retry",
                "before_tool_call",
                "call-retry",
                failed_params,
            )
            retry_decision = self._wait_decision(bridge.decisions_dir / "before-retry.json")
            self._emit(
                bridge.bridge_dir,
                "after-retry",
                "after_tool_call",
                "call-retry",
                failed_params,
            )
            bridge.stop()

            self.assertEqual(retry_decision["action"], "continue")
            self.assertEqual(adapter.pending_tool_failures, [])

    def test_openclaw_requests_affected_forward_replay(self) -> None:
        completed = SimpleNamespace(returncode=0, stderr="", stdout="")
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run", return_value=completed):
            root = Path(tmp)
            adapter, bridge = self._runtime(root, PGIRController())
            self._emit(
                bridge.bridge_dir,
                "before-a",
                "before_tool_call",
                "call-a",
                {"path": "/tmp_workspace/a.txt", "content": "a"},
                tool_name="write",
            )
            self._wait_decision(bridge.decisions_dir / "before-a.json")
            self._emit(
                bridge.bridge_dir,
                "after-a",
                "after_tool_call",
                "call-a",
                {"path": "/tmp_workspace/a.txt", "content": "a"},
                tool_name="write",
            )
            self._emit(
                bridge.bridge_dir,
                "before-b",
                "before_tool_call",
                "call-b",
                {"command": "cat /tmp_workspace/a.txt > /tmp_workspace/b.txt"},
                tool_name="exec",
            )
            self._wait_decision(bridge.decisions_dir / "before-b.json")
            self._emit(
                bridge.bridge_dir,
                "after-b",
                "after_tool_call",
                "call-b",
                {"command": "cat /tmp_workspace/a.txt > /tmp_workspace/b.txt"},
                tool_name="exec",
            )
            adapter.pending_tool_failures.append(
                ContractViolation(
                    contract_id="ancestor_invalid",
                    node_id="tool:call-a",
                    boundary=BoundaryKind.DEPENDENCY_CONSUMPTION,
                    responsible_nodes=("tool:call-a",),
                )
            )
            self._emit(
                bridge.bridge_dir,
                "before-c",
                "before_tool_call",
                "call-c",
                {"command": "echo c"},
            )
            decision = self._wait_decision(bridge.decisions_dir / "before-c.json")
            bridge.stop()

            self.assertEqual(decision["controller_action"], "repair_replay")
            self.assertIn("tool:call-b", decision["reason"])
            self.assertTrue(
                any(op["operation"] == "deterministic_replay" for op in adapter.operations)
            )

    def test_openclaw_requests_global_replan_when_root_is_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter, bridge = self._runtime(root, PGIRController())
            adapter.pending_tool_failures.append(
                ContractViolation(
                    contract_id="root_plan_invalid",
                    node_id="task_root",
                    boundary=BoundaryKind.DEPENDENCY_CONSUMPTION,
                    responsible_nodes=("task_root",),
                )
            )
            self._emit(
                bridge.bridge_dir,
                "before-plan",
                "before_tool_call",
                "call-plan",
                {"command": "echo plan"},
            )
            decision = self._wait_decision(bridge.decisions_dir / "before-plan.json")
            bridge.stop()

            self.assertEqual(decision["controller_action"], "global_replan")
            self.assertIn("Replace the root plan", decision["reason"])
            self.assertTrue(
                any(
                    op["operation"] == "request_model_mediated_global_replan"
                    for op in adapter.operations
                )
            )

    @staticmethod
    def _runtime(root: Path, controller):
        workspace = root / "workspace"
        exec_root = workspace / "exec"
        exec_root.mkdir(parents=True)
        (exec_root / "summary.md").write_text("original", encoding="utf-8")
        bridge_dir = root / "bridge"
        spec = AgentTaskSpec(
            task_id="test-openclaw",
            task={"pgir_require_plan": False},
            workspace_path=str(workspace),
            prompt="Create a separate report without changing existing files.",
            timeout_seconds=10,
            output_dir=root,
            model="fake",
        )
        adapter = OpenClawExecutorAdapter(spec, bridge_dir)
        bridge = OpenClawRuntimeBridge(bridge_dir, controller, adapter)
        bridge.start()
        controller.handle(ExecutionEvent(EventType.TASK_START, "task_root"), adapter)
        return adapter, bridge

    @staticmethod
    def _emit_before_mutation(bridge_dir: Path, event_id: str) -> None:
        OpenClawRuntimeTests._emit(
            bridge_dir,
            event_id,
            "before_tool_call",
            "call-1",
            {"command": "echo replacement > /tmp_workspace/summary.md"},
        )

    @staticmethod
    def _emit(
        bridge_dir: Path,
        event_id: str,
        kind: str,
        tool_call_id: str,
        params: dict,
        *,
        error: str | None = None,
        tool_name: str = "exec",
    ) -> None:
        record = {
            "event_id": event_id,
            "kind": kind,
            "event": {
                "toolName": tool_name,
                "toolCallId": tool_call_id,
                "params": params,
                "error": error,
            },
            "context": {"toolName": tool_name, "toolCallId": tool_call_id},
        }
        with (bridge_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    @staticmethod
    def _wait_decision(path: Path) -> dict:
        deadline = time.time() + 2
        while time.time() < deadline:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            time.sleep(0.02)
        raise AssertionError(f"Decision not written: {path}")


if __name__ == "__main__":
    unittest.main()
