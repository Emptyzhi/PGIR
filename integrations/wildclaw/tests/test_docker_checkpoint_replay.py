from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest
import uuid

from src.agents.base import AgentTaskSpec
from src.runtime import (
    AutoContractMonitor,
    BoundaryKind,
    ContractViolation,
    EventType,
    ExecutionEvent,
    PGIRController,
)
from src.runtime.openclaw import OpenClawExecutorAdapter


IMAGE = "wildclawbench-ubuntu:v1.3"


@unittest.skipUnless(shutil.which("docker"), "Docker is required")
class DockerCheckpointReplayTests(unittest.TestCase):
    def test_workspace_checkpoint_restore_and_deterministic_replay(self) -> None:
        if subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True
        ).returncode:
            self.skipTest(f"{IMAGE} is unavailable")
        name = f"pgir-runtime-test-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["docker", "run", "-d", "--name", name, IMAGE, "tail", "-f", "/dev/null"],
            check=True,
            capture_output=True,
        )
        try:
            subprocess.run(
                ["docker", "exec", name, "mkdir", "-p", "/tmp_workspace"],
                check=True,
            )
            subprocess.run(
                ["docker", "exec", name, "sh", "-lc", "printf before > /tmp_workspace/state.txt"],
                check=True,
            )
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "workspace" / "exec").mkdir(parents=True)
                spec = AgentTaskSpec(
                    task_id=name,
                    task={},
                    workspace_path=str(root / "workspace"),
                    prompt="test",
                    timeout_seconds=10,
                    output_dir=root,
                    model="fake",
                )
                adapter = OpenClawExecutorAdapter(spec, root / "bridge")
                adapter.require_declared_plan = False
                snapshot = adapter.snapshot("before-write")
                subprocess.run(
                    ["docker", "exec", name, "sh", "-lc", "printf changed > /tmp_workspace/state.txt"],
                    check=True,
                )
                adapter.restore(snapshot)
                self.assertEqual(self._read(name, "/tmp_workspace/state.txt"), "before")

                selective = adapter.snapshot("before-affected")
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        name,
                        "sh",
                        "-lc",
                        "printf affected > /tmp_workspace/affected.txt; "
                        "printf keep > /tmp_workspace/unaffected.txt",
                    ],
                    check=True,
                )
                adapter.recorded_actions = {
                    "tool:affected": {
                        "tool_name": "write",
                        "params": {
                            "path": "/tmp_workspace/affected.txt",
                            "content": "affected",
                        },
                        "produces": ("workspace:affected.txt",),
                        "replayable": True,
                        "finished": True,
                    }
                }
                self.assertTrue(
                    adapter.prepare_repair(
                        selective,
                        ("tool:affected",),
                        ("tool:affected",),
                    )
                )
                self.assertFalse(self._exists(name, "/tmp_workspace/affected.txt"))
                self.assertEqual(self._read(name, "/tmp_workspace/unaffected.txt"), "keep")

                adapter.execution_order = ["tool:write"]
                adapter.recorded_actions = {
                    "tool:write": {
                        "tool_name": "write",
                        "params": {
                            "path": "/tmp_workspace/generated.txt",
                            "content": "deterministic",
                        },
                        "replayable": True,
                        "finished": True,
                    }
                }
                self.assertTrue(adapter.replay_nodes(("tool:write",)))
                self.assertEqual(
                    self._read(name, "/tmp_workspace/generated.txt"),
                    "deterministic",
                )

                params = {
                    "command": "python3 -c \"open('/tmp_workspace/actual.txt','w').write('observed')\""
                }
                adapter.event_from_bridge(self._raw("before-actual", "before_tool_call", "actual", params))
                subprocess.run(
                    ["docker", "exec", name, "sh", "-lc", params["command"]],
                    check=True,
                )
                event = adapter.event_from_bridge(
                    self._raw("after-actual", "after_tool_call", "actual", params)
                )
                self.assertIn("workspace:actual.txt", event.produces)

                missing = adapter.event_from_bridge(
                    {
                        "event_id": "before-missing",
                        "kind": "before_tool_call",
                        "event": {
                            "toolName": "read",
                            "toolCallId": "missing",
                            "params": {"path": "/tmp_workspace/missing.txt"},
                        },
                        "context": {"toolName": "read", "toolCallId": "missing"},
                    }
                )
                violations = adapter.boundary_violations(missing)
                self.assertEqual(
                    violations[0].contract_id,
                    "artifact_input_availability",
                )
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    def test_controller_repairs_ancestor_and_replays_affected_forward_subgraph(self) -> None:
        if subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True
        ).returncode:
            self.skipTest(f"{IMAGE} is unavailable")
        name = f"pgir-controller-test-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["docker", "run", "-d", "--name", name, IMAGE, "tail", "-f", "/dev/null"],
            check=True,
            capture_output=True,
        )
        try:
            subprocess.run(["docker", "exec", name, "mkdir", "-p", "/tmp_workspace"], check=True)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "workspace" / "exec").mkdir(parents=True)
                spec = AgentTaskSpec(
                    task_id=name,
                    task={"pgir_require_plan": False},
                    workspace_path=str(root / "workspace"),
                    prompt="test",
                    timeout_seconds=10,
                    output_dir=root,
                    model="fake",
                )
                adapter = OpenClawExecutorAdapter(spec, root / "bridge")
                controller = PGIRController()
                controller.handle(ExecutionEvent(EventType.TASK_START, "task_root"), adapter)

                write = {"path": "/tmp_workspace/a.txt", "content": "repaired"}
                controller.handle(
                    adapter.event_from_bridge(
                        self._tool_raw("before-a", "before_tool_call", "a", write, "write")
                    ),
                    adapter,
                )
                subprocess.run(
                    ["docker", "exec", name, "sh", "-lc", "printf repaired > /tmp_workspace/a.txt"],
                    check=True,
                )
                controller.handle(
                    adapter.event_from_bridge(
                        self._tool_raw("after-a", "after_tool_call", "a", write, "write")
                    ),
                    adapter,
                )

                derive = {"command": "cat /tmp_workspace/a.txt > /tmp_workspace/b.txt"}
                decision_b = controller.handle(
                    adapter.event_from_bridge(
                        self._tool_raw("before-b", "before_tool_call", "b", derive, "exec")
                    ),
                    adapter,
                )
                self.assertEqual(decision_b.action.value, "continue", decision_b)
                subprocess.run(["docker", "exec", name, "sh", "-lc", derive["command"]], check=True)
                after_b = adapter.event_from_bridge(
                    self._tool_raw("after-b", "after_tool_call", "b", derive, "exec")
                )
                self.assertIsNotNone(after_b, adapter.operations)
                controller.handle(after_b, adapter)

                adapter.pending_tool_failures.append(
                    ContractViolation(
                        contract_id="ancestor_invalid",
                        node_id="tool:a",
                        boundary=BoundaryKind.DEPENDENCY_CONSUMPTION,
                        responsible_nodes=("tool:a",),
                    )
                )
                consume = {"command": "cat /tmp_workspace/b.txt > /tmp_workspace/c.txt"}
                decision = controller.handle(
                    adapter.event_from_bridge(
                        self._tool_raw("before-c", "before_tool_call", "c", consume, "exec")
                    ),
                    adapter,
                )

                self.assertEqual(decision.action.value, "repair_replay")
                self.assertEqual(self._read(name, "/tmp_workspace/a.txt"), "repaired")
                self.assertEqual(self._read(name, "/tmp_workspace/b.txt"), "repaired")
                self.assertFalse(self._exists(name, "/tmp_workspace/c.txt"))
                operations = [item["operation"] for item in adapter.operations]
                self.assertIn("repair_frontier", operations)
                self.assertIn("deterministic_replay", operations)
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    def test_synthesized_goal_contracts_verify_real_workspace_artifacts(self) -> None:
        if subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True
        ).returncode:
            self.skipTest(f"{IMAGE} is unavailable")
        name = f"pgir-goal-test-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["docker", "run", "-d", "--name", name, IMAGE, "tail", "-f", "/dev/null"],
            check=True,
            capture_output=True,
        )
        try:
            subprocess.run(["docker", "exec", name, "mkdir", "-p", "/tmp_workspace"], check=True)
            subprocess.run(
                [
                    "docker",
                    "exec",
                    name,
                    "sh",
                    "-lc",
                    "printf '%s' '%PDF-real' > /tmp_workspace/paper.pdf",
                ],
                check=True,
            )
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "workspace" / "exec").mkdir(parents=True)
                spec = AgentTaskSpec(
                    task_id=name,
                    task={"pgir_require_plan": False},
                    workspace_path=str(root / "workspace"),
                    prompt=(
                        "Download the paper to /tmp_workspace/paper.pdf and "
                        "write a report to /tmp_workspace/report.md."
                    ),
                    timeout_seconds=10,
                    output_dir=root,
                    model="fake",
                )
                adapter = OpenClawExecutorAdapter(spec, root / "bridge")
                monitor = AutoContractMonitor()
                final = ExecutionEvent(
                    EventType.FINAL_COMMIT,
                    "final",
                    boundary=BoundaryKind.FINAL_COMMIT,
                )
                violations = monitor.verify(final, adapter)
                self.assertEqual(
                    [violation.details["artifact"] for violation in violations],
                    ["workspace:report.md"],
                )
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        name,
                        "sh",
                        "-lc",
                        "printf report > /tmp_workspace/report.md",
                    ],
                    check=True,
                )
                self.assertEqual(monitor.verify(final, adapter), ())
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    @staticmethod
    def _read(container: str, path: str) -> str:
        return subprocess.run(
            ["docker", "exec", container, "cat", path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    @staticmethod
    def _exists(container: str, path: str) -> bool:
        return (
            subprocess.run(
                ["docker", "exec", container, "test", "-e", path],
                capture_output=True,
            ).returncode
            == 0
        )

    @staticmethod
    def _raw(event_id: str, kind: str, call: str, params: dict) -> dict:
        return DockerCheckpointReplayTests._tool_raw(event_id, kind, call, params, "exec")

    @staticmethod
    def _tool_raw(event_id: str, kind: str, call: str, params: dict, tool: str) -> dict:
        return {
            "event_id": event_id,
            "kind": kind,
            "event": {"toolName": tool, "toolCallId": call, "params": params},
            "context": {"toolName": tool, "toolCallId": call},
        }


if __name__ == "__main__":
    unittest.main()
