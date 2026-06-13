from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.agents.base import AgentTaskSpec
from src.runtime import BoundaryKind, ContractViolation, EventType, ExecutionEvent, PGIRController
from src.runtime.artifacts import tool_artifacts
from src.runtime.openclaw import OpenClawExecutorAdapter
from src.runtime.plan import ExecutionPlan, PlanContractMonitor


class PlanAndProvenanceTests(unittest.TestCase):
    def test_plan_is_declared_bound_and_verified_on_real_tool_events(self) -> None:
        adapter, controller = self._runtime()
        plan = {
            "nodes": [
                {
                    "id": "produce",
                    "tool": "write",
                    "parents": [],
                    "consumes": [],
                    "produces": ["workspace:a.txt"],
                    "required_params": ["path", "content"],
                },
                {
                    "id": "consume",
                    "tool": "read",
                    "parents": ["produce"],
                    "consumes": ["workspace:a.txt"],
                    "produces": [],
                    "required_params": ["path"],
                },
            ]
        }
        declared = adapter.event_from_bridge(
            self._raw("plan", "after_tool_call", "plan-call", {"nodes": plan["nodes"]}, "pgir_declare_plan")
        )
        self.assertEqual(declared.type, EventType.PLAN_DECLARED)
        controller.handle(declared, adapter)

        before_a = adapter.event_from_bridge(
            self._raw(
                "before-a",
                "before_tool_call",
                "a",
                {"path": "/tmp_workspace/a.txt", "content": "a"},
                "write",
            )
        )
        decision_a = controller.handle(before_a, adapter)
        self.assertEqual(decision_a.violations, ())
        controller.handle(
            adapter.event_from_bridge(
                self._raw(
                    "after-a",
                    "after_tool_call",
                    "a",
                    {"path": "/tmp_workspace/a.txt", "content": "a"},
                    "write",
                )
            ),
            adapter,
        )
        before_b = adapter.event_from_bridge(
            self._raw(
                "before-b",
                "before_tool_call",
                "b",
                {"path": "/tmp_workspace/a.txt"},
                "read",
            )
        )
        decision_b = controller.handle(before_b, adapter)

        self.assertEqual(decision_b.violations, ())
        self.assertEqual(adapter.plan.bindings["tool:a"], "produce")
        self.assertEqual(adapter.plan.bindings["tool:b"], "consume")
        self.assertEqual(adapter.tool_parents["tool:b"], ("tool:a",))
        self.assertEqual(adapter.artifact_producers["workspace:a.txt"], "tool:a")

    def test_pgir_requires_plan_before_first_task_tool(self) -> None:
        adapter, controller = self._runtime()
        event = adapter.event_from_bridge(
            self._raw(
                "before-a",
                "before_tool_call",
                "a",
                {"path": "/tmp_workspace/a.txt", "content": "a"},
                "write",
            )
        )
        decision = controller.handle(event, adapter)
        self.assertEqual(decision.action.value, "repair_continue")
        self.assertEqual(decision.violations[0].contract_id, "execution_plan_declared")

    def test_independent_actions_are_not_linked_by_time(self) -> None:
        adapter, _controller = self._runtime()
        first = adapter.event_from_bridge(
            self._raw(
                "before-a",
                "before_tool_call",
                "a",
                {"path": "/tmp_workspace/a.txt", "content": "a"},
                "write",
            )
        )
        adapter.event_from_bridge(
            self._raw(
                "after-a",
                "after_tool_call",
                "a",
                {"path": "/tmp_workspace/a.txt", "content": "a"},
                "write",
            )
        )
        second = adapter.event_from_bridge(
            self._raw(
                "before-b",
                "before_tool_call",
                "b",
                {"path": "/tmp_workspace/b.txt", "content": "b"},
                "write",
            )
        )
        self.assertEqual(first.parents, ("task_root",))
        self.assertEqual(second.parents, ("task_root",))

    def test_fan_in_comes_from_two_artifact_producers(self) -> None:
        adapter, _controller = self._runtime()
        for call in ("a", "b"):
            params = {"path": f"/tmp_workspace/{call}.txt", "content": call}
            adapter.event_from_bridge(self._raw(f"before-{call}", "before_tool_call", call, params, "write"))
            adapter.event_from_bridge(self._raw(f"after-{call}", "after_tool_call", call, params, "write"))
        event = adapter.event_from_bridge(
            self._raw(
                "before-c",
                "before_tool_call",
                "c",
                {
                    "command": "cat /tmp_workspace/a.txt /tmp_workspace/b.txt > /tmp_workspace/c.txt"
                },
                "exec",
            )
        )
        self.assertEqual(set(event.parents), {"tool:a", "tool:b"})
        self.assertEqual(event.boundary.value, "fan_in")

    def test_deterministic_replay_rejects_external_actions(self) -> None:
        adapter, _controller = self._runtime()
        adapter.execution_order = ["tool:write", "tool:web"]
        adapter.recorded_actions = {
            "tool:write": {
                "tool_name": "write",
                "params": {"path": "/tmp_workspace/a.txt", "content": "a"},
                "replayable": True,
                "finished": True,
            },
            "tool:web": {
                "tool_name": "web_fetch",
                "params": {"url": "https://example.com"},
                "replayable": False,
                "finished": True,
            },
        }
        completed = SimpleNamespace(returncode=0, stderr="", stdout="")
        with patch("subprocess.run", return_value=completed):
            self.assertTrue(adapter.replay_nodes(("tool:write",)))
            self.assertFalse(adapter.replay_nodes(("tool:web",)))
        self.assertFalse(adapter.operations[-1]["supported"])
        self.assertEqual(adapter.operations[-1]["unreplayable"], ["tool:web"])

    def test_multi_frontier_repair_uses_earliest_execution_checkpoint(self) -> None:
        adapter, _controller = self._runtime()
        adapter.execution_order = ["tool:z", "tool:a"]
        snapshots = {
            "tool:a": {"label": "later"},
            "tool:z": {"label": "earlier"},
        }
        selected = adapter.select_repair_checkpoint(
            snapshots,
            ("tool:a", "tool:z"),
        )
        self.assertEqual(selected["label"], "earlier")

    def test_artifact_normalization_matches_plan_and_tools(self) -> None:
        plan = ExecutionPlan.from_dict(
            {
                "nodes": [
                    {
                        "id": "n",
                        "tool": "read",
                        "parents": [],
                        "consumes": ["/tmp_workspace/a.txt"],
                        "produces": [],
                    }
                ]
            }
        )
        consumes, _ = tool_artifacts("read", {"path": "/tmp_workspace/a.txt"})
        self.assertEqual(plan.nodes["n"].consumes, consumes)

    def test_shell_redirect_is_output_not_input_with_optional_space(self) -> None:
        consumes, produces = tool_artifacts(
            "exec",
            {"command": "cat /tmp_workspace/a.txt > /tmp_workspace/b.txt"},
        )
        self.assertEqual(consumes, ("workspace:a.txt",))
        self.assertEqual(produces, ("workspace:b.txt",))

    def test_plan_output_contract_detects_missing_artifact(self) -> None:
        plan = ExecutionPlan.from_dict(
            {
                "nodes": [
                    {
                        "id": "n",
                        "tool": "write",
                        "parents": [],
                        "consumes": [],
                        "produces": ["workspace:required.txt"],
                    }
                ]
            }
        )
        plan.bindings["tool:n"] = "n"
        monitor = PlanContractMonitor()
        monitor.install(plan)
        adapter, _controller = self._runtime()
        violations = monitor.verify(
            ExecutionEvent(
                EventType.NODE_FINISH,
                "tool:n",
                boundary=BoundaryKind.FAN_OUT,
                produces=(),
                payload={"tool_name": "write", "params": {}},
            ),
            adapter,
        )
        self.assertEqual(violations[-1].contract_id, "plan_required_outputs")
        self.assertFalse(violations[-1].blocking)

    def test_logical_plan_artifact_links_successful_parent_to_child(self) -> None:
        adapter, controller = self._runtime()
        plan = {
            "nodes": [
                {
                    "id": "search",
                    "tool": "exec",
                    "parents": [],
                    "consumes": [],
                    "produces": ["paper_url"],
                },
                {
                    "id": "download",
                    "tool": "exec",
                    "parents": ["search"],
                    "consumes": ["paper_url"],
                    "produces": ["workspace:paper.pdf"],
                },
            ]
        }
        controller.handle(
            adapter.event_from_bridge(
                self._raw(
                    "plan",
                    "after_tool_call",
                    "plan-call",
                    {"nodes": plan["nodes"]},
                    "pgir_declare_plan",
                )
            ),
            adapter,
        )
        before_search = adapter.event_from_bridge(
            self._raw("before-search", "before_tool_call", "search", {"command": "echo url"}, "exec")
        )
        self.assertEqual(controller.handle(before_search, adapter).violations, ())
        controller.handle(
            adapter.event_from_bridge(
                self._raw("after-search", "after_tool_call", "search", {"command": "echo url"}, "exec")
            ),
            adapter,
        )
        before_download = adapter.event_from_bridge(
            self._raw(
                "before-download",
                "before_tool_call",
                "download",
                {"command": "echo pdf > /tmp_workspace/paper.pdf"},
                "exec",
            )
        )
        decision = controller.handle(before_download, adapter)

        self.assertEqual(decision.violations, ())
        self.assertEqual(before_download.parents, ("tool:search",))
        self.assertEqual(adapter.artifact_producers["paper_url"], "tool:search")

    def test_replacement_plan_clears_stale_execution_failure(self) -> None:
        adapter, controller = self._runtime()
        adapter.pending_tool_failures.append(
            ContractViolation(
                contract_id="tool_execution_success",
                node_id="tool:old",
                boundary=BoundaryKind.DEPENDENCY_CONSUMPTION,
                responsible_nodes=("tool:old",),
            )
        )
        replacement = adapter.event_from_bridge(
            self._raw(
                "plan",
                "after_tool_call",
                "plan-call",
                {
                    "nodes": [
                        {
                            "id": "replacement",
                            "tool": "exec",
                            "parents": [],
                            "consumes": [],
                            "produces": [],
                        }
                    ]
                },
                "pgir_declare_plan",
            )
        )
        controller.handle(replacement, adapter)
        self.assertEqual(adapter.pending_tool_failures, [])

    def test_unplanned_runtime_action_is_recorded_but_not_blocked(self) -> None:
        adapter, controller = self._runtime()
        controller.handle(
            adapter.event_from_bridge(
                self._raw(
                    "plan",
                    "after_tool_call",
                    "plan-call",
                    {
                        "nodes": [
                            {
                                "id": "planned_write",
                                "tool": "write",
                                "parents": [],
                                "consumes": [],
                                "produces": ["workspace:out.txt"],
                            }
                        ]
                    },
                    "pgir_declare_plan",
                )
            ),
            adapter,
        )
        event = adapter.event_from_bridge(
            self._raw("before-check", "before_tool_call", "check", {"command": "ls /tmp_workspace"}, "exec")
        )
        decision = controller.handle(event, adapter)
        self.assertEqual(decision.action.value, "continue")
        self.assertEqual(decision.violations[0].contract_id, "plan_node_binding")
        self.assertFalse(decision.violations[0].blocking)

    def test_plan_rejects_cycles(self) -> None:
        with self.assertRaises(ValueError):
            ExecutionPlan.from_dict(
                {
                    "nodes": [
                        {"id": "a", "parents": ["b"]},
                        {"id": "b", "parents": ["a"]},
                    ]
                }
            )

    def _runtime(self) -> tuple[OpenClawExecutorAdapter, PGIRController]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        workspace = root / "workspace"
        (workspace / "exec").mkdir(parents=True)
        spec = AgentTaskSpec(
            task_id="test-plan-runtime",
            task={"pgir_require_plan": True},
            workspace_path=str(workspace),
            prompt="test",
            timeout_seconds=10,
            output_dir=root,
            model="fake",
        )
        adapter = OpenClawExecutorAdapter(spec, root / "bridge")
        controller = PGIRController()
        controller.handle(ExecutionEvent(EventType.TASK_START, "task_root"), adapter)
        return adapter, controller

    @staticmethod
    def _raw(event_id: str, kind: str, call: str, params: dict, tool: str) -> dict:
        return {
            "event_id": event_id,
            "kind": kind,
            "event": {"toolName": tool, "toolCallId": call, "params": params},
            "context": {"toolName": tool, "toolCallId": call},
        }


if __name__ == "__main__":
    unittest.main()
