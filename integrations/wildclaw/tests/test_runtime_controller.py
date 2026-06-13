from __future__ import annotations

import unittest
from typing import Any

from src.runtime import (
    AgentExecutorAdapter,
    BoundaryKind,
    ControlAction,
    EventType,
    ExecutionEvent,
    NoRepairController,
    PGIRController,
)


class RecordingAdapter(AgentExecutorAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def pause(self) -> None:
        self.calls.append(("pause", None))

    def resume(self) -> None:
        self.calls.append(("resume", None))

    def snapshot(self, label: str) -> Any:
        self.calls.append(("snapshot", label))
        return {"label": label}

    def restore(self, snapshot: Any) -> None:
        self.calls.append(("restore", snapshot))

    def repair_nodes(self, node_ids: tuple[str, ...], violations: tuple[Any, ...]) -> bool:
        self.calls.append(("repair", node_ids))
        return True

    def replay_nodes(self, node_ids: tuple[str, ...]) -> bool:
        self.calls.append(("replay", node_ids))
        return True

    def global_replan(self, reason: str) -> bool:
        self.calls.append(("global_replan", reason))
        return True

    def boundary_violations(self, event: ExecutionEvent) -> tuple[Any, ...]:
        return tuple(event.payload.get("violations", ()))


class UnreplayableAdapter(RecordingAdapter):
    def replay_nodes(self, node_ids: tuple[str, ...]) -> bool:
        self.calls.append(("replay", node_ids))
        return False


def finish(node: str, *parents: str) -> ExecutionEvent:
    return ExecutionEvent(
        EventType.NODE_FINISH,
        node,
        parents=parents,
        produces=(f"{node}.out",),
    )


class PGIRControllerTests(unittest.TestCase):
    def test_pre_consumption_repair_does_not_replay(self) -> None:
        adapter = RecordingAdapter()
        controller = PGIRController()
        controller.handle(finish("plan"), adapter)
        controller.handle(finish("A", "plan"), adapter)

        decision = controller.handle(
            ExecutionEvent(
                EventType.BEFORE_CONSUME,
                "B",
                boundary=BoundaryKind.DEPENDENCY_CONSUMPTION,
                parents=("A",),
                consumes=("A.out",),
                payload={
                    "violations": [
                        {
                            "contract_id": "a-output-valid",
                            "responsible_nodes": ["A"],
                        }
                    ]
                },
            ),
            adapter,
        )

        self.assertEqual(decision.action, ControlAction.REPAIR_CONTINUE)
        self.assertEqual(decision.plan.frontier, ("A",))
        self.assertEqual(decision.plan.replay_nodes, ())
        self.assertIn(("repair", ("A",)), adapter.calls)
        self.assertFalse(any(call[0] == "replay" for call in adapter.calls))

    def test_fan_in_replays_only_affected_executed_descendants(self) -> None:
        adapter = RecordingAdapter()
        controller = PGIRController()
        for event in (
            finish("plan"),
            finish("A", "plan"),
            finish("B", "A"),
            finish("C", "plan"),
        ):
            controller.handle(event, adapter)

        decision = controller.handle(
            ExecutionEvent(
                EventType.FAN_IN,
                "D",
                boundary=BoundaryKind.FAN_IN,
                parents=("B", "C"),
                consumes=("B.out", "C.out"),
                payload={
                    "violations": [
                        {
                            "contract_id": "aggregate-input-valid",
                            "responsible_nodes": ["A"],
                        }
                    ]
                },
            ),
            adapter,
        )

        self.assertEqual(decision.action, ControlAction.REPAIR_REPLAY)
        self.assertEqual(decision.plan.frontier, ("A",))
        self.assertEqual(decision.plan.affected_subgraph, ("A", "B"))
        self.assertEqual(decision.plan.replay_nodes, ("B",))
        self.assertNotIn("C", decision.plan.affected_subgraph)

    def test_root_frontier_structurally_escalates_to_global_replan(self) -> None:
        adapter = RecordingAdapter()
        controller = PGIRController()
        controller.handle(finish("plan"), adapter)
        controller.handle(finish("A", "plan"), adapter)

        decision = controller.handle(
            ExecutionEvent(
                EventType.FINAL_COMMIT,
                "commit",
                boundary=BoundaryKind.FINAL_COMMIT,
                parents=("A",),
                payload={
                    "violations": [
                        {
                            "contract_id": "plan-invalid",
                            "responsible_nodes": ["plan"],
                        }
                    ]
                },
            ),
            adapter,
        )

        self.assertEqual(decision.action, ControlAction.GLOBAL_REPLAN)
        self.assertTrue(decision.plan.global_replan)
        self.assertTrue(any(call[0] == "global_replan" for call in adapter.calls))

    def test_unreplayable_affected_subgraph_escalates_to_global_replan(self) -> None:
        adapter = UnreplayableAdapter()
        controller = PGIRController()
        for event in (finish("plan"), finish("A", "plan"), finish("B", "A")):
            controller.handle(event, adapter)
        decision = controller.handle(
            ExecutionEvent(
                EventType.BEFORE_CONSUME,
                "C",
                boundary=BoundaryKind.DEPENDENCY_CONSUMPTION,
                parents=("B",),
                payload={
                    "violations": [
                        {
                            "contract_id": "ancestor-invalid",
                            "responsible_nodes": ["A"],
                        }
                    ]
                },
            ),
            adapter,
        )
        self.assertEqual(decision.action, ControlAction.GLOBAL_REPLAN)
        self.assertIn(("replay", ("B",)), adapter.calls)
        self.assertTrue(any(call[0] == "global_replan" for call in adapter.calls))

    def test_no_repair_controller_uses_same_event_surface_without_intervention(self) -> None:
        adapter = RecordingAdapter()
        controller = NoRepairController()
        decision = controller.handle(
            ExecutionEvent(
                EventType.FINAL_COMMIT,
                "commit",
                boundary=BoundaryKind.FINAL_COMMIT,
                payload={"violations": [{"contract_id": "failure"}]},
            ),
            adapter,
        )
        self.assertEqual(decision.action, ControlAction.CONTINUE)
        self.assertEqual(adapter.calls, [])

    def test_no_provenance_repairs_responsible_node_without_descendant_replay(self) -> None:
        adapter = RecordingAdapter()
        controller = PGIRController(use_provenance=False, variant="no_provenance")
        for event in (finish("plan"), finish("A", "plan"), finish("B", "A")):
            controller.handle(event, adapter)
        decision = controller.handle(
            ExecutionEvent(
                EventType.FAN_IN,
                "D",
                boundary=BoundaryKind.FAN_IN,
                parents=("B",),
                payload={
                    "violations": [
                        {"contract_id": "ancestor-invalid", "responsible_nodes": ["A"]}
                    ]
                },
            ),
            adapter,
        )
        self.assertEqual(decision.plan.frontier, ("A",))
        self.assertEqual(decision.plan.replay_nodes, ())
        self.assertFalse(any(call[0] == "replay" for call in adapter.calls))

    def test_local_leaf_repairs_observed_failure_site(self) -> None:
        adapter = RecordingAdapter()
        controller = PGIRController(repair_policy="local_leaf", variant="local_leaf")
        for event in (finish("plan"), finish("A", "plan"), finish("B", "A")):
            controller.handle(event, adapter)
        decision = controller.handle(
            ExecutionEvent(
                EventType.FAN_IN,
                "D",
                boundary=BoundaryKind.FAN_IN,
                parents=("B",),
                payload={
                    "violations": [
                        {"contract_id": "ancestor-invalid", "responsible_nodes": ["A"]}
                    ]
                },
            ),
            adapter,
        )
        self.assertEqual(decision.plan.frontier, ("D",))
        self.assertEqual(decision.plan.replay_nodes, ())
        self.assertIn(("repair", ("D",)), adapter.calls)


if __name__ == "__main__":
    unittest.main()
