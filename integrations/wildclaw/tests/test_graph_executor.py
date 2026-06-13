from __future__ import annotations

import unittest

from src.runtime import GraphExecutorAdapter, GraphNode, PGIRController


class GraphExecutorTests(unittest.TestCase):
    def test_pre_consumption_frontier_repair_continues_without_replay(self) -> None:
        nodes = [
            GraphNode("plan", lambda _: "plan"),
            GraphNode(
                "A",
                lambda _: "bad",
                parents=("plan",),
                repair=lambda _old, _violations: "good",
            ),
            GraphNode(
                "B",
                lambda inputs: f"B:{inputs['A']}",
                parents=("A",),
                validate_before_consume=lambda inputs: (
                    ()
                    if inputs["A"] == "good"
                    else (
                        {
                            "contract_id": "A-valid",
                            "responsible_nodes": ["A"],
                        },
                    )
                ),
            ),
        ]
        adapter = GraphExecutorAdapter(nodes)
        outputs = adapter.run(PGIRController())

        self.assertEqual(outputs["A"], "good")
        self.assertEqual(outputs["B"], "B:good")
        self.assertFalse(any(op["operation"] == "replay_nodes" for op in adapter.operations))

    def test_fan_in_repair_replays_only_affected_forward_subgraph(self) -> None:
        nodes = [
            GraphNode("plan", lambda _: "plan"),
            GraphNode(
                "A",
                lambda _: 1,
                parents=("plan",),
                repair=lambda _old, _violations: 10,
            ),
            GraphNode("B", lambda inputs: inputs["A"] * 2, parents=("A",)),
            GraphNode("C", lambda _inputs: 5, parents=("plan",)),
            GraphNode(
                "D",
                lambda inputs: inputs["B"] + inputs["C"],
                parents=("B", "C"),
                validate_before_consume=lambda inputs: (
                    ()
                    if inputs["B"] >= 20
                    else (
                        {
                            "contract_id": "B-large-enough",
                            "responsible_nodes": ["A"],
                        },
                    )
                ),
            ),
        ]
        adapter = GraphExecutorAdapter(nodes)
        outputs = adapter.run(PGIRController())

        self.assertEqual(outputs["A"], 10)
        self.assertEqual(outputs["B"], 20)
        self.assertEqual(outputs["C"], 5)
        self.assertEqual(outputs["D"], 25)
        replayed = [
            op["node"] for op in adapter.operations if op["operation"] == "replay_node"
        ]
        self.assertEqual(replayed, ["B"])


if __name__ == "__main__":
    unittest.main()
