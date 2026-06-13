from __future__ import annotations

import unittest

from eval.repair_frontier_experiment import evaluate_condition, generate_scenarios, summarize


class RepairFrontierExperimentTests(unittest.TestCase):
    def test_scenario_suite_covers_required_failure_structures(self) -> None:
        scenarios = generate_scenarios(seed=7, repeats=1)
        self.assertEqual(
            {scenario.category for scenario in scenarios},
            {
                "single_ancestor",
                "shared_ancestor",
                "independent_paths",
                "root_plan_failure",
                "surface_failure",
                "noisy_diagnosis",
            },
        )

    def test_pgir_batches_shared_ancestor_failures(self) -> None:
        scenario = next(
            scenario
            for scenario in generate_scenarios(seed=7, repeats=1)
            if scenario.category == "shared_ancestor"
        )
        pgir = evaluate_condition(scenario, "pgir")
        unbatched = evaluate_condition(scenario, "pgir_no_frontier_batching")
        self.assertEqual(pgir.repair_targets, ("source",))
        self.assertEqual(pgir.recovery_success, 1.0)
        self.assertLess(pgir.total_cost, unbatched.total_cost)

    def test_root_failure_structurally_escalates(self) -> None:
        scenario = next(
            scenario
            for scenario in generate_scenarios(seed=7, repeats=1)
            if scenario.category == "root_plan_failure"
        )
        pgir = evaluate_condition(scenario, "pgir")
        self.assertTrue(pgir.global_replan)
        self.assertEqual(set(pgir.repair_targets), set(scenario.nodes))

    def test_summary_reports_expected_structural_tradeoff(self) -> None:
        scenarios = generate_scenarios(seed=7, repeats=2)
        conditions = (
            "failure_site_retry",
            "diagnosis_full_replay",
            "full_trace_repair",
            "pgir_no_provenance",
            "pgir_no_frontier_batching",
            "pgir",
        )
        summary = summarize(
            [
                evaluate_condition(scenario, condition)
                for scenario in scenarios
                for condition in conditions
            ]
        )
        self.assertEqual(summary["overall"]["pgir"]["recovery_success"], 1.0)
        self.assertLess(
            summary["overall"]["pgir"]["total_cost"],
            summary["overall"]["full_trace_repair"]["total_cost"],
        )


if __name__ == "__main__":
    unittest.main()
