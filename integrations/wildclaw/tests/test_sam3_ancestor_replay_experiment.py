from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from eval.sam3_ancestor_replay_experiment import (
    BUGGY_EXPRESSION,
    FIXED_EXPRESSION,
    run_failure_site_retry,
    run_full_trace,
    run_pgir,
)


class Sam3AncestorReplayExperimentTests(unittest.TestCase):
    def test_real_task_shape_proves_ancestor_replay_tradeoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "box_ops.py"
            source.write_text(
                "import torch\n"
                "def box_cxcywh_to_xyxy(x):\n"
                "    x_c, y_c, w, h = x.unbind(-1)\n"
                f"    {BUGGY_EXPRESSION}\n"
                "    return torch.stack(b, dim=-1)\n",
                encoding="utf-8",
            )
            failure_site = run_failure_site_retry(source, root)
            full_trace = run_full_trace(source, root)
            pgir = run_pgir(source, root)

        self.assertFalse(failure_site.recovery_success)
        self.assertTrue(full_trace.recovery_success)
        self.assertTrue(pgir.recovery_success)
        self.assertTrue(pgir.root_repaired)
        self.assertEqual(pgir.controller_action, "repair_replay")
        self.assertEqual(pgir.frontier, ("10_box_ops_root",))
        self.assertEqual(
            set(pgir.affected_subgraph),
            {
                "10_box_ops_root",
                "30_decoder_path",
                "31_geometry_encoder_path",
                "32_processor_path",
                "33_visualization_path",
            },
        )
        self.assertEqual(len(pgir.replayed_nodes), 4)
        self.assertEqual(
            set(pgir.preserved_nodes),
            {"00_task_context", "20_unrelated_image_metadata"},
        )
        self.assertLess(pgir.total_recovery_scope, full_trace.total_recovery_scope)

    def test_fixture_exposes_expected_injected_bug(self) -> None:
        fixture = (
            Path(__file__).resolve().parents[1]
            / "workspace/02_Code_Intelligence/task_2_sam3_debug/exec"
            / "sam3/sam3/model/box_ops.py"
        )
        text = fixture.read_text(encoding="utf-8")
        self.assertIn(BUGGY_EXPRESSION, text)
        self.assertNotIn(FIXED_EXPRESSION, text)


if __name__ == "__main__":
    unittest.main()
