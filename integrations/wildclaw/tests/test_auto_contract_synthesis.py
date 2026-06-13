from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.agents.base import AgentTaskSpec
from src.runtime import (
    AutoContractMonitor,
    BoundaryKind,
    EventType,
    ExecutionEvent,
    ToolSemanticRegistry,
    synthesize_goal_contracts,
)
from src.runtime.openclaw import OpenClawExecutorAdapter


class AutoContractSynthesisTests(unittest.TestCase):
    def test_registry_infers_pdf_pipeline_without_declared_plan(self) -> None:
        registry = ToolSemanticRegistry()
        download = registry.analyze(
            "exec",
            {
                "command": (
                    "curl -L https://example.com/paper.pdf "
                    "-o /tmp_workspace/paper.pdf"
                )
            },
        )
        extract = registry.analyze(
            "exec",
            {
                "command": (
                    "pdftotext /tmp_workspace/paper.pdf "
                    "/tmp_workspace/paper.txt"
                )
            },
        )

        self.assertIn("url:https://example.com/paper.pdf", download.consumes)
        self.assertIn("workspace:paper.pdf", download.produces)
        self.assertEqual(extract.consumes, ("workspace:paper.pdf",))
        self.assertEqual(extract.produces, ("workspace:paper.txt",))
        self.assertFalse(download.replayable)
        self.assertTrue(extract.replayable)

    def test_write_content_does_not_create_false_provenance_artifacts(self) -> None:
        effect = ToolSemanticRegistry().analyze(
            "write",
            {
                "path": "/tmp_workspace/summary.md",
                "content": (
                    "Source: https://arxiv.org/abs/2111.06377; "
                    "downloaded as /tmp_workspace/MAE.pdf"
                ),
            },
        )
        self.assertEqual(effect.produces, ("workspace:summary.md",))
        self.assertEqual(effect.mutates, ("workspace:summary.md",))

    def test_shell_control_tokens_are_not_artifacts(self) -> None:
        effect = ToolSemanticRegistry().analyze(
            "exec",
            {
                "command": (
                    "pdftotext /tmp_workspace/MAE.pdf || "
                    "apt-get install -y poppler-utils"
                )
            },
        )
        self.assertNotIn("||", effect.consumes)
        self.assertNotIn("apt-get", effect.produces)

    def test_required_params_are_synthesized_and_blocking(self) -> None:
        monitor = AutoContractMonitor()
        event = ExecutionEvent(
            EventType.BEFORE_CONSUME,
            "tool:write",
            boundary=BoundaryKind.DEPENDENCY_CONSUMPTION,
            payload={"tool_name": "write", "params": {"path": "/tmp_workspace/out.md"}},
        )
        violations = monitor.verify(event, SimpleNamespace())
        self.assertEqual(violations[0].contract_id, "auto_required_tool_params")
        self.assertEqual(violations[0].details["missing"], ["content"])
        self.assertTrue(violations[0].blocking)

    def test_missing_expected_output_is_detected_after_successful_tool(self) -> None:
        monitor = AutoContractMonitor()
        before = ExecutionEvent(
            EventType.BEFORE_CONSUME,
            "tool:download",
            boundary=BoundaryKind.DEPENDENCY_CONSUMPTION,
            payload={
                "tool_name": "exec",
                "params": {
                    "command": (
                        "curl -L https://example.com/paper.pdf "
                        "-o /tmp_workspace/paper.pdf"
                    )
                },
            },
        )
        self.assertEqual(monitor.verify(before, SimpleNamespace()), ())
        after = ExecutionEvent(
            EventType.NODE_FINISH,
            "tool:download",
            boundary=BoundaryKind.FAN_OUT,
            produces=("tool-result:download",),
            payload={"tool_name": "exec", "params": before.payload["params"]},
        )
        violations = monitor.verify(after, SimpleNamespace())
        self.assertEqual(violations[0].contract_id, "auto_expected_tool_outputs")
        self.assertEqual(violations[0].details["missing"], ["workspace:paper.pdf"])

    def test_goal_contracts_are_synthesized_from_prompt(self) -> None:
        contracts = synthesize_goal_contracts(
            "Download the PDF to /tmp_workspace/MAE.pdf and write a report in "
            "/tmp_workspace/mae_summary.md. Read /tmp_workspace/source.txt first."
        )
        by_artifact = {contract.artifact: contract for contract in contracts}
        self.assertEqual(by_artifact["workspace:MAE.pdf"].kind, "valid_pdf")
        self.assertEqual(
            by_artifact["workspace:mae_summary.md"].kind,
            "exists_nonempty",
        )
        self.assertNotIn("workspace:source.txt", by_artifact)

        summary_contracts = synthesize_goal_contracts(
            "Summarize the paper in /tmp_workspace/summary.md."
        )
        self.assertEqual(
            summary_contracts,
            (
                type(summary_contracts[0])(
                    artifact="workspace:summary.md",
                    kind="exists_nonempty",
                    blocking=True,
                ),
            ),
        )

    def test_openclaw_uses_synthesized_effects_for_runtime_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            (workspace / "exec").mkdir(parents=True)
            spec = AgentTaskSpec(
                task_id="test-auto-contract",
                task={"pgir_require_plan": False},
                workspace_path=str(workspace),
                prompt="test",
                timeout_seconds=10,
                output_dir=root,
                model="fake",
            )
            adapter = OpenClawExecutorAdapter(spec, root / "bridge")
            before_download = adapter.event_from_bridge(
                self._raw(
                    "before-download",
                    "before_tool_call",
                    "download",
                    {
                        "command": (
                            "curl -L https://example.com/paper.pdf "
                            "-o /tmp_workspace/paper.pdf"
                        )
                    },
                )
            )
            self.assertEqual(
                before_download.payload["expected_produces"],
                ("workspace:paper.pdf",),
            )
            self.assertEqual(before_download.produces, ())
            adapter.artifact_producers["workspace:paper.pdf"] = "tool:download"
            before_extract = adapter.event_from_bridge(
                self._raw(
                    "before-extract",
                    "before_tool_call",
                    "extract",
                    {
                        "command": (
                            "pdftotext /tmp_workspace/paper.pdf "
                            "/tmp_workspace/paper.txt"
                        )
                    },
                )
            )
            self.assertEqual(before_extract.parents, ("tool:download",))
            self.assertEqual(before_extract.consumes, ("workspace:paper.pdf",))
            self.assertEqual(
                before_extract.payload["expected_produces"],
                ("workspace:paper.txt",),
            )

    @staticmethod
    def _raw(event_id: str, kind: str, call: str, params: dict) -> dict:
        return {
            "event_id": event_id,
            "kind": kind,
            "event": {"toolName": "exec", "toolCallId": call, "params": params},
            "context": {"toolName": "exec", "toolCallId": call},
        }


if __name__ == "__main__":
    unittest.main()
