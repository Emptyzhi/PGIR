from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from typing import Any

from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent
from src.agents.runtime_controlled import RuntimeControlledAgent
from src.runtime.workspace import WorkspaceExecutorAdapter


class FakeDelegate(BaseAgent):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    @property
    def expects_gateway(self) -> bool:
        return False

    @property
    def transcript_container_path(self) -> str:
        return "/tmp/fake.jsonl"

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        self.prompts.append(spec.prompt)
        return AgentExecution(elapsed_time=0.01)

    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict[str, Any]:
        return {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 2,
            "cost_usd": 0.0,
            "request_count": 1,
            "elapsed_time": elapsed_time,
        }


class RuntimeControlledAgentTests(unittest.TestCase):
    def test_conditions_share_same_wrapper_and_delegate_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            (workspace / "exec").mkdir(parents=True)
            for condition in (
                "no_repair_control",
                "full_trace_retry_control",
                "reflexion_verbal_retry",
                "pgir_hidden_taint_ancestor_repair",
                "pgir_no_provenance_taint",
                "local_leaf_retry_no_ancestor_control",
                "no_contract_retry_control",
            ):
                delegate = FakeDelegate()
                output = root / condition
                output.mkdir()
                agent = RuntimeControlledAgent(delegate, condition)
                spec = AgentTaskSpec(
                    task_id=f"task-{condition}",
                    task={"task_id": "task", "category": "test"},
                    workspace_path=str(workspace),
                    prompt="Original task prompt",
                    timeout_seconds=10,
                    output_dir=output,
                    model="fake",
                )
                agent.run_task(spec)
                self.assertEqual(len(delegate.prompts), 1)
                self.assertIn("Original task prompt", delegate.prompts[0])
                self.assertTrue((output / "runtime_control_report.json").exists())

    def test_workspace_guard_is_generic_and_skips_explicit_mutation_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            seed = workspace / "exec"
            seed.mkdir(parents=True)
            (seed / "existing.txt").write_text("original", encoding="utf-8")
            common = {
                "task_id": "task",
                "task": {"task_id": "task", "category": "any"},
                "workspace_path": str(workspace),
                "timeout_seconds": 10,
                "output_dir": root,
                "model": "fake",
            }
            protected = WorkspaceExecutorAdapter(
                AgentTaskSpec(prompt="Write a report to existing.txt", **common)
            )
            mutable = WorkspaceExecutorAdapter(
                AgentTaskSpec(prompt="Fix and modify existing.txt", **common)
            )
            self.assertTrue(protected.guard_enabled)
            self.assertFalse(mutable.guard_enabled)


if __name__ == "__main__":
    unittest.main()
