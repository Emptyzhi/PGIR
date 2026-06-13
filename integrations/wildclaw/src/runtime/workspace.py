from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from src.agents.base import AgentTaskSpec

from .adapter import AgentExecutorAdapter
from .model import BoundaryKind, ContractViolation, ExecutionEvent


INSPECT_WORKSPACE_SCRIPT = r"""
from pathlib import Path
import hashlib
import json
import sys

root = Path("/tmp_workspace")
snapshot = json.loads(sys.stdin.read() or "{}")
changes = []

def digest(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None

for rel, meta in sorted(snapshot.items()):
    current = root / rel
    if not current.exists():
        changes.append(
            {
                "relative_path": rel,
                "change_type": "deleted_or_moved",
                "old_sha256": meta.get("sha256"),
                "new_sha256": None,
            }
        )
        continue
    if not current.is_file():
        continue
    current_sha = digest(current)
    if current_sha is not None and current_sha != meta.get("sha256"):
        changes.append(
            {
                "relative_path": rel,
                "change_type": "modified",
                "old_sha256": meta.get("sha256"),
                "new_sha256": current_sha,
            }
        )

print(json.dumps({"changes": changes}, ensure_ascii=False))
"""


WORKSPACE_MANIFEST_SCRIPT = r"""
from pathlib import Path
import hashlib
import json

root = Path("/tmp_workspace")
manifest = {}
if root.exists():
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
            manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
print(json.dumps(manifest, sort_keys=True))
"""


MIGRATE_FILE_SCRIPT = r"""
from pathlib import Path
import json
import shutil
import sys

path = Path(sys.argv[1])
stem = path.stem
suffix = path.suffix
candidate = path.with_name(f"{stem}_pgir_migrated{suffix}")
index = 2
while candidate.exists():
    candidate = path.with_name(f"{stem}_pgir_migrated_{index}{suffix}")
    index += 1
candidate.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(path, candidate)
print(json.dumps({"migrated_to": str(candidate)}, ensure_ascii=False))
"""


class WorkspaceExecutorAdapter(AgentExecutorAdapter):
    """OpenClaw/WildClaw adapter for task and workspace boundaries.

    OpenClaw currently exposes whole-task execution, so this adapter can enforce
    task-start and final-commit workspace contracts. Tool-level adapters can use
    the same controller interface to emit finer-grained events.
    """

    def __init__(self, spec: AgentTaskSpec) -> None:
        self.spec = spec
        self.seed_snapshot = self._snapshot_seed_workspace()
        self.guard_enabled = self._should_guard_file_mutations()
        self.pending_changes: list[dict[str, Any]] = []
        self.paused = False
        self.operations: list[dict[str, Any]] = []
        self.checkpoint_index = 0

    def pause(self) -> None:
        self.paused = True
        self.operations.append({"operation": "pause"})

    def resume(self) -> None:
        self.paused = False
        self.operations.append({"operation": "resume"})

    def snapshot(self, label: str) -> Any:
        self.checkpoint_index += 1
        safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label)[:80]
        checkpoint = f"/tmp/pgir_checkpoints/{self.checkpoint_index:05d}_{safe_label}"
        proc = subprocess.run(
            [
                "docker",
                "exec",
                self.spec.task_id,
                "python3",
                "-c",
                (
                    "from pathlib import Path; import shutil,sys; "
                    "src=Path('/tmp_workspace'); dst=Path(sys.argv[1]); "
                    "shutil.rmtree(dst, ignore_errors=True); "
                    "dst.parent.mkdir(parents=True, exist_ok=True); "
                    "shutil.copytree(src,dst,symlinks=True)"
                ),
                checkpoint,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        snapshot = {
            "label": label,
            "checkpoint": checkpoint if proc.returncode == 0 else None,
            "seed_files": len(self.seed_snapshot),
            "guard_enabled": self.guard_enabled,
            "snapshot_error": proc.stderr.strip() if proc.returncode else "",
        }
        self.operations.append({"operation": "snapshot", **snapshot})
        return snapshot

    def restore(self, snapshot: Any) -> None:
        checkpoint = snapshot.get("checkpoint") if isinstance(snapshot, dict) else None
        if not checkpoint:
            self.operations.append(
                {"operation": "restore", "snapshot": snapshot, "restored": False}
            )
            return
        proc = subprocess.run(
            [
                "docker",
                "exec",
                self.spec.task_id,
                "python3",
                "-c",
                (
                    "from pathlib import Path; import shutil,sys; "
                    "src=Path(sys.argv[1]); dst=Path('/tmp_workspace'); "
                    "shutil.rmtree(dst, ignore_errors=True); "
                    "shutil.copytree(src,dst,symlinks=True)"
                ),
                checkpoint,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.operations.append(
            {
                "operation": "restore",
                "checkpoint": checkpoint,
                "restored": proc.returncode == 0,
                "stderr": proc.stderr.strip(),
            }
        )

    def prepare_repair(
        self,
        snapshot: Any,
        node_ids: tuple[str, ...],
        affected_nodes: tuple[str, ...] = (),
    ) -> bool:
        _ = affected_nodes
        if not snapshot or not snapshot.get("checkpoint"):
            self.operations.append(
                {
                    "operation": "prepare_repair",
                    "nodes": list(node_ids),
                    "restored": False,
                    "reason": "no checkpoint required by workspace repair",
                }
            )
            return True
        self.restore(snapshot)
        restored = bool(self.operations[-1].get("restored"))
        self.operations.append(
            {
                "operation": "prepare_repair",
                "nodes": list(node_ids),
                "restored": restored,
            }
        )
        return restored

    def repair_nodes(self, node_ids: tuple[str, ...], violations: tuple[Any, ...]) -> bool:
        _ = (node_ids, violations)
        restored: list[dict[str, Any]] = []
        for change in self.pending_changes:
            rel = change["relative_path"]
            host_path = Path(self.seed_snapshot[rel]["host_path"])
            migrated_to = None
            if change["change_type"] == "modified":
                migrated_to = self._migrate_container_file(rel)
            self._ensure_container_parent(rel)
            cp = subprocess.run(
                [
                    "docker",
                    "cp",
                    str(host_path),
                    f"{self.spec.task_id}:/tmp_workspace/{rel}",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            restored.append(
                {
                    "relative_path": rel,
                    "change_type": change["change_type"],
                    "migrated_to": migrated_to,
                    "restored": cp.returncode == 0,
                    "restore_stderr": cp.stderr.strip(),
                }
            )
        self.operations.append({"operation": "repair_nodes", "restored": restored})
        return bool(restored) and all(item["restored"] for item in restored)

    def replay_nodes(self, node_ids: tuple[str, ...]) -> bool:
        self.operations.append(
            {
                "operation": "replay_nodes",
                "nodes": list(node_ids),
                "supported": False,
            }
        )
        return False

    def global_replan(self, reason: str) -> bool:
        self.operations.append(
            {
                "operation": "global_replan",
                "reason": reason,
                "supported": False,
            }
        )
        return False

    def boundary_violations(self, event: ExecutionEvent) -> tuple[Any, ...]:
        _ = event
        return ()

    def protected_mutation_paths(self, event: ExecutionEvent) -> tuple[str, ...]:
        _ = event
        return ()

    def pending_execution_failures(self, event: ExecutionEvent) -> tuple[ContractViolation, ...]:
        _ = event
        return ()

    def workspace_changes(self, event: ExecutionEvent) -> tuple[dict[str, Any], ...]:
        if event.boundary != BoundaryKind.FINAL_COMMIT or not self.guard_enabled:
            return ()
        self.pending_changes = self._inspect_workspace_changes()
        return tuple(self.pending_changes)

    def capture_workspace_manifest(self) -> dict[str, str] | None:
        proc = subprocess.run(
            ["docker", "exec", self.spec.task_id, "python3", "-c", WORKSPACE_MANIFEST_SCRIPT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode:
            return None
        try:
            manifest = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None
        return manifest if isinstance(manifest, dict) else None

    def inspect_artifact(self, artifact: str) -> dict[str, Any]:
        if not artifact.startswith("workspace:"):
            return {"exists": False, "reason": "unsupported artifact namespace"}
        relative = artifact.removeprefix("workspace:")
        script = (
            "from pathlib import Path; import json,sys; "
            "p=Path('/tmp_workspace')/sys.argv[1]; "
            "data={'exists':p.exists(),'is_file':p.is_file(),'size':p.stat().st_size if p.is_file() else 0}; "
            "data['magic']=p.read_bytes()[:5].decode('latin1') if p.is_file() else ''; "
            "print(json.dumps(data))"
        )
        proc = subprocess.run(
            ["docker", "exec", self.spec.task_id, "python3", "-c", script, relative],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode:
            return {"exists": False, "reason": proc.stderr[-500:]}
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"exists": False, "reason": "artifact inspection parse failure"}
        return result if isinstance(result, dict) else {"exists": False}

    def report(self) -> dict[str, Any]:
        return {
            "adapter": "workspace",
            "capabilities": {
                "task_boundaries": True,
                "workspace_snapshot": True,
                "workspace_restore": True,
                "tool_events": False,
                "selective_replay": False,
                "global_replan": False,
            },
            "guard_enabled": self.guard_enabled,
            "seed_files": len(self.seed_snapshot),
            "pending_changes": self.pending_changes,
            "operations": self.operations,
        }

    def _snapshot_seed_workspace(self) -> dict[str, dict[str, Any]]:
        seed_root = Path(self.spec.workspace_path) / "exec"
        snapshot: dict[str, dict[str, Any]] = {}
        if not seed_root.is_dir():
            return snapshot
        for path in seed_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(seed_root).as_posix()
            try:
                data = path.read_bytes()
            except OSError:
                continue
            snapshot[rel] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "host_path": str(path),
            }
        return snapshot

    def _inspect_workspace_changes(self) -> list[dict[str, Any]]:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.spec.task_id,
                "python3",
                "-c",
                INSPECT_WORKSPACE_SCRIPT,
            ],
            input=json.dumps(
                {
                    rel: {"sha256": meta["sha256"], "size": meta["size"]}
                    for rel, meta in self.seed_snapshot.items()
                },
                ensure_ascii=False,
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            self.operations.append(
                {
                    "operation": "inspect_workspace",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr.strip(),
                }
            )
            return []
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return []
        return list(result.get("changes") or [])

    def _migrate_container_file(self, rel: str) -> str | None:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                self.spec.task_id,
                "python3",
                "-c",
                MIGRATE_FILE_SCRIPT,
                f"/tmp_workspace/{rel}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            return None
        try:
            return json.loads(proc.stdout).get("migrated_to")
        except json.JSONDecodeError:
            return None

    def _ensure_container_parent(self, rel: str) -> None:
        parent = str((Path("/tmp_workspace") / Path(rel).parent)).replace("\\", "/")
        subprocess.run(
            ["docker", "exec", self.spec.task_id, "mkdir", "-p", parent],
            capture_output=True,
        )

    def _should_guard_file_mutations(self) -> bool:
        prompt = self.spec.prompt.lower()
        explicit_mutation_authorizations = (
            "fix ",
            "fix all",
            "debug",
            "modify",
            "edit ",
            "update ",
            "patch ",
            "repair ",
            "rewrite",
            "replace ",
            "overwrite",
            "delete ",
            "remove ",
            "rename ",
            "correct ",
            "修复",
            "修改",
            "编辑",
            "更新",
            "替换",
            "覆盖",
            "删除",
            "重命名",
        )
        return bool(self.seed_snapshot) and not any(
            term in prompt for term in explicit_mutation_authorizations
        )
