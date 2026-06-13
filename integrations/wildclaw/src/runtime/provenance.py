from __future__ import annotations

from collections import defaultdict, deque

from .model import EventType, ExecutionEvent
from .plan import ExecutionPlan


class ProvenanceGraph:
    def __init__(self) -> None:
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, set[str]] = defaultdict(set)
        self.artifact_producer: dict[str, str] = {}
        self.executed: set[str] = set()
        self.started: set[str] = set()
        self.roots: set[str] = set()
        self.tainted: set[str] = set()

    def record(self, event: ExecutionEvent) -> None:
        node = event.node_id
        self.parents.setdefault(node, set())
        self.children.setdefault(node, set())
        inferred_parents = set(event.parents)
        inferred_parents.update(
            producer
            for artifact in event.consumes
            if (producer := self.artifact_producer.get(artifact)) is not None
        )
        for parent in inferred_parents:
            self.parents[node].add(parent)
            self.children[parent].add(node)
        if not self.parents[node]:
            self.roots.add(node)
        else:
            self.roots.discard(node)
        for artifact in event.produces:
            self.artifact_producer[artifact] = node
        if event.type == EventType.NODE_START:
            self.started.add(node)
        if event.type in {EventType.NODE_FINISH, EventType.TASK_FINISH}:
            self.executed.add(node)

    def seed_plan(self, plan: ExecutionPlan) -> None:
        """Install declared plan topology before runtime nodes are bound."""
        for node in plan.nodes.values():
            plan_node = f"plan:{node.node_id}"
            self.parents.setdefault(plan_node, set())
            self.children.setdefault(plan_node, set())
            for parent in node.parents:
                plan_parent = f"plan:{parent}"
                self.parents[plan_node].add(plan_parent)
                self.children[plan_parent].add(plan_node)
            if not node.parents:
                self.parents[plan_node].add("task_root")
                self.children["task_root"].add(plan_node)
            # Planned outputs are declarations, not produced artifacts. Actual
            # producer edges are installed only after a runtime node finishes.

    def bind_plan_node(self, runtime_node: str, plan_node_id: str) -> None:
        plan_node = f"plan:{plan_node_id}"
        self.parents[runtime_node].add(plan_node)
        self.children[plan_node].add(runtime_node)

    def ancestors(self, node_ids: set[str]) -> set[str]:
        return self._walk(node_ids, self.parents)

    def descendants(self, node_ids: set[str]) -> set[str]:
        return self._walk(node_ids, self.children)

    def minimal_frontier(self, responsible_nodes: set[str]) -> set[str]:
        """Keep the earliest responsible ancestors that cover later candidates."""
        frontier = set(responsible_nodes)
        for node in tuple(responsible_nodes):
            if self.ancestors({node}) & responsible_nodes:
                frontier.discard(node)
        return frontier

    def affected_executed_subgraph(self, frontier: set[str]) -> set[str]:
        return (self.descendants(frontier) | frontier) & self.executed

    def mark_tainted(self, node_ids: set[str]) -> None:
        self.tainted.update(node_ids)
        self.tainted.update(self.descendants(node_ids))

    def is_strict_local_frontier(self, frontier: set[str]) -> bool:
        """A frontier is structurally local when it does not collapse to a root plan.

        Comparing against only the executed prefix is unsound at runtime: an
        affected prefix may still be a strict subgraph of the not-yet-executed
        plan. Root reachability is the structural escalation criterion.
        """
        return bool(frontier) and not bool(frontier & self.roots)

    @staticmethod
    def _walk(start: set[str], edges: dict[str, set[str]]) -> set[str]:
        seen: set[str] = set()
        queue = deque(start)
        while queue:
            node = queue.popleft()
            for neighbor in edges.get(node, ()):
                if neighbor in seen or neighbor in start:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        return seen
