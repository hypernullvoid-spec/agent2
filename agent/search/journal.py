"""
Solution journal — the tree of attempted solutions.

Every node is one complete solution script plus what happened when it ran.
Nodes form a tree: drafts are roots; debug/improve nodes are children of
the node they modify. The journal is the agent's memory of the search and
is what makes the loop *converge* instead of wandering: every new prompt
is built from what the tree already learned.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    plan: str = ""
    code: str = ""
    stage: str = "draft"                  # draft | debug | improve
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    step: int = 0
    ctime: float = field(default_factory=time.time)

    # execution outcome
    term_out: str = ""
    exec_time: float = 0.0
    exit_code: int = 0
    timed_out: bool = False

    # review outcome
    analysis: str = ""
    metric: Optional[float] = None
    lower_is_better: bool = False
    is_buggy: bool = True                 # pessimistic until reviewed

    children: list[str] = field(default_factory=list)

    @property
    def is_good(self) -> bool:
        return not self.is_buggy and self.metric is not None

    def debug_depth(self, journal: "Journal") -> int:
        """How many consecutive debug ancestors this node has."""
        depth, node = 0, self
        while node.stage == "debug" and node.parent_id:
            depth += 1
            node = journal.get(node.parent_id)
        return depth

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(**d)


class Journal:
    def __init__(self):
        self.nodes: list[Node] = []
        self._by_id: dict[str, Node] = {}

    def append(self, node: Node) -> Node:
        node.step = len(self.nodes)
        self.nodes.append(node)
        self._by_id[node.id] = node
        if node.parent_id and node.parent_id in self._by_id:
            self._by_id[node.parent_id].children.append(node.id)
        return node

    def get(self, node_id: str) -> Node:
        return self._by_id[node_id]

    def __len__(self) -> int:
        return len(self.nodes)

    # ── queries the search policy needs ────────────────────────────────

    @property
    def draft_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.parent_id is None]

    @property
    def good_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.is_good]

    @property
    def buggy_leaves(self) -> list[Node]:
        return [n for n in self.nodes if n.is_buggy and not n.children]

    def best_node(self) -> Optional[Node]:
        good = self.good_nodes
        if not good:
            return None
        # direction is a per-run property; trust the majority vote of reviews
        lower = sum(n.lower_is_better for n in good) > len(good) / 2
        return min(good, key=lambda n: n.metric) if lower else max(good, key=lambda n: n.metric)

    # ── prompt memory ───────────────────────────────────────────────────

    def summarize(self, max_nodes: int = 12) -> str:
        """Compact history injected into draft/improve prompts."""
        if not self.nodes:
            return "(no attempts yet)"
        lines = []
        good = self.good_nodes[-max_nodes:]
        for n in good:
            lines.append(
                f"— attempt {n.step} [{n.stage}] metric={n.metric:.5g}"
                f"{' (lower better)' if n.lower_is_better else ''}: {n.plan[:300]}"
            )
        buggy = [n for n in self.nodes if n.is_buggy][-max(2, max_nodes - len(good)):]
        for n in buggy:
            reason = n.analysis[:200] or n.term_out[-200:]
            lines.append(f"— attempt {n.step} [{n.stage}] FAILED: {reason}")
        return "\n".join(lines) or "(no attempts yet)"

    # ── persistence / rendering ──────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"nodes": [n.to_dict() for n in self.nodes]}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Journal":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        j = cls()
        for nd in data["nodes"]:
            node = Node.from_dict(nd)
            j.nodes.append(node)
            j._by_id[node.id] = node
        return j

    def render_tree(self) -> str:
        """ASCII tree for reports / the dashboard."""
        best = self.best_node()
        out: list[str] = []

        def label(n: Node) -> str:
            mark = " ★" if best and n.id == best.id else ""
            metric = f" metric={n.metric:.5g}" if n.metric is not None else ""
            status = "buggy" if n.is_buggy else "good"
            return f"[{n.step}] {n.stage} ({status}){metric}{mark}"

        def walk(n: Node, prefix: str, is_last: bool):
            conn = "└─ " if is_last else "├─ "
            out.append(prefix + conn + label(n))
            kids = [self.get(c) for c in n.children]
            for i, k in enumerate(kids):
                walk(k, prefix + ("   " if is_last else "│  "), i == len(kids) - 1)

        roots = self.draft_nodes
        for i, r in enumerate(roots):
            walk(r, "", i == len(roots) - 1)
        return "\n".join(out) or "(empty)"
