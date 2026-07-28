"""
SearchAgent — the AIDE-style draft -> debug -> improve policy.
"""

from __future__ import annotations

import json
import random
import re
from typing import Optional

from agent.llm import create_client
from agent.search.config import SearchConfig
from agent.search.journal import Journal, Node

CODE_SYSTEM = """You are Swarn, an expert ML engineer competing to produce the best \
solution to a machine learning task. You write complete, runnable, single-file \
Python scripts. Rules:
- Respond with a short plan (a few sentences), then EXACTLY ONE ```python code block \
containing the COMPLETE solution script (imports to output). No other code blocks.
- The script must be self-contained and run top-to-bottom without arguments.
- Read data ONLY from the relative directory ./input (already prepared).
- Write any outputs (submission files, models, plots) to the current directory ./
- The script MUST evaluate its own solution on a held-out validation split and print \
the result on its own line in this exact format:  Final Validation Metric: <number>
- Prefer fast, strong baselines first (gradient boosting for tabular data); avoid \
approaches that need downloads or GPUs unless the task demands it.
- No placeholder code, no TODOs, no exceptions swallowed silently."""

REVIEW_SYSTEM = """You are a strict ML experiment reviewer. Given a task, a solution \
script, and its execution output, decide whether the run is buggy and extract the \
validation metric. A run is buggy if it raised, timed out, produced no metric, or \
clearly did not do what the task asks (e.g. evaluated on training data only)."""

REVIEW_TOOL = {
    "name": "submit_review",
    "description": "Submit the structured review of one solution run.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_bug": {"type": "boolean",
                       "description": "true if the run failed, raised, timed out, or produced no valid metric"},
            "summary": {"type": "string",
                        "description": "2-3 sentence analysis of what happened and what to try next"},
            "metric": {"type": ["number", "null"],
                       "description": "the validation metric value, or null if none was produced"},
            "lower_is_better": {"type": "boolean",
                                "description": "true if the metric improves as it decreases (loss, RMSE, error)"},
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better"],
    },
}

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
METRIC_RE = re.compile(r"Final Validation Metric:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def extract_code(text: str) -> str:
    blocks = CODE_BLOCK_RE.findall(text)
    if not blocks:
        return ""
    return max(blocks, key=len).strip()


def extract_metric_fallback(term_out: str) -> Optional[float]:
    m = METRIC_RE.findall(term_out)
    return float(m[-1]) if m else None


class SearchAgent:
    def __init__(self, task: str, config: SearchConfig, journal: Journal,
                 data_preview: str = "", evaluation_note: str = "",
                 knowledge_context: str = ""):
        self.task = task
        self.cfg = config
        self.journal = journal
        self.data_preview = data_preview
        self.evaluation_note = evaluation_note
        self.knowledge_context = knowledge_context
        self.code_llm = create_client(config.code_model)
        self.feedback_llm = create_client(config.feedback_model)

    # -- policy -----------------------------------------------------------

    def choose_action(self, reserved: frozenset = frozenset(),
                      pending_drafts: int = 0) -> tuple[str, Optional[Node]]:
        """Returns (stage, parent_node). `reserved`/`pending_drafts` exist
        for the parallel scheduler; the sequential path passes neither."""
        j = self.journal
        if len(j.draft_nodes) + pending_drafts < self.cfg.num_drafts:
            return "draft", None

        debuggable = [n for n in j.buggy_leaves
                      if n.debug_depth(j) < self.cfg.max_debug_depth
                      and n.id not in reserved]
        if debuggable and random.random() < self.cfg.debug_prob:
            return "debug", self._pick_debug_target(debuggable)

        ranked = self._ranked_good_nodes()
        if ranked:
            k = max(1, min(self.cfg.improve_topk, len(ranked)))
            pick = ranked[0] if k == 1 or random.random() < 0.7 else random.choice(ranked[1:k])
            return "improve", pick
        if debuggable:
            return "debug", self._pick_debug_target(debuggable)
        return "draft", None

    def _pick_debug_target(self, debuggable: list[Node]) -> Node:
        return min(debuggable, key=lambda n: (n.debug_depth(self.journal), -n.step))

    def _ranked_good_nodes(self) -> list[Node]:
        good = self.journal.good_nodes
        if not good:
            return []
        lower = sum(n.lower_is_better for n in good) > len(good) / 2
        return sorted(good, key=lambda n: n.metric, reverse=not lower)

    # -- prompt builders --------------------------------------------------

    def _task_header(self) -> str:
        parts = [f"# Task\n{self.task}"]
        if self.evaluation_note:
            parts.append(f"# Evaluation\n{self.evaluation_note}")
        if self.data_preview:
            parts.append(f"# Data overview\n{self.data_preview}")
        if self.knowledge_context:
            parts.append(self.knowledge_context)
        return "\n\n".join(parts)

    def draft_prompt(self) -> str:
        return (
            f"{self._task_header()}\n\n"
            f"# Memory (previous attempts on this task)\n"
            f"{self.journal.summarize(self.cfg.max_memory_nodes)}\n\n"
            f"# Instructions\nPropose a NEW solution approach that is meaningfully "
            f"different from previous attempts. Keep it simple and reliable — a "
            f"working baseline beats an ambitious crash. Then write the complete script."
        )

    def improve_prompt(self, parent: Node) -> str:
        return (
            f"{self._task_header()}\n\n"
            f"# Current best solution (validation metric: {parent.metric})\n"
            f"```python\n{parent.code}\n```\n\n"
            f"# Memory (previous attempts)\n"
            f"{self.journal.summarize(self.cfg.max_memory_nodes)}\n\n"
            f"# Instructions\nPropose ONE atomic, well-motivated improvement to this "
            f"solution (better features, better model, better hyperparameters, "
            f"ensembling…). Do not rewrite everything. Then write the complete improved script."
        )

    def debug_prompt(self, parent: Node) -> str:
        term = parent.term_out[-self.cfg.max_term_out_chars:]
        return (
            f"{self._task_header()}\n\n"
            f"# Buggy solution\n```python\n{parent.code}\n```\n\n"
            f"# Execution output\n```\n{term}\n```\n\n"
            f"# Instructions\nDiagnose the actual root cause from the output, state "
            f"the fix in your plan, then write the complete fixed script."
        )

    # -- one search step ---------------------------------------------------

    def propose(self, stage: Optional[str] = None,
                parent: Optional[Node] = None) -> Node:
        if stage is None:
            stage, parent = self.choose_action()
        prompt = {
            "draft": self.draft_prompt,
            "improve": lambda: self.improve_prompt(parent),
            "debug": lambda: self.debug_prompt(parent),
        }[stage]() if stage != "draft" else self.draft_prompt()

        resp = self.code_llm.call(
            CODE_SYSTEM, [{"role": "user", "content": prompt}],
            temperature=self.cfg.code_temperature,
        )
        text = resp.text
        code = extract_code(text)
        plan = CODE_BLOCK_RE.sub("", text).strip()[:2000]
        return Node(plan=plan, code=code, stage=stage,
                    parent_id=parent.id if parent else None)

    # -- review after execution ---------------------------------------------

    def review(self, node: Node) -> None:
        if not node.code:
            node.is_buggy, node.analysis = True, "Model produced no code block."
            return
        if node.timed_out:
            node.is_buggy = True
            node.analysis = f"Execution timed out after {node.exec_time:.0f}s. Use a faster approach."
            node.metric = None
            return

        term = node.term_out[-self.cfg.max_term_out_chars:]
        prompt = (
            f"# Task\n{self.task[:2000]}\n\n"
            f"# Solution script\n```python\n{node.code[:8000]}\n```\n\n"
            f"# Execution output (exit code {node.exit_code})\n```\n{term}\n```"
        )
        try:
            resp = self.feedback_llm.call(
                REVIEW_SYSTEM, [{"role": "user", "content": prompt}],
                tools=[REVIEW_TOOL],
                tool_choice={"type": "tool", "name": "submit_review"},
                temperature=self.cfg.feedback_temperature,
                max_tokens=1024,
            )
            review = self._parse_review(resp)
        except Exception as e:  # noqa: BLE001
            review = None
            node.analysis = f"(review call failed: {e})"

        if review:
            node.analysis = review.get("summary", "")
            node.metric = review.get("metric")
            node.lower_is_better = bool(review.get("lower_is_better", False))
            node.is_buggy = bool(review.get("is_bug", True)) or node.metric is None

        printed = extract_metric_fallback(node.term_out)
        if printed is not None:
            node.metric = printed
            if node.exit_code == 0 and review is None:
                node.is_buggy = False
        elif node.exit_code != 0:
            node.is_buggy = True

    @staticmethod
    def _parse_review(resp) -> Optional[dict]:
        for b in resp.tool_uses():
            if b.name == "submit_review":
                return b.input
        m = re.search(r"\{.*\}", resp.text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None
