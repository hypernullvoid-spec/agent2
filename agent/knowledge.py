"""
Cross-run knowledge — the self-improvement engine.

HeyNeo (heyneo.com) treats every task as a cold start. This module makes each
run leave the platform smarter, borrowing three ideas from
NousResearch/hermes-agent:

  1. Bounded, self-curated playbook (agent/knowledge/playbook.md)
     A hard character cap forces *curation*, not accumulation: when a new
     lesson would overflow the cap, the oldest lessons are dropped. The
     playbook is injected verbatim into every future search prompt, so it
     must stay small enough to never crowd out the task itself.

  2. Post-run reflection (reflect_on_run)
     After a search finishes, one cheap low-temperature LLM call reviews the
     run digest and extracts up to 5 GENERALIZABLE lessons ("LightGBM with
     early stopping beat tuned XGBoost on wide tabular data", "stratify
     splits when classes are imbalanced — attempt 3 failed without it").
     Task-specific trivia is explicitly rejected by the prompt.

  3. Searchable run archive (SQLite FTS5, agent/knowledge/runs.db)
     Every finished run's task, outcome and winning code is indexed.
     At the start of a new run, the most similar past runs are retrieved by
     full-text match and their summaries injected as prior art — "you have
     solved something like this before, here is what won."

Everything degrades gracefully: no DB, no playbook, no API key → empty
strings, never an exception into the search loop.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Optional

DEFAULT_KNOWLEDGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "knowledge"))

PLAYBOOK_MAX_CHARS = 6000          # hard cap — forces curation (hermes-style)
LESSON_MAX_CHARS = 300             # one lesson = one distilled sentence or two
MAX_SIMILAR_RUNS = 3

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


class KnowledgeStore:
    """Playbook + run archive rooted at one directory."""

    def __init__(self, root: Optional[str] = None):
        self.root = os.path.abspath(
            root or os.environ.get("SWARN_KNOWLEDGE_DIR") or DEFAULT_KNOWLEDGE_DIR)
        os.makedirs(self.root, exist_ok=True)
        self.playbook_path = os.path.join(self.root, "playbook.md")
        self.db_path = os.path.join(self.root, "runs.db")

    # ── playbook (bounded procedural memory) ────────────────────────────

    def playbook(self) -> str:
        try:
            with open(self.playbook_path, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    def _lessons(self) -> list[str]:
        text = self.playbook()
        return [ln[2:].strip() for ln in text.splitlines() if ln.startswith("- ")]

    def add_lessons(self, lessons: list[str]) -> int:
        """Merge new lessons in; dedupe; drop oldest beyond the char cap.
        Returns how many lessons were actually added."""
        current = self._lessons()
        existing_lower = {l.lower() for l in current}
        added = 0
        for lesson in lessons:
            lesson = " ".join(str(lesson).split())[:LESSON_MAX_CHARS]
            if not lesson or lesson.lower() in existing_lower:
                continue
            current.append(lesson)
            existing_lower.add(lesson.lower())
            added += 1

        # enforce the cap by dropping the OLDEST lessons first
        def render(ls: list[str]) -> str:
            return "# Playbook — lessons from past runs\n\n" + \
                   "\n".join(f"- {l}" for l in ls) + "\n"

        while current and len(render(current)) > PLAYBOOK_MAX_CHARS:
            current.pop(0)

        if added or current:
            with open(self.playbook_path, "w", encoding="utf-8") as f:
                f.write(render(current) if current else "")
        return added

    # ── run archive (FTS5) ──────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS runs USING fts5("
            "run_id, task, summary, code, metric UNINDEXED, ts UNINDEXED)")
        return conn

    def index_run(self, run_id: str, task: str, summary: str,
                  code: str = "", metric: Optional[float] = None) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO runs (run_id, task, summary, code, metric, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, task[:4000], summary[:4000], code[:20000],
                     "" if metric is None else str(metric), time.time()))
        except sqlite3.Error:
            pass  # archive is best-effort, never fatal

    def search_runs(self, query: str, k: int = MAX_SIMILAR_RUNS) -> list[dict]:
        terms = _WORD_RE.findall(query)[:24]
        if not terms or not os.path.exists(self.db_path):
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT run_id, task, summary, metric FROM runs "
                    "WHERE runs MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, k)).fetchall()
        except sqlite3.Error:
            return []
        return [{"run_id": r[0], "task": r[1], "summary": r[2], "metric": r[3]}
                for r in rows]

    def get_run_code(self, run_id: str) -> str:
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT code FROM runs WHERE run_id = ? LIMIT 1",
                    (run_id,)).fetchone()
            return row[0] if row else ""
        except sqlite3.Error:
            return ""

    # ── context assembly (what the search prompts receive) ─────────────

    def context_for_task(self, task: str) -> str:
        """Playbook + similar past runs, formatted for prompt injection.
        Empty string when there is nothing useful."""
        parts: list[str] = []
        pb = self.playbook()
        if pb:
            parts.append(pb)
        similar = self.search_runs(task)
        if similar:
            lines = ["# Similar past runs (prior art — reuse what worked)"]
            for r in similar:
                metric = f" (best metric: {r['metric']})" if r["metric"] else ""
                lines.append(f"- [{r['run_id']}]{metric} task: {r['task'][:160]}")
                if r["summary"]:
                    lines.append(f"  outcome: {r['summary'][:240]}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────── reflection

REFLECT_SYSTEM = """You are the retrospective reviewer for an autonomous ML \
engineering agent. Given the digest of one finished solution-search run, \
extract up to 5 GENERALIZABLE lessons that would make FUTURE runs on OTHER \
tasks better. Rules:
- Each lesson must transfer across tasks ("early stopping + low learning \
rate beat deeper trees on small tabular data"), not restate task trivia \
("the label column was 'survived'").
- Prefer lessons about failures: what crashed, why, and how it was fixed.
- One or two sentences each, concrete and actionable.
- If the run taught nothing new, return an empty list."""

REFLECT_TOOL = {
    "name": "submit_lessons",
    "description": "Submit the distilled, generalizable lessons from this run.",
    "input_schema": {
        "type": "object",
        "properties": {
            "lessons": {
                "type": "array",
                "items": {"type": "string"},
                "description": "0-5 generalizable lessons, each 1-2 sentences",
            },
        },
        "required": ["lessons"],
    },
}


def reflect_on_run(task: str, journal, feedback_llm, store: KnowledgeStore,
                   run_id: str = "") -> list[str]:
    """One cheap LLM call → lessons merged into the playbook.
    Never raises; returns the lessons that were stored."""
    try:
        best = journal.best_node()
        digest = (
            f"# Task\n{task[:2000]}\n\n"
            f"# Run digest ({len(journal)} attempts)\n"
            f"{journal.summarize(20)}\n\n"
            f"# Outcome\n"
            + (f"Best metric {best.metric} via [{best.stage}] node; plan: "
               f"{best.plan[:600]}" if best else "No working solution found.")
        )
        resp = feedback_llm.call(
            REFLECT_SYSTEM, [{"role": "user", "content": digest}],
            tools=[REFLECT_TOOL],
            tool_choice={"type": "tool", "name": "submit_lessons"},
            temperature=0.2, max_tokens=1024,
        )
        lessons: list[str] = []
        for b in resp.tool_uses():
            if b.name == "submit_lessons":
                raw = b.input.get("lessons", [])
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        raw = [raw]
                lessons = [str(x) for x in raw][:5]
        if lessons:
            store.add_lessons(lessons)
        return lessons
    except Exception:  # noqa: BLE001 — reflection must never kill a finished run
        return []
