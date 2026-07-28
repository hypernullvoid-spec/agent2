"""Cross-run knowledge — playbook curation, FTS archive, reflection."""

import tempfile

from agent.knowledge import (KnowledgeStore, PLAYBOOK_MAX_CHARS,
                             reflect_on_run)
from agent.llm.mock_client import MockLLMClient, tool_response
from agent.search.journal import Journal, Node


def _store() -> KnowledgeStore:
    return KnowledgeStore(tempfile.mkdtemp(prefix="swarn_knowledge_"))


def test_add_lessons_and_dedupe():
    s = _store()
    assert s.add_lessons(["Always stratify imbalanced splits.",
                          "always stratify imbalanced splits."]) == 1
    assert s.add_lessons(["Always stratify imbalanced splits."]) == 0
    assert "stratify" in s.playbook()


def test_playbook_cap_drops_oldest():
    s = _store()
    lessons = [f"Lesson number {i}: " + "x" * 250 for i in range(60)]
    s.add_lessons(lessons)
    pb = s.playbook()
    assert len(pb) <= PLAYBOOK_MAX_CHARS
    assert "Lesson number 59" in pb          # newest kept
    assert "Lesson number 0:" not in pb      # oldest dropped


def test_run_archive_fts_search():
    s = _store()
    s.index_run("run-a", "Predict house prices, minimize RMSE",
                "best metric 0.12 with gradient boosting", code="print('hi')",
                metric=0.12)
    s.index_run("run-b", "Classify images of cats and dogs",
                "best accuracy 0.94 with a CNN", metric=0.94)
    hits = s.search_runs("predict prices for houses RMSE regression")
    assert hits and hits[0]["run_id"] == "run-a"
    assert s.get_run_code("run-a") == "print('hi')"


def test_context_for_task_combines_playbook_and_runs():
    s = _store()
    s.add_lessons(["Gradient boosting first on tabular data."])
    s.index_run("run-a", "Predict churn probability", "best AUC 0.88", metric=0.88)
    ctx = s.context_for_task("Predict churn for telecom customers")
    assert "Playbook" in ctx and "run-a" in ctx


def test_empty_store_yields_empty_context():
    assert _store().context_for_task("anything at all") == ""


def test_reflect_on_run_stores_lessons():
    s = _store()
    journal = Journal()
    n = Node(plan="use lightgbm", code="...", stage="draft")
    n.is_buggy, n.metric, n.lower_is_better = False, 0.9, False
    journal.append(n)

    llm = MockLLMClient(script=[tool_response("submit_lessons", {
        "lessons": ["LightGBM with early stopping is a strong tabular baseline."]})])
    lessons = reflect_on_run("predict y", journal, llm, s)
    assert len(lessons) == 1
    assert "LightGBM" in s.playbook()


def test_reflect_never_raises_on_junk():
    s = _store()
    llm = MockLLMClient(script=["not a tool call at all"])
    assert reflect_on_run("task", Journal(), llm, s) == []
