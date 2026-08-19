"""
Tests for agent/task_router.py — how `swarn run` decides between the
document fast path and the ReAct agent.

The asymmetry in these tests is deliberate and mirrors the asymmetry in the
cost of being wrong. Routing a document question to the agent wastes LLM
calls and prints an extra paraphrase; routing a "train a model and write a
report" task to the document reader silently drops most of the requested
work. So the misrouting tests concentrate on the second direction.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.task_router import (  # noqa: E402
    Route,
    find_document_paths,
    is_document,
    looks_like_question,
    mentions_other_work,
    route,
)


def _docdir():
    """A temp dir holding one PDF, one PNG, one CSV — real files on disk,
    because the router deliberately refuses to fast-path a path that is not
    actually there."""
    tmp = tempfile.mkdtemp()
    for name in ("invoice.pdf", "report.pdf", "scan.png", "churn.csv"):
        with open(os.path.join(tmp, name), "wb") as fh:
            fh.write(b"%PDF-1.4\n")
    return tmp


# ── suffix classification ──────────────────────────────────────────────────

def test_document_suffixes_are_recognized_case_insensitively():
    for name in ("a.pdf", "A.PDF", "b.png", "c.JPEG", "d.tiff", "e.webp"):
        assert is_document(name), name


def test_non_document_suffixes_are_rejected():
    for name in ("data.csv", "train.py", "model.pkl", "notes.txt", "a.pdf.bak"):
        assert not is_document(name), name


# ── path discovery ─────────────────────────────────────────────────────────

def test_explicit_paths_are_found():
    tmp = _docdir()
    found = find_document_paths("what is the total", ["invoice.pdf"], cwd=tmp)
    assert found == [os.path.join(tmp, "invoice.pdf")]


def test_a_path_named_inside_the_task_text_is_found():
    tmp = _docdir()
    found = find_document_paths("what is the total in invoice.pdf", cwd=tmp)
    assert found == [os.path.join(tmp, "invoice.pdf")]


def test_a_path_that_does_not_exist_is_not_returned():
    # Left for the agent to explain, rather than fast-pathed into a confusing
    # error from the document reader about a file the user never had.
    tmp = _docdir()
    assert find_document_paths("what is in missing.pdf", cwd=tmp) == []


def test_a_non_document_file_is_not_treated_as_a_document():
    tmp = _docdir()
    assert find_document_paths("what is in churn.csv", cwd=tmp) == []


def test_prose_mentioning_a_document_without_a_path_probes_nothing():
    # "the invoice" must not be resolved against the filesystem.
    tmp = _docdir()
    assert find_document_paths("what is the total on the invoice", cwd=tmp) == []


def test_explicit_paths_come_before_ones_found_in_the_text():
    tmp = _docdir()
    found = find_document_paths("compare against report.pdf", ["invoice.pdf"], cwd=tmp)
    assert found == [os.path.join(tmp, "invoice.pdf"), os.path.join(tmp, "report.pdf")]


def test_the_same_document_named_twice_is_listed_once():
    tmp = _docdir()
    found = find_document_paths("what is the total in invoice.pdf", ["invoice.pdf"], cwd=tmp)
    assert found == [os.path.join(tmp, "invoice.pdf")]


def test_trailing_punctuation_does_not_break_a_path():
    tmp = _docdir()
    for task in ("read invoice.pdf.", "read 'invoice.pdf'", "read (invoice.pdf)"):
        assert find_document_paths(task, cwd=tmp) == [os.path.join(tmp, "invoice.pdf")]


def test_an_absolute_path_is_used_as_given():
    tmp = _docdir()
    abs_path = os.path.join(tmp, "invoice.pdf")
    assert find_document_paths(f"total in {abs_path}", cwd="/") == [abs_path]


# ── the two signals ────────────────────────────────────────────────────────

def test_other_work_is_detected():
    for task in ("train a model on this", "plot the line items",
                 "write a summary to notes.md", "index this directory"):
        assert mentions_other_work(task), task


def test_a_plain_question_carries_no_other_work_signal():
    for task in ("what is the total GST charged",
                 "who signed this and on what date",
                 "how much was the discount"):
        assert not mentions_other_work(task), task


def test_question_shapes_are_recognized():
    for task in ("what is the total", "the total?", "list the line items",
                 "summarise page 3", "how many pages are there"):
        assert looks_like_question(task), task


def test_a_bare_imperative_is_not_a_question():
    assert not looks_like_question("process this file end to end")


# ── routing ────────────────────────────────────────────────────────────────

def test_a_question_about_one_document_takes_the_fast_path():
    tmp = _docdir()
    decision = route("what is the total GST charged", ["invoice.pdf"], cwd=tmp)
    assert decision.kind == "ask"
    assert decision.is_fast_path
    assert decision.documents == [os.path.join(tmp, "invoice.pdf")]


def test_a_task_with_no_document_goes_to_the_agent():
    tmp = _docdir()
    decision = route("train the best model on churn.csv", cwd=tmp)
    assert decision.kind == "agent"
    assert decision.documents == []


def test_a_document_question_plus_other_work_goes_to_the_agent():
    # The failure this guards: fast-pathing would answer the question and
    # silently never plot anything.
    tmp = _docdir()
    decision = route("read invoice.pdf and plot the line items", cwd=tmp)
    assert decision.kind == "agent"
    assert decision.documents == [os.path.join(tmp, "invoice.pdf")]


def test_two_documents_go_to_the_agent():
    # doc_qa answers about one document at a time, so "compare these" is
    # agent work however question-shaped it looks.
    tmp = _docdir()
    decision = route("which of these has the higher total",
                     ["invoice.pdf", "report.pdf"], cwd=tmp)
    assert decision.kind == "agent"
    assert len(decision.documents) == 2


def test_a_non_question_about_a_document_goes_to_the_agent():
    tmp = _docdir()
    decision = route("process invoice.pdf end to end", cwd=tmp)
    assert decision.kind == "agent"


def test_force_ask_overrides_the_heuristic():
    tmp = _docdir()
    decision = route("process invoice.pdf end to end", cwd=tmp, force="ask")
    assert decision.kind == "ask"
    assert decision.documents == [os.path.join(tmp, "invoice.pdf")]


def test_force_agent_overrides_the_heuristic():
    tmp = _docdir()
    decision = route("what is the total", ["invoice.pdf"], cwd=tmp, force="agent")
    assert decision.kind == "agent"
    # Still resolved, so the agent can be told which file the task is about.
    assert decision.documents == [os.path.join(tmp, "invoice.pdf")]


def test_forcing_ask_without_a_document_still_reports_none():
    # The CLI turns this into an actionable error rather than calling the
    # document reader with nothing to read.
    decision = route("what is the total", cwd=_docdir(), force="ask")
    assert decision.kind == "ask"
    assert decision.documents == []


def test_every_route_explains_itself():
    tmp = _docdir()
    for task, paths in [("what is the total", ["invoice.pdf"]),
                        ("train a model", None),
                        ("read invoice.pdf and plot it", None)]:
        decision = route(task, paths, cwd=tmp)
        assert isinstance(decision, Route)
        assert decision.reason, f"no reason given for {task!r}"


def test_an_image_is_routed_the_same_way_as_a_pdf():
    tmp = _docdir()
    decision = route("what does this say", ["scan.png"], cwd=tmp)
    assert decision.kind == "ask"
    assert decision.documents == [os.path.join(tmp, "scan.png")]
