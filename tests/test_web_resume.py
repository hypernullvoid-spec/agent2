"""
Tests for web conversation continuity — resume_session_ids on /api/jobs.

The web dashboard historically ran every job with a fresh, history-less
AgentLoop, so a follow-up task started blind to the run it was following up
on. These tests pin the fix: seed_history() builds the same synthetic
exchange the REPL's /resume does, the job registry threads the ids through
to the runner, and the API rejects bad requests at submit time (a wrong
session id must be a 422 the frontend can show, not a failed job).

No LLM is ever called: the API tests only exercise requests that fail
validation before a runner thread starts, and seed_history is tested against
a stubbed session store.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.web import jobs as jobs_module  # noqa: E402
from agent.web.jobs import Job, seed_history  # noqa: E402


class _FakeStore:
    """Stands in for SessionStore: knows two sessions, s1 and s2."""

    def recall_as_text(self, sid):
        return f"transcript of {sid}"

    def get_session(self, sid):
        return {"id": sid} if sid in ("s1", "s2") else None


def _with_fake_store(fn):
    """Run fn with agent.memory.memory.get_session_store stubbed out."""
    import agent.memory.memory as memory_module

    original = memory_module.get_session_store
    memory_module.get_session_store = lambda: _FakeStore()
    try:
        return fn()
    finally:
        memory_module.get_session_store = original


def test_seed_history_shape_and_order():
    history = _with_fake_store(lambda: seed_history(["s1", "s2"]))
    # The REPL's /resume shape: one user message carrying the transcripts,
    # one assistant acknowledgement — so the next real task is a clean turn.
    assert [m["role"] for m in history] == ["user", "assistant"]
    body = history[0]["content"]
    assert "transcript of s1" in body and "transcript of s2" in body
    # Oldest first: the thread reads top-to-bottom like the conversation did.
    assert body.index("transcript of s1") < body.index("transcript of s2")


def test_job_carries_resume_ids_in_summary():
    job = Job(id="j1", task="t", method="react", resume_session_ids=["s1"])
    assert job.summary()["resume_session_ids"] == ["s1"]
    # And the default stays an independent list per job, not a shared one.
    a, b = Job(id="a", task="t", method="react"), Job(id="b", task="t", method="react")
    a.resume_session_ids.append("s1")
    assert b.resume_session_ids == []


def test_registry_submit_defensive_copy():
    ids = ["s1"]
    registry = jobs_module.JobRegistry()
    # Don't let the runner thread actually run an agent: replace _run with a
    # no-op before submit spawns it.
    registry._run = lambda job: None
    job = registry.submit(task="t", method="react", resume_session_ids=ids)
    ids.append("s2")
    assert job.resume_session_ids == ["s1"]


def test_api_rejects_resume_on_non_react():
    from fastapi.testclient import TestClient
    import agent.web.dashboard as dashboard

    client = TestClient(dashboard.app)
    r = client.post("/api/jobs", json={"task": "t", "method": "team",
                                       "resume_session_ids": ["s1"]})
    assert r.status_code == 422
    assert "react" in r.json()["detail"]


def test_api_rejects_unknown_resume_session():
    from fastapi.testclient import TestClient
    import agent.web.dashboard as dashboard

    original = dashboard.get_session_store
    dashboard.get_session_store = lambda: _FakeStore()
    try:
        client = TestClient(dashboard.app)
        r = client.post("/api/jobs", json={"task": "t", "method": "react",
                                           "resume_session_ids": ["s1", "nope"]})
    finally:
        dashboard.get_session_store = original
    assert r.status_code == 422
    # s1 exists — only the unknown id is reported.
    detail = r.json()["detail"]
    assert "nope" in detail and "not found: s1" not in detail
