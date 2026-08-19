"""
[frontend-api] NEW FILE: async job registry for the dashboard API.

The original POST /api/run blocks until the whole agent run finishes —
unusable from a real frontend (a Next.js page can't hold one fetch open
for a 30-minute AIDE search, can't cancel it, can't show two runs at
once). This module gives dashboard.py the same task-lifecycle shape
agent/integrations/mcp_server.py already uses for MCP clients
(submit → id, poll status, read messages), generalized:

  registry.submit(...)  → Job, runner thread started, returns immediately
  registry.get(id)      → status + result + buffered progress events
  registry.cancel(id)   → cooperative cancel (see per-method notes below)
  registry.add_listener → dashboard.py subscribes to push every job event
                           over the /ws/live websocket as it happens

Three execution methods:
  "react"  AgentLoop (Phase 1–15 toolset). Cancellable between steps via
           AgentLoop.run(stop_event=...); its per-step feed reaches the
           websocket through SessionStore's subscriber hook as before —
           the job carries session_id (reported via on_session as soon as
           the session exists) so the frontend can correlate the two.
  "aide"   run_search solution-tree search. Progress comes from its
           on_step callback (per-node events buffered on the job AND
           pushed to listeners). Cancel works by raising JobCancelled
           from inside on_step at the next completed node.
  "team"   Orchestrator (Planner→Coder→Reviewer→Tester). No mid-run
           cancel — the orchestrator drives its own internal AgentLoops
           and exposes no stop hook; cancel() marks the request and the
           job runs to completion.

In-process and in-memory by design — same singleton reasoning as
SessionStore: jobs submitted here run in the dashboard's process, which
is exactly what makes their live steps observable on this process's
websocket (see dashboard.py's module docstring).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent.paths import WORKSPACE_DIR

VALID_METHODS = ("react", "aide", "team")


class JobCancelled(Exception):
    """Raised inside a runner thread to abort a cancelled job."""


def attach_data_note(task: str, data_dir: Optional[str]) -> str:
    """
    Append the uploaded data's workspace-relative location to a ReAct/team
    task. The agent's file tools resolve paths relative to WORKSPACE_DIR,
    so this is how the model finds out where the uploaded files live.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return task
    rel = os.path.relpath(data_dir, WORKSPACE_DIR)
    names = ", ".join(sorted(os.listdir(data_dir))) or "(empty)"
    return task + f"\n\nData files for this task are in the workspace directory '{rel}': {names}"


@dataclass
class Job:
    id: str
    task: str
    method: str                          # react | aide | team
    status: str = "queued"               # queued | running | complete | failed | cancelled
    data_dir: Optional[str] = None
    steps: int = 10                      # aide only: search budget
    model: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    cancel_requested: bool = False
    session_id: Optional[str] = None     # react/team — known shortly after start
    run_id: Optional[str] = None         # aide — known when the search returns
    result: Optional[dict] = None
    error: Optional[str] = None
    events: list = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "method": self.method,
            "status": self.status,
            "created": self.created,
            "started": self.started or None,
            "finished": self.finished or None,
            "cancel_requested": self.cancel_requested,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "n_events": len(self.events),
            "last_event": self.events[-1] if self.events else None,
        }

    def detail(self, since: int = 0) -> dict:
        d = self.summary()
        d["result"] = self.result
        d["error"] = self.error
        d["events"] = self.events[since:]
        d["events_from"] = since
        return d


class JobRegistry:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._listeners: list[Callable[[Job, dict], None]] = []

    # ── observation ────────────────────────────────────────────────────

    def add_listener(self, cb: Callable[[Job, dict], None]) -> None:
        """cb(job, event) fires synchronously, from the runner thread, for
        every event. Must not raise (defended anyway)."""
        self._listeners.append(cb)

    def _emit(self, job: Job, event: dict) -> None:
        event = {"ts": time.time(), **event}
        job.events.append(event)
        for cb in self._listeners:
            try:
                cb(job, event)
            except Exception:  # noqa: BLE001 — a broken listener must not kill the job
                pass

    # ── lifecycle ──────────────────────────────────────────────────────

    def submit(self, task: str, method: str = "react",
               data_dir: Optional[str] = None, steps: int = 10,
               model: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], task=task, method=method,
                  data_dir=data_dir, steps=steps, model=model)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True,
                         name=f"job-{job.id}").start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in ("queued", "running"):
            job.cancel_requested = True
            job._stop.set()
            self._emit(job, {"type": "cancel_requested"})
        return job

    # ── runner (one daemon thread per job) ─────────────────────────────

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.started = time.time()
        self._emit(job, {"type": "status", "status": "running"})
        try:
            if job.method == "aide":
                job.result = self._run_aide(job)
            elif job.method == "team":
                job.result = self._run_team(job)
            else:
                job.result = self._run_react(job)
            # a cooperative react cancel comes back as a normal result
            # with outcome "cancelled" rather than an exception
            if job.cancel_requested and (job.result or {}).get("outcome") == "cancelled":
                job.status = "cancelled"
            else:
                job.status = "complete"
        except JobCancelled:
            job.status = "cancelled"
        except Exception as e:  # noqa: BLE001 — job must record any failure
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
        finally:
            job.finished = time.time()
            self._emit(job, {"type": "status", "status": job.status,
                             "result": job.result, "error": job.error})

    def _run_react(self, job: Job) -> dict:
        from agent.core.agent_loop import AgentLoop
        from agent.core.self_correction import SelfCorrectionPolicy
        from agent.llm import DEFAULT_MODEL
        from agent.observability.observability import GuardrailPolicy

        def on_session(session):
            job.session_id = session.id
            self._emit(job, {"type": "session", "session_id": session.id})

        agent = AgentLoop(
            model=job.model or DEFAULT_MODEL,
            correction_policy=SelfCorrectionPolicy(),
            guardrail_policy=GuardrailPolicy(),
        )
        return agent.run(attach_data_note(job.task, job.data_dir),
                         stop_event=job._stop, on_session=on_session)

    def _run_aide(self, job: Job) -> dict:
        from agent.search import SearchConfig, run_search

        def on_step(node, journal):
            if job._stop.is_set():
                raise JobCancelled()
            best = journal.best_node()
            self._emit(job, {
                "type": "node",
                "step": node.step,
                "stage": node.stage,
                "is_buggy": node.is_buggy,
                "metric": node.metric,
                "best_metric": best.metric if best else None,
                "note": ((node.analysis or node.plan) or "")[:300],
            })

        result = run_search(job.task, data_dir=job.data_dir,
                            config=SearchConfig(steps=job.steps),
                            on_step=on_step)
        job.run_id = result.run_id
        return {
            "method": "aide",
            "run_id": result.run_id,
            "steps_done": result.steps_done,
            "best_metric": result.best.metric if result.best else None,
            "solution_path": result.solution_path if result.best else None,
            "report_path": result.report_path,
        }

    def _run_team(self, job: Job) -> dict:
        from agent.core.orchestrator import Orchestrator
        from agent.llm import DEFAULT_MODEL
        from agent.observability.observability import GuardrailPolicy

        orchestrator = Orchestrator(
            model=job.model or DEFAULT_MODEL,
            guardrail_policy=GuardrailPolicy(),
        )
        return orchestrator.run(attach_data_note(job.task, job.data_dir))


_registry: Optional[JobRegistry] = None


def get_job_registry() -> JobRegistry:
    """Per-process singleton, same pattern as memory.py's get_session_store()."""
    global _registry
    if _registry is None:
        _registry = JobRegistry()
    return _registry
