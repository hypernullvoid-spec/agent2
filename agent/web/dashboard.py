"""
Phase 16: Web Dashboard

A FastAPI app giving the "VS Code output panel" experience the original
HeyNeo platform has, but for this project: a live view of whatever the
agent is doing right now, streamed over a websocket, plus a browser for
past sessions (Phase 5's trace.json/summary.md files).

Why a websocket, and why the live data has to come from memory.py's new
subscriber hook, not from polling trace.json
─────────────────────────────────────────────────────────────────────────────
Phase 5's SessionStore only ever writes trace.json/summary.md once, at
close_session() — i.e. only after a run is already finished. There is
no file to poll for a session that's still in progress; one simply
doesn't exist yet. A dashboard that polled the filesystem would only
ever show *already-completed* runs, never anything live — which would
make "live dashboard" a contradiction in terms.

The actual fix (made in memory.py, not here) is a small, additive
pub/sub hook: SessionStore.subscribe_to_all_sessions(callback) registers
a callback that Session.add_step() fires synchronously on every single
step, for every session created from that point on — regardless of
whether/when that session is ever persisted to disk. This module's
ConnectionManager is the actual subscriber: it registers itself with
the global SessionStore once, at startup, and fans out every step it
receives to whichever websocket clients are currently connected. A step
that happens while zero clients are connected is simply not delivered
anywhere live (no buffering) — it'll still show up later via the normal
recall_session()/trace.json path once the run completes, the same as
it always has. The dashboard adds a live view; it doesn't change what
gets persisted or when.

Why a separately-launched `swarn run` process can't stream to this dashboard
─────────────────────────────────────────────────────────────────────────────
get_session_store() (memory.py) is a per-process singleton — `_store`
is a plain module-level global, fresh in every Python interpreter. If
you run `swarn serve` in one terminal and `swarn run "task"` in a *different*
terminal, that's two separate OS processes with two completely separate
SessionStore instances; the dashboard's subscriber, registered on
*its own* process's store, will never see steps added by the *other*
process's store. No in-process pub/sub mechanism can bridge that gap
without an external broker (Redis, a socket, etc.) — which would be
real infrastructure this project's "no extra dependencies beyond what's
necessary" stance deliberately avoids adding for one feature.

The fix used here: the dashboard exposes its own `/api/run` endpoint
that triggers an agent run *inside the dashboard's own process* (via a
background asyncio task wrapping a thread, since AgentLoop.run() is
synchronous) — so the steps genuinely happen in the same process whose
SessionStore the websocket manager is subscribed to. `main.py`'s REPL
remains what it always was: a separate, simpler way to run the agent
interactively, without live dashboard streaming. If you want to *watch*
a run live, trigger it through the dashboard (the page's "Run a task"
box, or `POST /api/run`) — if you want the ordinary interactive REPL
experience, use `main.py` or `swarn run` as before; you just won't see
those particular runs appear in the live feed, only in the session
history once they complete (recall_session/`/api/sessions/{id}` still
work for those, unaffected).

Endpoints (full request/response reference: agent/web/API.md, or /docs)
───────────
  GET    /                       — the built-in dashboard page (single HTML file)
  WS     /ws/live                — live feed: session steps + job events

  POST   /api/jobs               — submit a run (react/aide/team), returns immediately
  GET    /api/jobs               — list jobs, newest first
  GET    /api/jobs/{id}          — job status + result + progress events (?since=N)
  POST   /api/jobs/{id}/cancel   — cooperative cancel

  POST   /api/upload             — upload data file(s) → workspace/uploads/<ts>/
  GET    /api/uploads            — list uploaded dataset batches (for re-use)
  DELETE /api/uploads/{batch}    — delete an uploaded batch

  GET    /api/sessions           — Phase 5's session index (ReAct history)
  GET    /api/sessions/{id}      — one session's full trace.json
  GET    /api/runs               — AIDE search-run index
  GET    /api/runs/{id}          — one run's journal + report markdown
  GET    /api/runs/{id}/files    — list a run's artifact files
  GET    /api/runs/{id}/files/{path} — download one run artifact
  GET    /api/workspace/files    — browse the agent workspace (?path=subdir)
  GET    /api/workspace/file     — download a workspace file (?path=...)
  GET    /api/playbook           — cross-run learned lessons

  POST   /api/run                — LEGACY blocking run (kept for the built-in page)

Running it
────────────
  python -m agent.web.dashboard          (dev, uses uvicorn's --reload-friendly run)
  swarn serve --port 8420               (Phase 16's CLI wrapper, see cli.py)

Note on scope
───────────────
This serves the dashboard; it does not, itself, run agent tasks. The
agent (whether via main.py's REPL or cli.py's `swarn run`) is what
generates the steps the dashboard streams — this module is purely an
observability surface, the same "infrastructure, not a tool the agent
calls" framing Phase 15's ObservabilityHooks used for OTel tracing.
"""

import asyncio
import json
import os
import time
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from agent.llm import DEFAULT_MODEL  # deployed model (see agent/llm/router.py)
from agent.memory.memory import get_session_store, StepKind
from agent.paths import WORKSPACE_DIR, safe_filename
from agent.web.jobs import VALID_METHODS, attach_data_note, get_job_registry, seed_history

app = FastAPI(title="swarn dashboard (Phase 16)")

# [frontend-api] NEW: CORS for the separate Next.js frontend (dev server on
# :3000 by default) — without this, the browser blocks the frontend's fetch()
# calls because it runs on a different origin than this API.
# Override with a comma-separated SWARN_CORS_ORIGINS env var in deployment.
_cors_origins = [o.strip() for o in os.environ.get(
    "SWARN_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_SUBDIR = "uploads"   # inside WORKSPACE_DIR, so ReAct's workspace-rooted tools can read them


class RunRequest(BaseModel):
    task: str
    model: str = DEFAULT_MODEL  # display only — calls hard-route to the deployed endpoint
    # [frontend-api] NEW fields: let the caller pick the engine and attach an
    # uploaded dataset (used by the legacy blocking /api/run endpoint below).
    method: str = "react"       # "react" (AgentLoop) or "aide" (solution-tree search)
    data_dir: Optional[str] = None  # absolute path from /api/upload, or any local dir
    steps: int = 10             # AIDE only: search budget (nodes to try)
    # react only: completed session ids (oldest first) whose transcripts seed
    # this run's context — how a follow-up continues an earlier conversation.
    resume_session_ids: List[str] = []


class ConnectionManager:
    """
    Holds the set of currently-connected websocket clients and fans out
    every live step to all of them. Registers itself with SessionStore
    exactly once (see module docstring) — this is the bridge between
    "a step just happened, synchronously, inside whatever thread/process
    is running the agent loop" and "an async websocket client wants to
    hear about it."

    Threading note: Session.add_step() (and therefore this manager's
    _on_step callback) runs on whatever thread the agent loop itself is
    running on — which is NOT necessarily the same thread/event loop
    FastAPI's websocket connections live on if the agent and the
    dashboard are run as separate processes (the common case: `swarn
    serve` in one terminal, `swarn run "..."` in another). Because of that,
    _on_step cannot directly await an async websocket send from a
    synchronous callback — instead it hands the step off to a thread-safe
    queue, and a background asyncio task drains that queue and does the
    actual async broadcasting. This is the same kind of sync→async
    bridge problem Phase 12's mcp_integration.py solved for a different
    reason (calling async MCP code from a sync tool registry); here it's
    the mirror image (a sync callback needing to feed an async consumer).
    """

    def __init__(self):
        self.active: set[WebSocket] = set()
        self._queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_to_running_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once, at FastAPI startup, from inside the running event loop."""
        self._loop = loop
        self._queue = asyncio.Queue()

    def on_step(self, session, step) -> None:
        """
        The actual SessionStore subscriber callback — synchronous, may
        be called from a different thread than the dashboard's event
        loop. Never raises (per memory.py's contract that a broken
        subscriber must not crash the agent run) — wrapped in try/except
        as a second layer of defense even though add_step() already
        catches subscriber exceptions itself.
        """
        payload = {
            "channel":    "session",
            "session_id": session.id,
            "task":       session.task[:120],
            "kind":       step.kind.value if isinstance(step.kind, StepKind) else str(step.kind),
            "timestamp":  step.timestamp,
            "data":       step.data,
        }
        self.push(payload)

    # [frontend-api] NEW: on_step above used to do this queue hand-off itself;
    # the logic was extracted into push() so job events (AIDE node progress,
    # status changes) can ride the same websocket as session steps. Session
    # payloads also gained a "channel": "session" field so the frontend can
    # tell the two frame types apart.
    def push(self, payload: dict) -> None:
        """Thread-safe fire-and-forget broadcast of any payload to all
        websocket clients. Never raises; drops silently before startup."""
        if self._loop is None or self._queue is None:
            return   # dashboard process hasn't finished starting up yet — drop silently
        try:
            asyncio.run_coroutine_threadsafe(self._queue.put(payload), self._loop)
        except Exception:
            pass

    async def broadcast_loop(self) -> None:
        """Background task: drain the queue, send each item to every connected client."""
        while True:
            payload = await self._queue.get()
            text = json.dumps(payload, default=str)
            dead = []
            for ws in self.active:
                try:
                    await ws.send_text(text)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.active.discard(ws)


manager = ConnectionManager()


# [frontend-api] NEW: bridge from the job registry to the websocket. Every
# progress event a background job emits (see agent/web/jobs.py) is wrapped in
# a "channel": "job" frame and broadcast live to all connected clients.
def _on_job_event(job, event) -> None:
    """JobRegistry listener → websocket. Runs on the job's runner thread."""
    manager.push({
        "channel": "job",
        "job_id":  job.id,
        "method":  job.method,
        "status":  job.status,
        "task":    job.task[:120],
        "event":   event,
    })


@app.on_event("startup")
async def _startup():
    manager.bind_to_running_loop(asyncio.get_running_loop())
    get_session_store().subscribe_to_all_sessions(manager.on_step)
    # [frontend-api] NEW: subscribe the websocket to job events as well
    get_job_registry().add_listener(_on_job_event)
    asyncio.create_task(manager.broadcast_loop())


# ───────────────────────────────────────────────── REST endpoints

@app.get("/api/sessions")
def api_sessions(limit: int = 20):
    """Phase 5's session index (most recent first), as JSON."""
    store = get_session_store()
    return {"sessions": store._index[:limit]}


@app.get("/api/sessions/{session_id}")
def api_session_detail(session_id: str):
    """One session's full trace.json — only available once the run has completed."""
    store = get_session_store()
    data = store.get_session(session_id)
    if data is None:
        return {"error": f"No completed session found matching '{session_id}'. "
                          "It may still be running — watch /ws/live instead."}
    return data


# ── V2: solution-tree search runs (runs/<id>/journal.json + report.md) ──

def _runs_dir() -> str:
    import os
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))


@app.get("/api/runs")
def api_runs(limit: int = 50):
    """List solution-search runs (newest first) with their best metric."""
    import os
    root = _runs_dir()
    out = []
    if os.path.isdir(root):
        for rid in sorted(os.listdir(root), reverse=True)[:limit]:
            jpath = os.path.join(root, rid, "journal.json")
            entry = {"run_id": rid, "nodes": 0, "best_metric": None}
            if os.path.isfile(jpath):
                try:
                    from agent.search.journal import Journal
                    j = Journal.load(jpath)
                    best = j.best_node()
                    entry["nodes"] = len(j)
                    entry["best_metric"] = best.metric if best else None
                except Exception:  # noqa: BLE001 — a corrupt journal shouldn't 500 the list
                    pass
            out.append(entry)
    return {"runs": out}


@app.get("/api/runs/{run_id}")
def api_run_detail(run_id: str):
    """One run's journal (full tree) plus its report markdown."""
    import json as _json
    import os
    rdir = os.path.join(_runs_dir(), run_id)
    if not os.path.isdir(rdir):
        return {"error": f"no run '{run_id}'"}
    result: dict = {"run_id": run_id}
    jpath = os.path.join(rdir, "journal.json")
    if os.path.isfile(jpath):
        with open(jpath, encoding="utf-8") as f:
            result["journal"] = _json.load(f)
    rpath = os.path.join(rdir, "report.md")
    if os.path.isfile(rpath):
        with open(rpath, encoding="utf-8") as f:
            result["report_markdown"] = f.read()
    return result


@app.get("/api/playbook")
def api_playbook():
    """V3: the cross-run playbook — lessons the agent learned from past runs."""
    try:
        from agent.memory.knowledge import KnowledgeStore
        return {"playbook": KnowledgeStore().playbook()}
    except Exception:  # noqa: BLE001
        return {"playbook": ""}


# [frontend-api] NEW endpoint: file upload from the browser ("give the
# flexibility to add the file the user wants to analyze data on").
@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    """
    Save uploaded data files into workspace/uploads/<timestamp>/ and return
    that directory. The returned `data_dir` is what /api/run expects:
    AIDE passes it to run_search() as the task's data directory; ReAct
    gets it appended to the task text as a workspace-relative path (the
    agent's file tools are rooted at WORKSPACE_DIR, so uploads must live
    inside it to be reachable).
    """
    batch = str(int(time.time()))
    dest = os.path.join(WORKSPACE_DIR, UPLOADS_SUBDIR, batch)
    os.makedirs(dest, exist_ok=True)
    saved = []
    for f in files:
        # safe_filename() strips dots (it was written for directory names),
        # which would break extensions ("movies.csv" → "movies_csv") — so
        # sanitize the stem and extension separately and keep the dot.
        stem, ext = os.path.splitext(os.path.basename(f.filename or "upload.dat"))
        name = safe_filename(stem) or "upload"
        if ext:
            name += "." + safe_filename(ext.lstrip("."))
        path = os.path.join(dest, name)
        with open(path, "wb") as out:
            while chunk := await f.read(1 << 20):
                out.write(chunk)
        saved.append(name)
    return {
        "data_dir": dest,                                   # absolute — for AIDE
        "relative_dir": f"{UPLOADS_SUBDIR}/{batch}",        # workspace-relative — for ReAct
        "files": saved,
    }


# ── [frontend-api] NEW: async job API (what the Next.js frontend should use) ──
#
# Unlike the legacy POST /api/run (which blocks until the whole run finishes),
# these endpoints return immediately and let the frontend poll or stream —
# the actual work happens on background threads inside agent/web/jobs.py.
#
#   POST /api/jobs                submit → returns immediately with the job
#   GET  /api/jobs                 list all jobs, newest first
#   GET  /api/jobs/{id}?since=N     status + result + events[N:] (poll with
#                                   since=<last n_events> for increments;
#                                   or skip polling and watch /ws/live)
#   POST /api/jobs/{id}/cancel      cooperative cancel (react: between steps,
#                                   aide: at the next completed node,
#                                   team: unsupported — runs to completion)

class JobRequest(BaseModel):
    task: str
    method: str = "react"           # react | aide | team
    data_dir: Optional[str] = None  # from /api/upload (required for aide)
    steps: int = 10                 # aide search budget
    model: str = ""                 # display only — calls hard-route to the deployed endpoint
    # react only: completed session ids (oldest first) whose transcripts seed
    # this run's context. The server keeps no thread state between jobs, so a
    # follow-up passes EVERY session id in its conversation, not just the
    # latest — see jobs.seed_history for why.
    resume_session_ids: List[str] = []


@app.post("/api/jobs")
def api_job_submit(body: JobRequest):
    """Validate the request, hand it to the JobRegistry (which starts a
    background thread), and return the job summary right away — the frontend
    gets an id it can watch instead of a request that hangs for the whole run."""
    if body.method not in VALID_METHODS:
        raise HTTPException(422, f"method must be one of {VALID_METHODS}")
    if body.method == "aide" and (not body.data_dir or not os.path.isdir(body.data_dir)):
        raise HTTPException(422, "AIDE needs a data directory — upload files first "
                                 "and pass the returned data_dir")
    if body.data_dir and not os.path.isdir(body.data_dir):
        raise HTTPException(422, f"data_dir does not exist: {body.data_dir}")
    # Validated here, at submit time, rather than in the runner thread — a bad
    # session id should be a 422 the frontend can show, not a failed job.
    if body.resume_session_ids:
        if body.method != "react":
            raise HTTPException(422, "resume_session_ids only applies to method "
                                     "'react' — aide and team runs don't take "
                                     "conversation context")
        store = get_session_store()
        missing = [s for s in body.resume_session_ids if store.get_session(s) is None]
        if missing:
            raise HTTPException(422, f"resume session(s) not found: {', '.join(missing)} "
                                     "— a session's trace exists once its run has "
                                     "completed; check GET /api/sessions")
    job = get_job_registry().submit(task=body.task, method=body.method,
                                    data_dir=body.data_dir, steps=body.steps,
                                    model=body.model,
                                    resume_session_ids=body.resume_session_ids)
    return job.summary()


@app.get("/api/jobs")
def api_jobs():
    """All jobs submitted to this server process, newest first."""
    return {"jobs": [j.summary() for j in get_job_registry().list()]}


@app.get("/api/jobs/{job_id}")
def api_job_detail(job_id: str, since: int = 0):
    """One job's status + result + progress events. Pass ?since=<n_events
    from the previous poll> to receive only the events you haven't seen."""
    job = get_job_registry().get(job_id)
    if job is None:
        raise HTTPException(404, f"no job '{job_id}'")
    return job.detail(since=since)


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    """Cooperative cancel — react stops between steps, aide at the next
    finished node, team can't stop mid-pipeline (we say so in the response)."""
    job = get_job_registry().cancel(job_id)
    if job is None:
        raise HTTPException(404, f"no job '{job_id}'")
    if job.method == "team" and job.status == "running":
        return {**job.summary(),
                "note": "team runs can't be interrupted mid-run — cancel recorded, "
                        "but the pipeline will run to completion"}
    return job.summary()


# ── [frontend-api] NEW: uploaded datasets (list / delete) ───────────────
# Lets the frontend show previously uploaded files so users can re-run
# analyses on the same dataset without uploading it again every time.

@app.get("/api/uploads")
def api_uploads():
    """Every uploaded dataset batch, newest first — pass a batch's
    `data_dir` straight into POST /api/jobs to re-use it."""
    root = os.path.join(WORKSPACE_DIR, UPLOADS_SUBDIR)
    batches = []
    if os.path.isdir(root):
        for batch in sorted(os.listdir(root), reverse=True):
            bdir = os.path.join(root, batch)
            if not os.path.isdir(bdir):
                continue
            files = [{"name": n, "size": os.path.getsize(os.path.join(bdir, n))}
                     for n in sorted(os.listdir(bdir))]
            batches.append({
                "batch": batch,
                "data_dir": bdir,
                "relative_dir": f"{UPLOADS_SUBDIR}/{batch}",
                "created": os.path.getmtime(bdir),
                "files": files,
            })
    return {"uploads": batches}


@app.delete("/api/uploads/{batch}")
def api_upload_delete(batch: str):
    """Remove one uploaded dataset batch. The safe_filename() equality check
    rejects anything with slashes/dots so a crafted batch name can't reach
    outside workspace/uploads/."""
    import shutil
    if batch != safe_filename(batch):
        raise HTTPException(422, "invalid batch name")
    bdir = os.path.join(WORKSPACE_DIR, UPLOADS_SUBDIR, batch)
    if not os.path.isdir(bdir):
        raise HTTPException(404, f"no upload batch '{batch}'")
    shutil.rmtree(bdir)
    return {"deleted": batch}


# ── [frontend-api] NEW: artifacts — browse & download run outputs ───────
# The whole point of a run is its outputs (report.md, best_solution.py,
# submission.csv, plots). These endpoints let the frontend list and download
# them; every path goes through _guarded_join so a request can never read
# files outside the runs/ or workspace/ directories.

def _guarded_join(root: str, rel_path: str) -> str:
    """Resolve rel_path inside root; 403 on traversal attempts."""
    full = os.path.realpath(os.path.join(root, rel_path))
    if full != os.path.realpath(root) and not full.startswith(os.path.realpath(root) + os.sep):
        raise HTTPException(403, "path escapes the allowed directory")
    return full


@app.get("/api/runs/{run_id}/files")
def api_run_files(run_id: str):
    """Recursive file listing of one search run's directory (journal,
    report, best_solution.py, workspace outputs like submission.csv)."""
    rdir = _guarded_join(_runs_dir(), run_id)
    if not os.path.isdir(rdir):
        raise HTTPException(404, f"no run '{run_id}'")
    files = []
    for dirpath, _dirnames, filenames in os.walk(rdir):
        for name in filenames:
            full = os.path.join(dirpath, name)
            files.append({"path": os.path.relpath(full, rdir),
                          "size": os.path.getsize(full)})
            if len(files) >= 500:
                return {"run_id": run_id, "files": files, "truncated": True}
    return {"run_id": run_id, "files": sorted(files, key=lambda f: f["path"])}


@app.get("/api/runs/{run_id}/files/{file_path:path}")
def api_run_file(run_id: str, file_path: str):
    """Download one file from a run directory."""
    full = _guarded_join(_runs_dir(), os.path.join(run_id, file_path))
    if not os.path.isfile(full):
        raise HTTPException(404, f"no file '{file_path}' in run '{run_id}'")
    return FileResponse(full, filename=os.path.basename(full))


@app.get("/api/workspace/files")
def api_workspace_files(path: str = ""):
    """Non-recursive listing of a workspace directory (ReAct outputs,
    plots/, uploads/…). `path` is workspace-relative; empty = root."""
    full = _guarded_join(WORKSPACE_DIR, path or ".")
    if not os.path.isdir(full):
        raise HTTPException(404, f"no workspace directory '{path}'")
    entries = []
    for name in sorted(os.listdir(full)):
        p = os.path.join(full, name)
        entries.append({"name": name, "is_dir": os.path.isdir(p),
                        "size": os.path.getsize(p) if os.path.isfile(p) else None})
    return {"path": path, "entries": entries}


@app.get("/api/workspace/file")
def api_workspace_file(path: str):
    """Download one workspace file (e.g. plots/chart.png, submission.csv)."""
    full = _guarded_join(WORKSPACE_DIR, path)
    if not os.path.isfile(full):
        raise HTTPException(404, f"no workspace file '{path}'")
    return FileResponse(full, filename=os.path.basename(full))


# ── [frontend-api] legacy blocking endpoint — kept only so the built-in
# HTML page below keeps working; a real frontend should use /api/jobs ──

@app.post("/api/run")
async def api_run(body: RunRequest):
    """
    Trigger an agent run IN THIS PROCESS and stream its steps live over
    every connected websocket. This is the only way to get a genuinely
    live feed in this dashboard — see the module docstring's section on
    why a separately-launched `swarn run` process cannot stream here.

    AgentLoop.run() is synchronous and blocking (it makes real,
    blocking Anthropic API calls) — running it directly on FastAPI's
    event loop would freeze every other request (including the
    websocket broadcast loop) for the run's entire duration. Offloaded
    to a thread via run_in_executor so the event loop stays responsive;
    the live steps still arrive via the same thread-safe
    asyncio.run_coroutine_threadsafe bridge ConnectionManager.on_step
    already uses for exactly this reason.

    Returns immediately with the session ID FastAPI assigns before the
    run starts producing steps is impossible here — there's no way to
    know the session's UUID before AgentLoop.run() creates it
    internally — so this endpoint blocks until the run finishes and
    returns the final outcome. Watch /ws/live from the moment you call
    this (or just before) to see the steps as they happen rather than
    only the final result.
    """
    loop = asyncio.get_running_loop()

    if body.method == "aide":
        # [frontend-api] NEW branch: AIDE-style solution-tree search (same
        # engine as `swarn solve`), selected by the page's method radio.
        # Its progress lives in runs/<id>/journal.json, not SessionStore,
        # so it shows up in the "Search runs" list rather than /ws/live.
        if not body.data_dir or not os.path.isdir(body.data_dir):
            return {"error": f"AIDE needs a data directory — upload a file first "
                             f"(got: {body.data_dir!r})"}
        from agent.search import SearchConfig, run_search

        def _solve():
            return run_search(body.task, data_dir=body.data_dir,
                              config=SearchConfig(steps=body.steps))
        result = await loop.run_in_executor(None, _solve)
        return {
            "method": "aide",
            "run_id": result.run_id,
            "steps_done": result.steps_done,
            "best_metric": result.best.metric if result.best else None,
            "solution_path": result.solution_path if result.best else None,
            "report_path": result.report_path,
        }

    # default: ReAct (Phase 1–15 AgentLoop)
    from agent.core.agent_loop import AgentLoop
    from agent.core.self_correction import SelfCorrectionPolicy
    from agent.observability.observability import GuardrailPolicy

    # [frontend-api] NEW: if a dataset was uploaded, tell the agent where it
    # lives (workspace-relative — its file tools are rooted at WORKSPACE_DIR).
    task = attach_data_note(body.task, body.data_dir)

    if body.resume_session_ids:
        store = get_session_store()
        missing = [s for s in body.resume_session_ids if store.get_session(s) is None]
        if missing:
            return {"error": f"resume session(s) not found: {', '.join(missing)}"}

    agent = AgentLoop(
        model=body.model,
        correction_policy=SelfCorrectionPolicy(),
        guardrail_policy=GuardrailPolicy(),
        keep_history=bool(body.resume_session_ids),
    )
    if body.resume_session_ids:
        agent.history = seed_history(body.resume_session_ids)
    result = await loop.run_in_executor(None, agent.run, task)
    return result


# ───────────────────────────────────────────────── websocket

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    manager.active.add(websocket)
    try:
        while True:
            # Dashboard doesn't expect the client to send anything — this
            # just keeps the connection open and detects disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.active.discard(websocket)


# ───────────────────────────────────────────────── dashboard page

@app.get("/", response_class=HTMLResponse)
def dashboard_page():
    # A single self-contained HTML file, no build step — consistent
    # with this project's "no extra tooling beyond what's necessary"
    # stance (the FastAPI/React combo mentioned in the original
    # blueprint is one valid option; plain HTML+JS is the other option
    # the blueprint explicitly allows, and it's the simpler one to ship
    # as one file with zero npm/build dependencies).
    return _DASHBOARD_HTML


_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>swarn dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; }
  header { padding: 16px 24px; border-bottom: 1px solid #21262d; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  #status { font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #30363d; }
  #status.connected { background: #1a4d2e; color: #4ade80; }
  main { display: grid; grid-template-columns: 320px 1fr; height: calc(100vh - 57px); }
  #sidebar { border-right: 1px solid #21262d; overflow-y: auto; padding: 12px; }
  #sidebar h2 { font-size: 12px; text-transform: uppercase; color: #8b949e; margin: 8px 0; }
  .session-row { padding: 8px; border-radius: 6px; cursor: pointer; font-size: 13px; margin-bottom: 4px; }
  .session-row:hover { background: #161b22; }
  .session-row .outcome { font-size: 11px; padding: 1px 6px; border-radius: 8px; }
  .outcome-complete { background: #1a4d2e; color: #4ade80; }
  .outcome-other { background: #4d2e1a; color: #fbbf24; }
  #feed { overflow-y: auto; padding: 16px 24px; font-size: 13px; font-family: ui-monospace, monospace; }
  .step { padding: 6px 0; border-bottom: 1px solid #161b22; }
  .step .kind { color: #58a6ff; font-weight: 600; }
  .step .task { color: #8b949e; }
  .step pre { margin: 4px 0 0; white-space: pre-wrap; word-break: break-word; color: #c9d1d9; }
  .step.tool_call .kind { color: #d29922; }
  .step.tool_result .kind { color: #3fb950; }
  .step.correction .kind { color: #f85149; }
  .step.plan .kind { color: #a371f7; }
</style>
</head>
<body>
<header>
  <h1>swarn — live agent dashboard</h1>
  <span id="status">connecting…</span>
</header>
<main>
  <div id="sidebar">
    <h2>Run a task (this process)</h2>
    <textarea id="task-input" rows="3" style="width:100%;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:6px;font-family:inherit;font-size:12px"></textarea>
    <!-- [frontend-api] NEW controls: pick the engine (ReAct agent vs AIDE
         tree search), set AIDE's search budget, and attach data files -->
    <div style="display:flex;gap:12px;margin-top:6px;font-size:12px;align-items:center">
      <label style="cursor:pointer"><input type="radio" name="method" value="react" checked> ReAct</label>
      <label style="cursor:pointer"><input type="radio" name="method" value="aide"> AIDE</label>
      <span id="aide-opts" style="display:none;color:#8b949e">steps
        <input id="steps-input" type="number" value="10" min="1" max="100"
               style="width:48px;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:2px 4px">
      </span>
    </div>
    <input id="file-input" type="file" multiple
           style="margin-top:6px;width:100%;font-size:11px;color:#8b949e">
    <div id="upload-info" style="font-size:11px;color:#8b949e;margin-top:2px"></div>
    <button id="run-btn" style="margin-top:6px;width:100%;background:#238636;color:white;border:none;border-radius:6px;padding:6px;cursor:pointer">Run &amp; watch live</button>
    <h2 style="margin-top:16px">Recent sessions</h2>
    <div id="session-list">Loading…</div>
    <h2 style="margin-top:16px">Search runs (tree search)</h2>
    <div id="run-list">Loading…</div>
    <h2 style="margin-top:16px">Playbook <span style="text-transform:none;color:#6e7681">(learned lessons)</span></h2>
    <div id="playbook" style="font-size:12px;color:#8b949e;white-space:pre-wrap"></div>
  </div>
  <div id="feed"></div>
</main>
<script>
const feed = document.getElementById('feed');
const statusEl = document.getElementById('status');
const sessionList = document.getElementById('session-list');

function appendStep(payload) {
  const div = document.createElement('div');
  div.className = 'step ' + payload.kind;
  const time = new Date(payload.timestamp * 1000).toLocaleTimeString();
  div.innerHTML = `<span class="kind">${payload.kind}</span> `
                + `<span class="task">[${payload.session_id.slice(0,8)}] ${payload.task}</span> `
                + `<span style="color:#6e7681">${time}</span>`
                + `<pre>${JSON.stringify(payload.data, null, 2)}</pre>`;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/live`);
  ws.onopen = () => { statusEl.textContent = 'live'; statusEl.className = 'connected'; };
  ws.onclose = () => {
    statusEl.textContent = 'disconnected — retrying…';
    statusEl.className = '';
    setTimeout(connect, 2000);
  };
  ws.onmessage = (event) => appendStep(JSON.parse(event.data));
}

async function loadSessions() {
  const res = await fetch('/api/sessions?limit=20');
  const data = await res.json();
  if (!data.sessions.length) {
    sessionList.innerHTML = '<div style="color:#8b949e;font-size:12px">No completed sessions yet.</div>';
    return;
  }
  sessionList.innerHTML = data.sessions.map(s => {
    const outcomeClass = s.outcome === 'complete' ? 'outcome-complete' : 'outcome-other';
    return `<div class="session-row" onclick="loadDetail('${s.id}')">
              <div>${(s.task || '').slice(0, 40)}</div>
              <span class="outcome ${outcomeClass}">${s.outcome || '?'}</span>
              <span style="color:#6e7681;font-size:11px"> ${s.duration_s || '?'}s</span>
            </div>`;
  }).join('');
}

async function loadDetail(sessionId) {
  const res = await fetch(`/api/sessions/${sessionId}`);
  const data = await res.json();
  feed.innerHTML = '';
  if (data.error) {
    feed.innerHTML = `<div style="color:#8b949e">${data.error}</div>`;
    return;
  }
  for (const step of data.steps) {
    appendStep({ session_id: data.id, task: data.task, kind: step.kind, timestamp: step.timestamp, data: step.data });
  }
}


async function loadRuns() {
  const res = await fetch('/api/runs?limit=25');
  const data = await res.json();
  const el = document.getElementById('run-list');
  if (!data.runs.length) {
    el.innerHTML = '<div style="color:#8b949e;font-size:12px">No search runs yet — try `swarn solve`.</div>';
    return;
  }
  el.innerHTML = data.runs.map(r =>
    `<div class="session-row" onclick="loadRunDetail('${r.run_id}')">
       <div>${r.run_id}</div>
       <span style="color:#6e7681;font-size:11px">${r.nodes} nodes</span>
       ${r.best_metric !== null ? `<span class="outcome outcome-complete">best ${Number(r.best_metric).toPrecision(5)}</span>` : '<span class="outcome outcome-other">no solution</span>'}
     </div>`).join('');
}

async function loadRunDetail(runId) {
  const res = await fetch(`/api/runs/${runId}`);
  const data = await res.json();
  feed.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'step';
  div.innerHTML = `<span class="kind">search run</span> <span class="task">${runId}</span>`
                + `<pre>${(data.report_markdown || '(no report)').replace(/</g, '&lt;')}</pre>`;
  feed.appendChild(div);
}

async function loadPlaybook() {
  const res = await fetch('/api/playbook');
  const data = await res.json();
  document.getElementById('playbook').textContent =
    data.playbook || '(empty — fills up as search runs complete)';
}

connect();
loadSessions();
loadRuns();
loadPlaybook();
setInterval(loadSessions, 5000);
setInterval(loadRuns, 7000);

// [frontend-api] NEW: toggle the AIDE-only "steps" option with the method
// radio — AIDE needs a data directory, so a file is required only for it
document.querySelectorAll('input[name="method"]').forEach(r => r.onchange = () => {
  document.getElementById('aide-opts').style.display =
    document.querySelector('input[name="method"]:checked').value === 'aide' ? '' : 'none';
});

// [frontend-api] NEW: send the chosen files to POST /api/upload before the
// run starts; the returned data_dir is passed along in the /api/run body
async function uploadFiles() {
  const input = document.getElementById('file-input');
  if (!input.files.length) return null;
  const form = new FormData();
  for (const f of input.files) form.append('files', f);
  const res = await fetch('/api/upload', {method: 'POST', body: form});
  const data = await res.json();
  document.getElementById('upload-info').textContent =
    `uploaded ${data.files.join(', ')} → ${data.relative_dir}`;
  return data;
}

// [frontend-api] CHANGED: the run button now uploads any chosen files first,
// then submits {task, method, steps, data_dir} instead of just {task}
document.getElementById('run-btn').onclick = async () => {
  const task = document.getElementById('task-input').value.trim();
  if (!task) return;
  const method = document.querySelector('input[name="method"]:checked').value;
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Uploading…';
  try {
    const upload = await uploadFiles();
    if (method === 'aide' && !upload) {
      document.getElementById('upload-info').textContent = 'AIDE needs a data file — choose one first.';
      return;
    }
    feed.innerHTML = '';
    btn.textContent = 'Running…';
    const body = {task, method, steps: parseInt(document.getElementById('steps-input').value) || 10};
    if (upload) body.data_dir = upload.data_dir;
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const result = await res.json();
    appendStep({session_id: result.session_id || result.run_id || '--------', task,
                kind: 'complete', timestamp: Date.now()/1000, data: result});
    loadSessions();
    loadRuns();
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run & watch live';
  }
};
</script>
</body>
</html>
"""
