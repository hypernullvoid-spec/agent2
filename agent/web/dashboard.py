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
SessionStore the websocket manager is subscribed to. The `swarn` REPL
remains what it always was: a separate, simpler way to run the agent
interactively, without live dashboard streaming. If you want to *watch*
a run live, trigger it through the dashboard (the page's "Run a task"
box, or `POST /api/run`) — if you want the ordinary interactive REPL
experience, use `swarn` or `swarn run` as before; you just won't see
those particular runs appear in the live feed, only in the session
history once they complete (recall_session/`/api/sessions/{id}` still
work for those, unaffected).

Endpoints
───────────
  GET  /                     — the dashboard page (single HTML file, no build step)
  GET  /api/sessions          — Phase 5's session index, as JSON
  GET  /api/sessions/{id}      — one session's full trace.json
  POST /api/run                 — trigger an agent run IN THIS PROCESS, streamed live
  WS   /ws/live                  — live step-by-step feed of runs triggered via /api/run

Running it
────────────
  python -m agent.web.dashboard          (dev, uses uvicorn's --reload-friendly run)
  swarn serve --port 8420               (Phase 16's CLI wrapper, see cli.py)

Note on scope
───────────────
This serves the dashboard; it does not, itself, run agent tasks. The
agent (whether via the interactive REPL or `swarn run`) is what
generates the steps the dashboard streams — this module is purely an
observability surface, the same "infrastructure, not a tool the agent
calls" framing Phase 15's ObservabilityHooks used for OTel tracing.
"""

import asyncio
import json
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.llm import DEFAULT_MODEL  # deployed model (see agent/llm/router.py)
from agent.memory import get_session_store, StepKind

app = FastAPI(title="swarn dashboard (Phase 16)")


class RunRequest(BaseModel):
    task: str
    model: str = DEFAULT_MODEL  # display only — calls hard-route to the deployed endpoint


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
        if self._loop is None or self._queue is None:
            return   # dashboard process hasn't finished starting up yet — drop silently
        payload = {
            "session_id": session.id,
            "task":       session.task[:120],
            "kind":       step.kind.value if isinstance(step.kind, StepKind) else str(step.kind),
            "timestamp":  step.timestamp,
            "data":       step.data,
        }
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


@app.on_event("startup")
async def _startup():
    manager.bind_to_running_loop(asyncio.get_running_loop())
    get_session_store().subscribe_to_all_sessions(manager.on_step)
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
    from agent.paths import RUNS_DIR
    return RUNS_DIR


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
    from agent.core.agent_loop import AgentLoop
    from agent.core.self_correction import SelfCorrectionPolicy
    from agent.observability import GuardrailPolicy

    agent = AgentLoop(
        model=body.model,
        correction_policy=SelfCorrectionPolicy(),
        guardrail_policy=GuardrailPolicy(),
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, agent.run, body.task)
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

document.getElementById('run-btn').onclick = async () => {
  const task = document.getElementById('task-input').value.trim();
  if (!task) return;
  feed.innerHTML = '';
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Running…';
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task}),
    });
    const result = await res.json();
    appendStep({session_id: result.session_id, task, kind: 'complete', timestamp: Date.now()/1000, data: result});
    loadSessions();
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run & watch live';
  }
};
</script>
</body>
</html>
"""
