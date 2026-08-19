# swarn dashboard API — frontend integration reference

Base URL: `http://localhost:8420` (`swarn serve --port 8420`).
Interactive schema: `GET /docs` (FastAPI's built-in Swagger UI).

CORS is enabled for `http://localhost:3000` and `http://127.0.0.1:3000` by
default. Set `SWARN_CORS_ORIGINS` (comma-separated) for other origins.

## Recommended flow for the Next.js frontend

1. (optional) `POST /api/upload` the user's data files → get `data_dir`
2. `POST /api/jobs` with `{task, method, data_dir, steps}` → get `id` immediately
3. Open `WS /ws/live` (or poll `GET /api/jobs/{id}?since=N`) for progress
4. When status is `complete`: fetch outputs —
   - AIDE: `GET /api/runs/{run_id}` (report) and `/api/runs/{run_id}/files`
   - ReAct: `GET /api/sessions/{session_id}` (trace) and `/api/workspace/files`

---

## Jobs (async runs — use these, not the legacy `POST /api/run`)

### `POST /api/jobs`
Submit a run. Returns immediately; the run executes on a background thread.

```json
{
  "task":     "Predict movie ratings; report RMSE.",   // required
  "method":   "react",       // "react" | "aide" | "team" (default "react")
  "data_dir": "/abs/path",   // from /api/upload; REQUIRED for "aide"
  "steps":    10,            // aide only: search budget (nodes)
  "model":    ""             // display only — calls route to the deployed endpoint
}
```

Response `200` — job summary:

```json
{
  "id": "a1b2c3d4e5f6",
  "task": "...",
  "method": "aide",
  "status": "queued",              // queued | running | complete | failed | cancelled
  "created": 1787114588.1,
  "started": null,
  "finished": null,
  "cancel_requested": false,
  "session_id": null,              // react/team: set shortly after start
  "run_id": null,                  // aide: set when the search finishes
  "n_events": 0,
  "last_event": null
}
```

Errors: `422` for a bad method, missing/invalid `data_dir`.

Method semantics:
- `react` — the ReAct AgentLoop (full tool set). Live steps stream on the
  websocket under `channel: "session"` once `session_id` is known.
- `aide` — solution-tree search over `data_dir`. Per-node progress arrives as
  job events (`type: "node"`).
- `team` — Planner→Coder→Reviewer→Tester pipeline. No mid-run cancel.

### `GET /api/jobs`
`{"jobs": [<summary>, ...]}` — newest first.

### `GET /api/jobs/{id}?since=N`
Summary plus `result`, `error`, and `events[N:]`. Poll incrementally by
passing the previous response's `n_events` as `since`.

Event shapes (all include `ts`):
```json
{"type": "status", "status": "running"}
{"type": "session", "session_id": "…"}                       // react
{"type": "node", "step": 3, "stage": "draft", "is_buggy": false,
 "metric": 0.91, "best_metric": 0.93, "note": "…"}           // aide
{"type": "cancel_requested"}
{"type": "status", "status": "complete", "result": {…}, "error": null}  // final
```

`result` when complete:
- react/team: `{"outcome": "complete", "summary": "…", "session_id": "…"}`
  (team: `{"final_outcome": …, "report_markdown": …}`)
- aide: `{"run_id": "…", "steps_done": 10, "best_metric": 0.93,
  "solution_path": "…", "report_path": "…"}`

### `POST /api/jobs/{id}/cancel`
Cooperative: react stops before its next step; aide stops at the next
completed node; team runs to completion (the response carries a `note`).
Final status becomes `"cancelled"`.

---

## Uploads

### `POST /api/upload`  (multipart/form-data, field name `files`, repeatable)
Saves to `workspace/uploads/<timestamp>/`. Response:

```json
{"data_dir": "/…/workspace/uploads/1787114588",
 "relative_dir": "uploads/1787114588",
 "files": ["movies.csv"]}
```

### `GET /api/uploads`
All previously uploaded batches (newest first) so users can re-use a dataset
without re-uploading: `{"uploads": [{"batch", "data_dir", "relative_dir",
"created", "files": [{"name", "size"}]}]}`.

### `DELETE /api/uploads/{batch}`
Deletes one batch directory. `404` if unknown.

---

## History & artifacts

### `GET /api/sessions?limit=20` / `GET /api/sessions/{id}`
ReAct session index / one session's full step trace.

### `GET /api/runs?limit=50` / `GET /api/runs/{id}`
AIDE run index (`run_id`, `nodes`, `best_metric`) / one run's full
`journal` (the solution tree) + `report_markdown`.

### `GET /api/runs/{id}/files` and `GET /api/runs/{id}/files/{path}`
List (recursive, `path` + `size`) and download a run's artifacts —
`report.md`, `best_solution.py`, `workspace/submission.csv`, etc.
Path traversal is rejected with `403`.

### `GET /api/workspace/files?path=` and `GET /api/workspace/file?path=`
Browse (non-recursive; entries have `name`, `is_dir`, `size`) and download
files from the ReAct agent's workspace — e.g. `plots/…png` for chart images.

### `GET /api/playbook`
`{"playbook": "…"}` — cross-run learned lessons (markdown-ish text).

---

## `WS /ws/live`

One socket, two channels, JSON text frames:

```json
// every ReAct/team agent step, as it happens
{"channel": "session", "session_id": "…", "task": "…",
 "kind": "plan|tool_call|tool_result|correction|complete|error",
 "timestamp": 1787114588.1, "data": {…}}

// every job lifecycle/progress event (incl. AIDE nodes)
{"channel": "job", "job_id": "…", "method": "aide", "status": "running",
 "task": "…", "event": {…}}   // event shapes as in GET /api/jobs/{id}
```

Correlate: a job's `session_id` (from its `session` event or job summary)
matches the `session_id` on `channel: "session"` frames. The server never
expects incoming messages; send anything (e.g. "ping") only to keep
intermediaries from idling out. No history replay — fetch state via REST
after (re)connecting, then stream.

## Notes

- No auth: the API executes tasks and reads the workspace — keep it bound to
  localhost or put it behind your own auth proxy before exposing it.
- `POST /api/run` still exists for the built-in page but blocks until the
  run finishes — don't use it from the frontend.
