# 13 — APIs

Four public surfaces: the CLI, the dashboard HTTP/WS API, the MCP server tools, and the
programmatic Python API.

## 1. CLI (`swarn`, `agent/cli.py`)

| Command | Arguments / options | Exit code |
|---|---|---|
| `swarn run "<task>"` | `--model` (ignored for routing) | 0 iff outcome `complete`, else 1 |
| `swarn team "<task>"` | `--model`, `--no-tester` | 0 iff `complete`, else 1 |
| `swarn solve "<task>"` | `-d/--data DIR` (required unless `--resume`), `-s/--steps` (20), `-t/--time-limit`, `--drafts` (4), `-m/--model`, `--feedback-model`, `--exec-timeout` (600), `-w/--workers`, `--token-budget`, `--resume RUN_ID`, `--no-learn` | 0 iff a best node exists; 2 for missing data dir; else 1 |
| `swarn sessions` | `-n/--limit` (10) | 0 |
| `swarn recall <id-prefix>` | | 0 |
| `swarn index <path>` | | 0 |
| `swarn playbook` | `--clear` | 0 |
| `swarn guardrail-benchmark` | | 0 |
| `swarn serve` | `-p/--port` (8420), `--host` (127.0.0.1) | blocks |
| `swarn mcp-serve` | | blocks (stdio) |

Also reachable without installation: `python -m agent.cli <command>`.

## 2. Dashboard HTTP/WS API (`agent/dashboard.py`)

No authentication. Binds 127.0.0.1 by default.

| Method & path | Request | Response |
|---|---|---|
| `GET /` | — | Embedded single-file HTML dashboard |
| `GET /api/sessions?limit=20` | — | `{"sessions": [index entries]}` — id, task[:80], model, outcome, duration_s, tool_calls, corrections, started_at |
| `GET /api/sessions/{id}` | id = UUID prefix (≥4 chars) | Full `trace.json` dict, or `{"error": …}` if not yet completed |
| `POST /api/run` | `{"task": str, "model": str?}` | **Blocks until the run finishes** (run executes in a thread-pool executor); returns `{"outcome", "summary", "session_id"}`. Live steps stream on the websocket meanwhile |
| `WS /ws/live` | client sends anything to keep-alive | JSON per step: `{session_id, task, kind, timestamp, data}` — only for runs in *this* process |
| `GET /api/runs?limit=50` | — | `{"runs":[{run_id, nodes, best_metric}]}` from `runs/*/journal.json` (corrupt journals tolerated) |
| `GET /api/runs/{run_id}` | — | `{run_id, journal: {...}, report_markdown: str}` |
| `GET /api/playbook` | — | `{"playbook": str}` |

The embedded page polls `/api/sessions` (5s) and `/api/runs` (7s), auto-reconnects the
websocket (2s), and posts to `/api/run` from a textarea.

## 3. MCP server tools (`agent/mcp_server.py`)

Exposed over stdio via FastMCP as server name `swarn`:

| Tool | Signature | Behavior |
|---|---|---|
| `swarn_submit_task` | `(task, data_dir="", steps=12, mode="auto", model="")` | mode auto→`solve` iff data_dir given else `agent`. Spawns a daemon thread; returns `task_id` immediately |
| `swarn_task_status` | `(task_id)` | `running|complete|failed`, elapsed, latest message, result when done |
| `swarn_get_messages` | `(task_id)` | Header + last 100 progress messages + result |
| `swarn_list_tasks` | `()` | One line per submitted task |

Progress messages: solve mode uses the `on_step` journal callback; agent mode captures the
loop's stdout via `redirect_stdout` and keeps the last 200 lines. Solve mode sets
`reflect=True`.

## 4. Programmatic Python API

```python
# ReAct agent
from agent.agent_loop import AgentLoop
from agent.self_correction import SelfCorrectionPolicy
from agent.observability import GuardrailPolicy
result = AgentLoop(correction_policy=SelfCorrectionPolicy(),
                   guardrail_policy=GuardrailPolicy()).run("task")
# → {"outcome": "complete|no_tool_use|max_corrections|max_iterations",
#    "summary": str|None, "session_id": str}

# Tree search
from agent.search import SearchConfig, run_search
res = run_search("Predict y. Metric: AUC.", data_dir="data/",
                 config=SearchConfig(steps=25, time_limit_secs=3600),
                 evaluation_note="", on_step=None, resume_run_id=None)
# → SearchResult(run_id, run_dir, journal, best: Node|None, steps_done, wall_time)
#   res.solution_path, res.report_path

# Multi-agent
from agent.orchestrator import Orchestrator
out = Orchestrator().run("task")
# → {"final_outcome": str, "state": BlackboardState, "report_markdown": str}

# Tool registry (for embedding/hosting)
from agent.tools import get_tool_definitions, run_tool, TOOL_REGISTRY

# LLM layer
from agent.llm import create_client
resp = create_client().call(system, messages, tools=[...], tool_choice=None)
```

### Generated deployment API (per packaged model)

`workspace/deployments/<artifact_id>/app.py` exposes:
- `GET /health` → `{"status": "ok", "artifact_id": ...}`
- `POST /predict` — Pydantic model with one `float` field per (sanitized) training feature
  column → `{"<target>": float}`; 400 with detail on inference failure.
- Auto OpenAPI docs at `/docs`.

## API sequence — dashboard live run

```mermaid
sequenceDiagram
    participant B as Browser
    participant F as FastAPI loop
    participant EX as Thread pool
    participant AL as AgentLoop
    participant SS as SessionStore

    B->>F: WS /ws/live (connect)
    B->>F: POST /api/run {task}
    F->>EX: run_in_executor(agent.run, task)
    EX->>AL: run(task)
    loop each step
        AL->>SS: session.add_step(...)
        SS-->>F: on_step → run_coroutine_threadsafe(queue.put)
        F-->>B: ws.send_text(step json)
    end
    AL-->>EX: {"outcome", "summary", "session_id"}
    EX-->>F: result
    F-->>B: 200 JSON (POST resolves)
```
