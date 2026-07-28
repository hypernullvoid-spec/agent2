# Swarn — Autonomous ML Engineering Agent

A from-scratch competitor to [HeyNeo](https://heyneo.com) (the autonomous ML
engineer), built on the Phase 1–16 foundation, the V2 upgrades, and a V3 pass
designed to *beat* HeyNeo on the things that decide benchmarks and real work:

**V2 (kept):**

1. **A solution tree search engine** (`agent/search/`) — the AIDE-style
   draft → debug → improve loop that tops the MLE-bench leaderboard, instead of
   a single-pass ReAct agent hand-driving individual tools.
2. **BYO-LLM** (`agent/llm/`) — Anthropic, OpenAI, Ollama, vLLM, Gemini, Groq,
   or any OpenAI-compatible endpoint, selected by one model-spec string.
3. **Docker-optional execution** (`agent/execution.py`) — Docker when the
   daemon is up, a real cross-platform subprocess backend (Windows-safe,
   per-call timeouts) when it isn't.

**V3 (new — where we pull ahead of HeyNeo):**

4. **Parallel tree search** (`swarn solve … --workers 4`) — HeyNeo expands one
   solution at a time; our scheduler pipelines propose→execute→review across
   a thread pool with reservation-aware policy (no duplicate debugging, no
   draft explosion). Four drafts now cost roughly one draft's wall time.
5. **Self-improvement across runs** (`agent/knowledge.py`) — every finished
   run is reflected on by a cheap LLM call that distills *generalizable*
   lessons into a hard-capped playbook (hermes-agent style bounded memory),
   and archived in a SQLite FTS5 index. Future runs get the playbook plus
   the most similar past runs injected as prior art. HeyNeo starts every task
   cold; Swarn gets smarter with use. Inspect with `swarn playbook`.
6. **Checkpoint / resume** — `swarn solve --resume <run_id>` continues a killed
   run from its crash-safe `journal.json`, adding `--steps` more nodes.
7. **Static gate** (`agent/search/static_check.py`) — syntax errors, missing
   metric prints, and guaranteed hangs are scored buggy *without* burning a
   sandbox execution (opencode's diagnostics-before-run pattern).
8. **Budgets** — `--token-budget` stops a run before costs run away
   (transparent, unlike HeyNeo's opaque credits); time budgets shrink per-node
   timeouts to fit.
9. **Doom-loop detection + context compaction** (`agent/doom_loop.py`) — the
   ReAct loop detects same-call-same-result repetition (result-hash aware, so
   polling never false-positives) and injects a corrective note; old tool
   results are compacted once the conversation crosses a size budget.
   Iteration cap is now configurable (`SWARN_MAX_ITERATIONS`, default 30).
10. **MCP server** (`swarn mcp-serve`) — expose the whole platform to Claude
    Code, Cursor, Windsurf, Zed, or any MCP client via
    `swarn_submit_task` / `swarn_task_status` / `swarn_get_messages` /
    `swarn_list_tasks`. Register: `claude mcp add swarn -- swarn mcp-serve`.
11. **Docker timeout fix** — a timed-out container exec is now actually
    killed (container recycled), not silently left eating CPU.
12. **Dashboard upgrades** — `swarn serve` now renders search runs (tree +
    report) and the live playbook alongside the session feed.

The original 16-phase documentation lives in `README-phases-1-16.md`; all of
those capabilities (repo-RAG, self-correction, memory, data/feature/training/
eval/deploy tools, multi-agent orchestration, MCP, multimodal RAG, fine-tuning,
guardrails, CLI, dashboard) still work unchanged.

### V3 quick reference

```bash
# parallel search with budgets, learning enabled (default)
swarn solve "Predict churn. Metric: AUC." -d ./data -s 24 --workers 4 --token-budget 500000

# resume a killed run with 10 more nodes
swarn solve "Predict churn. Metric: AUC." --resume 20260714-091534-ba9b4f -s 10

# what has the agent learned so far?
swarn playbook

# serve the platform to Claude Code / Cursor over MCP
swarn mcp-serve
```

Env vars added in V3: `SWARN_SEARCH_WORKERS`, `SWARN_KNOWLEDGE_DIR`,
`SWARN_MAX_ITERATIONS`, `SWARN_CONTEXT_CHAR_BUDGET`.

---

## Why tree search is the headline feature

HeyNeo's benchmark claim is #1 on MLE-bench (34.2%). What wins that benchmark is
not a smarter chat loop — it's *systematic experimentation*: generate several
complete solution scripts, run them, measure them, fix the ones that crash,
and iteratively improve whichever is winning, until the budget runs out. That
is exactly what `agent/search/` implements:

```
            ┌─ draft ── run ── review ──┐        journal (solution tree)
task ──────►│  draft ── run ── review   ├──────► pick action for next step:
            └─ draft ── run ── review ──┘          • < num_drafts?  → draft
                     ▲                             • buggy leaf?    → debug (p=0.5)
                     └──── debug / improve ◄─────  • else           → improve best
```

- **Journal** (`journal.py`) — every attempt is a node in a tree with its code,
  execution output, reviewed metric, and buggy/good verdict. Crash-safe: saved
  to `runs/<id>/journal.json` after every step.
- **Policy** (`agent.py`) — drafts until `num_drafts` roots exist, then debugs
  buggy leaves with probability `debug_prob` (up to `max_debug_depth`
  consecutive fixes per branch) or makes one atomic improvement to a top-k
  good node (epsilon-greedy).
- **Review** (`agent.py`) — a second, low-temperature LLM call is *forced*
  (via tool call) to return `{is_bug, summary, metric, lower_is_better}`.
  The script's own printed `Final Validation Metric:` line wins on
  disagreement — regex + reviewer, not vibes.
- **Data preview** (`data_preview.py`) — file tree, CSV dtypes and heads
  injected into every prompt so generated code targets real columns.
- **Report** (`report.py`) — `runs/<id>/report.md` with the ASCII solution
  tree, metric history table, and failure analysis.

### Run it

```bash
swarn solve "Predict survival. Metric: accuracy." --data ./titanic --steps 20
swarn solve "Forecast demand, minimize RMSE." -d ./data -s 30 -t 3600 -m openai:gpt-4o
```

Or let the ReAct agent decide — it has a `solve_ml_task` tool and its
system prompt tells it to prefer search for "build the best model" tasks:

```bash
swarn run "Train the best possible model on workspace/churn.csv predicting churn"
```

Programmatic:

```python
from agent.search import SearchConfig, run_search
result = run_search("Predict y. Metric: AUC.", data_dir="data/",
                    config=SearchConfig(steps=25, time_limit_secs=3600))
print(result.best.metric, result.solution_path)
```

---

## Deployed model endpoint (BYO-LLM removed)

Every LLM call — single agent, team pipeline, search engine, dashboard, MCP
server — routes to **one deployed, OpenAI-compatible endpoint**, configured
in `agent/llm/router.py` (see the "PRODUCTION ENDPOINT — CHANGE HERE"
banner). For testing it points at a Qwen 3.5 9B deployment on Modal.

Swap in the production deployed model without code changes:

```bash
SWARN_DEPLOYED_MODEL=<served-model-name>
SWARN_DEPLOYED_BASE_URL=<https://.../v1>
SWARN_DEPLOYED_API_KEY=<key, or "dummy" if unsecured>
```

`--model` flags and `SWARN_CODE_MODEL`/`SWARN_FEEDBACK_MODEL` are accepted
but ignored for routing (display/log only) — except `mock:*`, the scripted
offline client that powers the test suite. All calls get retries with
exponential backoff and token accounting.

## Execution backends

```
SWARN_SANDBOX=docker|subprocess   # force; default = auto-detect
SWARN_EXEC_TIMEOUT=300            # default per-call timeout (seconds)
SWARN_SANDBOX_IMAGE=python:3.11-slim
```

Docker gives isolation (memory/CPU caps, bind-mounted workspace). The
subprocess backend gives universality: `sys.executable` (works on Windows),
hard timeouts, head+tail output truncation, structured `ExecResult` with exit
code and timing — which the search engine uses to detect timeouts and score
runs.

## Dashboard

`swarn serve` — everything from Phase 16, plus:

- `GET /api/runs` — all search runs with node counts and best metrics
- `GET /api/runs/{id}` — full journal (the tree) + report markdown
- `GET /api/playbook` — the cross-run playbook (learned lessons)

## Tests

```bash
python tests/run_tests.py    # zero-dependency runner (57 tests)
pytest tests/                # same tests, if pytest is installed
```

Coverage: deployed-endpoint routing, Anthropic↔OpenAI message/tool conversion, retry
policy, subprocess execution + timeouts, journal tree ops and persistence,
search policy transitions (incl. parallel reservations), static gate,
knowledge store + reflection, doom-loop detection, context compaction, review
parsing — plus full offline end-to-end searches (sequential and parallel,
resume, token budgets) on synthetic data with mock LLMs and real execution.

## Setup

```bash
cd swarn
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                # enables the `swarn` command
cp .env.example .env                            # optional: SWARN_DEPLOYED_* overrides (no API key needed for the test Qwen endpoint)
swarn solve --help
```

## What changed vs. phases 1–16 (file map)

```
agent/
  llm/                    NEW  LLM layer (hard-routed to the deployed endpoint)
    base.py                    normalized blocks/response, retries, usage tracking
    anthropic_client.py        REMOVED (tombstone stub — BYO-LLM stripped)
    openai_client.py           OpenAI-compat client for the deployed endpoint
    router.py                  deployed endpoint config ("CHANGE HERE" for production)
    mock_client.py             scripted client for offline tests
  search/                 NEW  the MLE-bench engine
    config.py journal.py agent.py runner.py report.py data_preview.py
    static_check.py            V3: pre-execution AST gate
  knowledge.py            NEW  V3: playbook + FTS5 run archive + reflection
  doom_loop.py            NEW  V3: repetition guard for the ReAct loop
  mcp_server.py           NEW  V3: MCP server (swarn mcp-serve)
  execution.py            NEW  Docker + subprocess backends, ExecResult
  llm_client.py           now a shim over agent/llm (back-compat)
  sandbox.py              now a shim over agent/execution (back-compat)
  tools.py                + solve_ml_task tool
  prompts.py              + tells the agent when to prefer tree search
  cli.py                  + `swarn solve` (--workers/--resume/--token-budget),
                            mcp-serve, playbook
  dashboard.py            + /api/runs + /api/playbook endpoints
main.py                   provider-aware API-key check
tests/                    NEW  57 tests + zero-dependency runner
```
