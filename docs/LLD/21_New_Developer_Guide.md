# 21 — New Developer Guide

## Get it running

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .                # enables the `swarn` command
cp .env.example .env            # optional; defaults point at the test Qwen endpoint

python tests/run_tests.py       # offline test suite (mock LLMs) — should pass first
swarn --help
python main.py                  # interactive REPL
```

No API key is needed for the default test endpoint. Docker is optional (subprocess
fallback with a startup warning). The heavy ML deps (torch, chromadb, …) are only needed
when the corresponding tools run — the core loop works without them installed, because all
subsystem imports are lazy.

## The mental model (memorize these five things)

1. **Everything is a tool.** The LLM's only lever is calling names in `TOOL_REGISTRY`
   (`agent/tools.py`). `AgentLoop` never special-cases capabilities (except recognizing
   `finish_task` as the stop signal).
2. **One LLM endpoint.** `agent/llm/router.py` hard-routes every call; model flags are
   decorative; `mock:*` is the offline switch.
3. **Errors are strings.** Tools never raise. The correction policy turns error strings
   into hints; the model retries; three consecutive failures abort.
4. **Two brains.** The ReAct loop hand-drives tools; the tree search
   (`agent/search/`) mass-produces whole solution scripts and keeps the best. The ReAct
   agent can invoke the search via the `solve_ml_task` tool.
5. **Singletons per process.** State (datasets, models, sessions store, MCP connections)
   lives in module singletons — nothing is shared between two `swarn` processes except
   what's on disk (`sessions/`, `runs/`, `knowledge/`, `workspace/`, `.chroma/`).

## How a request flows (single agent)

`you> build a model on data.csv` → `AgentLoop.run` → LLM sees `SYSTEM_PROMPT` + task +
tool schemas → emits tool_use blocks → `run_tool` executes → result is
correction/guardrail/doom-annotated → appended as `tool_result` → repeat ≤30 iterations →
`finish_task(summary)` ends it → session persisted to `sessions/<uuid>/`.

Full traces: [18_Sequence_Diagrams.md](18_Sequence_Diagrams.md).

## Recommended code-reading order

| Step | File | Why |
|---|---|---|
| 1 | `agent/tools.py` (top 180 lines) | The registry, `get_tool_definitions`, `run_tool`, `_safe_path` — the system's spine |
| 2 | `agent/agent_loop.py` | The whole ReAct loop in one class; every policy hook is visible in `run()` |
| 3 | `agent/llm/base.py` + `router.py` + `openai_client.py` | Message normalization, retries, the routing rule |
| 4 | `agent/self_correction.py`, `doom_loop.py`, `observability.py` | The three enrichment layers, in their application order |
| 5 | `agent/memory.py` | Sessions: what gets recorded, when it's persisted, the pub/sub hook |
| 6 | `agent/search/journal.py` → `agent.py` → `runner.py` | The tree search, in data → policy → orchestration order |
| 7 | `agent/execution.py` | Both backends and `ExecResult` |
| 8 | `agent/roles.py` + `orchestrator.py` | Team mode |
| 9 | `agent/knowledge.py` | The self-improvement loop |
| 10 | Skim `data_pipeline.py` → `feature_engineering.py` → `model_training.py` → `evaluation.py` → `deployment.py` | The ML chain; each consumes the previous one's registry/artifacts |
| 11 | `mcp_integration.py`, `dashboard.py`, `mcp_server.py` | The concurrency edges (read module docstrings first — they're excellent) |

Also read `README-phases-1-16.md` for the historical "why" of each phase.

## Where to add code

| I want to… | Go to |
|---|---|
| Give the agent a new capability | `tools.py` (+ maybe a new subsystem module) — see [20_Extension_Guide.md](20_Extension_Guide.md) |
| Change agent behavior/instructions | `prompts.py` (mind the `━━━` headers — `roles.py` slices on them) |
| Tune the ReAct loop (caps, compaction) | `agent_loop.py` constants / `SWARN_*` env |
| Tune search behavior | `search/config.py` (+ `cli.solve` flag mapping) |
| Change the LLM endpoint | env vars or `llm/router.py` banner block |
| Add a pipeline role | `roles.py` + `orchestrator.py` |
| Add dashboard views | `dashboard.py` (endpoints + the embedded HTML string) |
| Change what's remembered across runs | `knowledge.py` (playbook/archive), `memory.py` (sessions) |

## Debugging tips (implementation-derived)

- **Replay any run:** `swarn recall <id>` or open `sessions/<uuid>/summary.md` — every
  plan, tool call (pre-execution), result, correction, and outcome is there.
- **Search post-mortems:** `runs/<id>/report.md` (tree + failures) and `journal.json`
  (every node's code + full term_out). `swarn solve --resume <id> -s N` continues it.
- **Run offline:** set `SWARN_CODE_MODEL=mock:x` / `SWARN_FEEDBACK_MODEL=mock:x` — but
  note only *search* honors those; for the ReAct loop you must pass a `mock:*` model spec
  programmatically (`AgentLoop(model="mock:x")` works because the shim calls
  `create_client(model)`).
- **Watch live:** `swarn serve`, then trigger runs through the dashboard's own Run box —
  runs from other terminals will *not* stream (per-process store; documented in
  `dashboard.py`).
- **Doom loops / injections / corrections** show as single ⚠ lines in the terminal
  (`ui.warn`) — grep the session trace for `correction` steps.
- **Docker acting up:** `SWARN_SANDBOX=subprocess` forces the fallback;
  `SWARN_EXEC_TIMEOUT` raises per-call limits.
- **Tests:** `python tests/run_tests.py` — no pytest, no network, no key. New test modules
  must be added to its `MODULES` list.

## Common gotchas (all verified)

- A `swarn run` in one terminal can't see models/datasets trained in another — artifacts
  are in-memory per process.
- `roles.py` allow-list typos don't error; the role just silently loses the tool.
- `SWARN_MAX_ITERATIONS` etc. are read at import — set them before launching, not mid-run.
- The `report` REPL command only shows team runs from *this* REPL process.
- `get_guardrail_findings` (the tool) reads the module singleton policy, while the REPL/CLI
  loop uses its own instance — the tool may report "no findings" even when the loop flagged
  some; the REPL `guardrails` command reads the right one.
- Search `reflect` (playbook learning) is off by default in `SearchConfig`; the CLI turns
  it on unless `--no-learn`.
