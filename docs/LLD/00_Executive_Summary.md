# 00 — Executive Summary

## What this system is

**Swarn** (package name `swarn`, version 2.0.0 per `pyproject.toml`) is a from-scratch
autonomous AI engineering agent focused on machine-learning work. It is a single Python
package (`agent/`, with subpackages `agent.llm` and `agent.search`) that offers **two
complementary agent paradigms** plus a multi-agent coordination layer:

1. **A ReAct tool-calling loop** (`agent/agent_loop.py`) — the LLM is given a registry of
   ~45 tools (file I/O, sandboxed code execution, repo-RAG semantic search, a full tabular-ML
   pipeline from ingestion to deployment packaging, multimodal indexing, LoRA fine-tuning,
   MCP client integration) and iterates *think → call tool → observe result* until it calls
   `finish_task`, errors out, or hits an iteration cap.

2. **An AIDE-style solution tree search** (`agent/search/`) — for "build the best model on
   this data" tasks. The engine drafts several complete solution scripts, executes each in a
   sandbox, has a second LLM call review the result (`{is_bug, metric, ...}` via forced tool
   call), then iteratively **debugs** buggy leaves or **improves** the best node until a
   step/time/token budget is exhausted. Every attempt is a node in a persisted tree
   (`runs/<id>/journal.json`), enabling crash-safe **resume**.

3. **A fixed multi-agent pipeline** (`agent/orchestrator.py` + `agent/roles.py`) —
   Planner → Coder → Reviewer → Tester, where each role is an `AgentLoop` instance with a
   role-specific system prompt and a restricted tool allow-list, coordinated through a small
   "blackboard" of role summaries.

## How it is used

Four entry points (see [03_Startup_Sequence.md](03_Startup_Sequence.md)):

| Entry point | File | Purpose |
|---|---|---|
| `python main.py` | `main.py` | Interactive REPL (single agent + `team` command) |
| `swarn <cmd>` | `agent/cli.py` | One-shot CLI: `run`, `team`, `solve`, `sessions`, `recall`, `index`, `playbook`, `serve`, `mcp-serve`, `guardrail-benchmark` |
| `swarn serve` | `agent/dashboard.py` | FastAPI web dashboard: live websocket step feed, session/search-run browsing, playbook view |
| `swarn mcp-serve` | `agent/mcp_server.py` | MCP server exposing the platform (submit/status/messages/list) to Claude Code, Cursor, etc. |

## Key design decisions (verified in code)

- **One deployed LLM endpoint.** Every LLM call in the codebase is hard-routed to a single
  OpenAI-compatible endpoint configured in `agent/llm/router.py` (default: a test Qwen 3.5 9B
  deployment on Modal; overridable via `SWARN_DEPLOYED_*` env vars). All `--model` flags and
  model parameters are display-only. The single exception is the `mock:*` spec, which returns
  a scripted `MockLLMClient` used by the offline test suite.
- **The tool registry is the only thing that grows.** New capability = new `@tool`-decorated
  function in `agent/tools.py` (or dynamic MCP registration). `AgentLoop`'s control flow is
  never modified for new capabilities.
- **Errors are strings, never exceptions.** `run_tool()` catches everything and returns
  `"Error ..."` strings; the self-correction policy, guardrail scanner, and doom-loop
  detector then *enrich* those strings so the model can react.
- **Docker-optional execution.** `agent/execution.py` auto-detects Docker; falls back to a
  cross-platform subprocess backend with hard timeouts. Both return a structured `ExecResult`.
- **Per-process singletons.** Nearly every subsystem (session store, data pipeline, feature
  engine, model trainer, evaluator, packager, context engine, MCP manager, sandbox) is a
  lazily-created module-level singleton (`get_xxx()` accessor).
- **Self-improvement across runs.** `agent/knowledge.py` maintains a hard-capped playbook of
  distilled lessons (post-run LLM reflection) and an SQLite FTS5 archive of past runs; both
  are injected into future search prompts.
- **Safety layers on the ReAct loop.** Self-correction hints + abort budget
  (`self_correction.py`), prompt-injection scanning of tool results (`observability.py`),
  doom-loop repetition detection (`doom_loop.py`), deterministic context compaction
  (`agent_loop.py`), optional OpenTelemetry tracing.

## Scale of the codebase

~11,000 lines of Python. Largest modules: `tools.py` (1,243 — mostly declarative tool
schemas), `dashboard.py` (505, includes embedded HTML), `model_training.py` (479),
`mcp_integration.py` (440), `multimodal_rag.py` (428). Tests: 10 files, run by a
zero-dependency runner (`tests/run_tests.py`) or pytest.

## What is NOT in this system (verified)

- No database beyond SQLite (knowledge archive) and JSON files (sessions, journals, index).
- No authentication/authorization on the dashboard or MCP server (localhost tooling).
- No async agent loop — async exists only at the edges (dashboard, MCP client bridge).
- No fine-tuning of the main LLM — `finetuning.py` trains small *local* HuggingFace models.
- No streaming of LLM responses — all calls are blocking request/response
  (`BaseLLMClient.call()`; no streaming API is used anywhere).
- No cost-in-dollars tracking — token counts only (`Usage` in `agent/llm/base.py`).
