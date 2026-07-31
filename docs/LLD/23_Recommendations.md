# 23 — Recommendations

Practical, behavior-preserving improvements, ordered by value/effort. Each references the
debt item ([22_Technical_Debt.md](22_Technical_Debt.md)) or design doc it addresses.

## Quick wins (small, safe)

1. **Close sessions on LLM failure** (Debt A2): wrap the body of `AgentLoop.run` in
   `try/except LLMError` (or `finally`) that sets `outcome="llm_error"`, logs an ERROR
   step, and calls `close_session`. Preserves the trace of everything that happened before
   the failure.
2. **Register `atexit(close_sandbox)` in `cli.py`** (Debt A4) — or call `close_backend()`
   in a `finally` inside the `run`/`team` commands — so one-shot runs stop the Docker
   container they started.
3. **Unify guardrail instances** (Debt A3): have `main.py`/`cli.py`/`dashboard.py` use
   `get_guardrail_policy()` instead of constructing `GuardrailPolicy()` directly (one-line
   change per site); the `get_guardrail_findings` tool then reports the real findings.
4. **Fix `_verdict_is_approval` word-boundary matching** (Debt A7): use a regex like
   `\b(NEEDS_CHANGES|FAIL)\b` and check it against the first line only — the prompts
   already ask roles to *lead* with the verdict.
5. **Delete or implement the vestigial items** (Debt B): remove `_last_leaderboard`,
   `summary_of`, unused accessors; either implement `apply_saved_transform` (the fitted
   transformer state already exists) or drop the state + docstring promise. Implementing it
   is the higher-value choice — applying a fitted transform to a held-out set is a real
   workflow gap today.
6. **Fix README drift** (Debt G): test count, `main.py` description; clean the
   working-tree comment typo in `search/runner.py`.
7. **Enforce the documented 4-char minimum** in `SessionStore.get_session` (Debt A10).

## Correctness improvements (moderate)

8. **Split validation from selection in HPO** (Debt A1): inside
   `tune_hyperparameters`, carve a validation fold out of the training split for the
   Optuna objective (or use `cross_val_score` on train), and report final metrics on the
   untouched test split. Keeps the artifact contract identical.
9. **Collision-proof `_py_identifier`** (Debt A5): deduplicate sanitized names
   (`name`, `name_2`, …) and record the original→sanitized mapping in `metadata.json`;
   generated `FEATURE_ORDER` then stays aligned by construction.
10. **`plot_roc_curve` label handling** (Debt A6): pass
    `pos_label=model.classes_[1]` (sklearn) or map labels first.
11. **Per-task stdout capture in `mcp_server`** (Debt A8): replace `redirect_stdout` with
    a `SessionStore` subscriber (the hook already exists in `memory.py`) to build the
    transcript from structured steps instead of scraping prints — also removes the
    cross-task interleaving.
12. **Thread-safe usage accounting** (Debt A9): guard `Usage.add` with a small lock or use
    per-worker counters summed by `_Budget`.

## Structural refactors (larger, optional)

13. **Extract a `paths.py`** exporting `WORKSPACE_DIR`, `RUNS_DIR`, `SESSIONS_DIR`,
    `KNOWLEDGE_DIR`, `_safe_path`, and a shared `safe_filename()` (Debt C) — seven modules
    currently re-derive these.
14. **Split `tools.py`** into per-phase modules (`tools/files.py`, `tools/data.py`, …)
    that all import the same `registry.py` (`@tool`, `TOOL_REGISTRY`, `run_tool`,
    `get_tool_definitions`). Import them from `tools/__init__.py` to preserve the public
    surface. Zero behavior change; large navigability gain.
15. **Introduce `logging`** behind `ui.py` (Debt F): keep Rich rendering for TTYs, add a
    standard logger for files/CI; wire `SWARN_ENABLE_TRACING` into `cli.py` entry points
    too (it currently works only in the REPL).
16. **Make `POST /api/run` asynchronous** (Debt F): return a run token immediately and
    let clients follow `/ws/live`; requires generating the session id before
    `AgentLoop.run` (e.g. let the caller pass one to `new_session`) — the docstring already
    explains this is the current blocker.
17. **Move the dashboard HTML to a static file** with proper escaping of interpolated step
    data (Debt E, Security §6) — `Jinja2`/`html.escape` on `task` and step payloads closes
    the self-XSS hole if `--host` is ever widened.
18. **Redact secrets in traces** (Security §4): mask `connection_string`-like tool inputs
    (`load_sql`) before `session.add_step(TOOL_CALL, input=…)` and in `ui.tool_call`.

## Testing improvements

19. Add tests for the currently-untested seams: `AgentLoop` policy ordering (a mock-LLM
    ReAct run asserting correction→guardrail→doom order and session persistence on
    `LLMError`), `Orchestrator` verdict routing (pure string logic, cheap to test),
    `_py_identifier` collisions, `compact_messages` boundary behavior (partially covered in
    `test_doom_loop.py`), and MCP dynamic-registration visibility
    (`get_tool_definitions` with `connect_mcp_server` in the allow-list).
20. Auto-discover test modules in `run_tests.py` (glob `test_*.py`) so new files can't be
    silently skipped by forgetting the `MODULES` list.

## Documentation improvements

21. Keep this LLD set in CI review scope: the highest-drift areas are `README.md` feature
    claims and per-module docstrings that promise methods (Debt B/G). A lightweight check —
    grep docstring-mentioned public names against `dir(module)` — would have caught
    `apply_saved_transform`.
22. Document the deliberate trust boundaries in one place (currently spread across
    `roles.py`, `tools.py`, `observability.py` docstrings) — [16_Security.md](16_Security.md)
    can serve as the seed.

## Non-recommendations (things that look odd but are deliberate — do not "fix")

- Guardrail **flagging without blocking** — documented rationale (the model must see the
  attack) in `observability.py`.
- Journal full-file rewrite per node — that *is* the crash-safety mechanism.
- Blackboard-style summary passing between roles (rather than shared history) — documented
  design choice with real benefits (clean per-role sessions).
- Lazy imports everywhere — required for optional heavy deps and the tools↔MCP cycle.
- `mcp_integration`'s one-owner-task architecture — required by anyio cancel scopes; a
  "simpler" per-call coroutine version is exactly the bug it fixes.
