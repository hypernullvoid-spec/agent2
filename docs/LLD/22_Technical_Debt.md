# 22 — Technical Debt

**Observations** (verified in code) are separated from **assessment** (judgment).
Recommendations live in [23_Recommendations.md](23_Recommendations.md).

## A. Potential bugs / correctness risks

1. **HPO selects on the test split** — `model_training.tune_hyperparameters`'s Optuna
   objective fits on `X_train` and scores on `X_test`, then the "best" params are chosen by
   that same test score and the reported metrics come from it (`model_training.py:361–376`).
   *Assessment:* classic selection-leak; tuned artifacts' reported metrics are
   optimistically biased. (Ironically, `feature_engineering.py`'s docstring calls
   re-fitting on test data "a common and serious ML correctness bug this avoids by design".)
2. **Uncaught `LLMError` loses the session** — `AgentLoop.run()` has no try/except around
   `self.llm.call`; after retry exhaustion the exception unwinds without
   `close_session()`, so the trace (including all TOOL_CALL/RESULT steps) is never written
   (`agent_loop.py:180–194`, `llm/base.py:156`). The REPL then dies with a traceback.
3. **Guardrail findings split across instances** — the `get_guardrail_findings` tool reads
   the module singleton `get_guardrail_policy()` (`tools.py:1189–1191`), but
   `cli.py`/`dashboard.py` construct their **own** `GuardrailPolicy` for the
   loop. The tool can report "no findings" while the loop has flagged several. (The REPL's
   `guardrails` command reads the correct instance.)
4. **Docker container can outlive one-shot CLI runs** — only the REPL path registers
   `atexit(close_sandbox)`. A `swarn run` that auto-detected Docker starts a persistent
   `tail -f /dev/null` container that is never stopped when the process exits
   (`auto_remove` only fires after a stop/kill). Search runs are safe (`finally:
   backend.close()`).
5. **`_py_identifier` collisions in generated APIs** — `deployment._py_identifier` maps
   e.g. `"a b"` and `"a-b"` both to `a_b`; duplicate Pydantic fields would silently
   collapse, misaligning `FEATURE_ORDER` with the model's training columns.
6. **`plot_roc_curve` assumes {0,1}-style labels** — `predict_proba(...)[:, 1]` +
   `roc_curve(y_test, proba)` without `pos_label`; string-labeled binary targets raise at
   runtime (returned as an error string, but the feature is unusable for such data).
7. **Verdict parsing is substring-based** — a Tester summary like "PASSED; no FAILures
   found" contains `FAIL` → treated as rejection (`orchestrator._verdict_is_approval`).
   Mitigated only by prompt discipline.
8. **`mcp_server` agent-mode stdout capture is process-global** — `redirect_stdout` swaps
   `sys.stdout` for the whole process; two concurrent agent-mode tasks (or a concurrent
   solve task's prints) interleave transcripts (`mcp_server.py:92–96`).
9. **Unlocked shared counters in parallel search** — worker threads call the same cached
   LLM clients; `Usage.add` isn't synchronized, so token-budget accounting can undercount
   under races. Also, because clients are cached per-endpoint process-wide, *any other* LLM
   use in the process counts toward a search's token budget baseline drift.
10. **Session `get_session` prefix rule not enforced** — docstring says "min 4 chars", code
    matches any prefix (`memory.py:300–308`); a 1-char prefix returns the first match.

## B. Dead / vestigial code

| Item | Evidence |
|---|---|
| `FeatureEngine.apply_saved_transform` | Referenced in class docstring; method does not exist. The `_fitted_transformer/_fitted_feature_names/_fitted_target_col` state is written but never read |
| `ModelTrainer._last_leaderboard` | Assigned, never read |
| `BlackboardState.summary_of()` | Defined, no callers |
| `KnowledgeStore.get_run_code()` | Only called from tests |
| `Usage.cache_read_tokens` | Never populated by any client |
| `DoomLoopDetector.reset()` | Never called (a fresh detector is created per run instead) |
| `MCPManager.shutdown()` | No entry point calls it; MCP subprocesses/loop rely on daemon-thread teardown |
| `get_observability_hooks()` | Singleton accessor with no callers (hooks are constructed directly) |

## C. Duplication

- `WORKSPACE_DIR` is now defined once in `agent/paths.py` and imported — but **three**
  modules still recompute it independently (`data_analysis.py:38`, `data_report.py:41`,
  `ml/model_training.py:42`); a divergence in one silently splits the workspace. *(Was
  seven; the rest were consolidated.)*
- ~~`_safe_path` duplicated~~ — **resolved.** `tools.py` and `data_pipeline.py` both import
  `safe_path` from `agent/paths.py`.
- `load_cloud_storage`'s s3/gs branches are identical code.
- `_embed_and_upsert` intentionally duplicates `ContextEngine`'s batching loop (commented
  as a trade-off).
- Safe-filename sanitizers appear in four places (`evaluation._safe_name`,
  `deployment._deployments_dir`, `finetuning._run_dir`, `mcp_integration._make_local_name`).

## D. Coupling / encapsulation

- `evaluation.compare_models` → `trainer._trained_models`;
  `multimodal_rag` → `ContextEngine._collection/_embedder/_ensure_ready` (+ attaches
  `_clip_embedder` dynamically); `dashboard` → `store._index`. All private-attribute
  reaches, each self-acknowledged in comments.
- `roles.py` slices `SYSTEM_PROMPT` by `str.index` on `━━━` headers — renaming a header in
  `prompts.py` breaks import of `roles`/`orchestrator` (`ValueError` at import time).
- The correction policy's `_is_error` couples to exact string formats produced by
  `run_tool`/`ExecResult.as_text` — changing either format silently degrades detection.

## E. Large elements

- `tools.py` (1,243 lines) — mostly schema literals; still one file with 45+ concerns.
- `dashboard.py` embeds a ~175-line HTML/JS page as a Python string (no templating,
  no escaping helpers).
- `AgentLoop.run` (~185 lines) handles compaction, LLM calls, policy layering, logging,
  and exit conditions in one method.

## F. Observability / operability gaps

- No logging framework — all diagnostics are `print()` or Rich console writes;
  no levels, no timestamps outside sessions.
- ~~OTel tracing is only reachable from the REPL~~ — **resolved.** Both `_run_interactive`
  and `_run_headless` call `_make_observability_hooks()`, so `SWARN_ENABLE_TRACING` works
  for `swarn run` and `swarn team`. `serve`/`mcp-serve` still cannot enable it.
- The dashboard's `POST /api/run` blocks for the whole run (documented), so HTTP timeouts
  on long tasks surface as client errors while the run continues.

## G. Documentation drift (code vs docs)

- `README.md` says "57 tests"; the suite currently has **58** `test_` functions.
- `README.md`'s file map says `main.py` has a "provider-aware API-key check" — removed.
  `main.py` is now a 30-line shim to `agent.cli:main` and contains no logic at all.
- `feature_engineering` docstring promises `apply_saved_transform` (see B).
- Uncommitted working-tree change (`agent/search/runner.py`): a comment with typos
  ("runningn", trailing whitespace) replacing a blank line — cosmetic, should be cleaned
  before merge.

## H. Performance notes

- `_message_chars` walks the entire message list every iteration (O(n) per step; fine at
  current scales).
- `Journal.save` rewrites the whole JSON after every node (crash-safety trade-off;
  documented intent).
- PyTorch MLP trains full-batch for a fixed 100 epochs regardless of data size — cheap but
  arbitrary.
- `validate_dataset` computes z-scores column-by-column in Python loops — fine for
  typical sizes, quadratic-ish feel on very wide frames.

## I. Security-relevant debt

(Details in [16_Security.md](16_Security.md).) Highlights: unauthenticated dashboard that
can execute arbitrary tasks; subprocess backend = unsandboxed host execution; SQL
connection strings (with passwords) flow through LLM tool args into session traces;
dashboard HTML injects step data via `innerHTML` without escaping.
