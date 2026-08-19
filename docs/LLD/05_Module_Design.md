# 05 — Module Design

Per-module deep dive. Format: purpose → public API → internals → callers → dependencies →
error handling → extension points. (Class-level detail is in
[06_Class_Design.md](06_Class_Design.md); lifecycles in [04_Agent_Lifecycle.md](04_Agent_Lifecycle.md).)

---

## `agent/core/agent_loop.py`

- **Purpose:** the ReAct control loop; the only module whose control flow evolves across
  phases.
- **Public API:** `AgentLoop(...)`, `AgentLoop.run(task) -> dict`;
  module functions `compact_messages(messages) -> int`, `_message_chars(messages)`;
  constants `MAX_ITERATIONS`, `CONTEXT_CHAR_BUDGET`.
- **Callers:** `main.py` (REPL), `cli.py run`, `orchestrator.py` (per role),
  `dashboard.py /api/run`, `mcp_server.py` (mode="agent").
- **Dependencies:** `ui`, `agent.llm` (`DEFAULT_MODEL`), `llm_client.LLMClient`,
  `tools.get_tool_definitions/run_tool`, `prompts.SYSTEM_PROMPT`, `memory`,
  `self_correction`, `doom_loop`.
- **Data owned:** the in-flight `messages` list (Anthropic-style) and the current `Session`.
- **Error handling:** relies on `run_tool()` never raising; LLM failures (`LLMError` after
  retries) propagate **uncaught** out of `run()` — the session is then never closed (see
  [22_Technical_Debt.md](22_Technical_Debt.md)).
- **Extension points:** constructor injection of policies; `system_prompt`/`tool_names` for
  new roles.

## `agent/runtime/tools.py`

- **Purpose:** the tool registry and every built-in tool definition.
- **Public API:** `tool(description, schema)` decorator; `TOOL_REGISTRY: dict[str, dict]`;
  `get_tool_definitions(names=None) -> list[dict]` (Anthropic tools format);
  `run_tool(name, input) -> str`; `WORKSPACE_DIR`; `MCP_TOOL_PREFIX = "mcp_"`;
  plus each tool function (also importable directly, e.g. `index_project` from `main.py`).
- **Internals worth knowing:**
  - `get_tool_definitions` allow-list semantics: `None` → everything; a list *containing
    `connect_mcp_server`* → the list **plus every `mcp_*` tool currently registered** (the
    documented trust rule: a role allowed to open MCP connections may use what they produce);
    otherwise → exactly the listed names, unknown names silently skipped.
  - `run_tool` wraps dispatch in `try/except Exception` → `"Error running '<name>': …"`.
  - `_safe_path(path)` resolves against `WORKSPACE_DIR` and raises `ValueError` on escape
    (the exception is then stringified by `run_tool`).
  - Every tool body lazily imports its subsystem (`from agent.X import get_x`).
- **Callers:** `agent_loop.py` (definitions + dispatch), `roles.py` (allow-lists),
  `mcp_integration.py` (writes to `TOOL_REGISTRY`), `main.py`/`cli.py` (`index_project`).
- **Extension point:** add a `@tool`-decorated function (see [20_Extension_Guide.md](20_Extension_Guide.md)).

## `agent/messaging/prompts.py`

- **Purpose:** the single-agent `SYSTEM_PROMPT` — operating loop, self-correction contract,
  full tool catalogue with per-phase workflows, workspace rules, ambiguity policy.
- **Consumers:** `agent_loop.py` (default prompt), `roles.py` (slices out the
  "Core operating loop" and "Workspace" sections by `str.index()` on `━━━` headers).
- **Fragility:** renaming the `━━━ Core operating loop ━━━`, `━━━ Available tools ━━━`, or
  `━━━ Workspace ━━━` headers breaks `roles.py` **at import time**.

## `agent/llm/` (package) and `agent/llm/llm_client.py`

- **`base.py`** — normalized content blocks; `Usage` token accounting (`input/output/
  cache_read/calls`, `add()`, `summary()`); `LLMResponse` (`.text` join, `.tool_uses()`);
  `BaseLLMClient.call()` = retry shell (5 attempts, exponential backoff `2^n + rand` capped
  30s, retryable iff the error message contains one of `RETRYABLE_MARKERS`); subclass hook
  `_call_api(...)`; `complete()` convenience for text-only calls.
- **`router.py`** — `DEPLOYED_MODEL_NAME/BASE_URL/API_KEY` (env-first), `DEFAULT_MODEL`
  alias; `create_client(spec, cache=True)` with `_client_cache` and one-time
  ignored-spec notices (`_ignored_spec_notices`).
- **`openai_client.py`** — `OpenAICompatClient`: static converters
  `_convert_tools` (Anthropic `input_schema` → OpenAI `function.parameters`),
  `_convert_tool_choice` (`{"type":"tool","name":n}` → forced function; `"any"` →
  `"required"`), `_convert_messages` (system message first; assistant text+`tool_calls`;
  user `tool_result` blocks → `role:"tool"` messages). `_call_api` maps finish reasons
  (`tool_calls→tool_use`, `stop→end_turn`, `length→max_tokens`) and token usage.
  Malformed tool-call JSON arguments become `{"_raw": ...}` instead of raising.
- **`mock_client.py`** — scripted client; each `_call_api` records the call and pops the
  next scripted item (str | LLMResponse | callable), else uses `fallback`, else
  `"mock response"`.
- **`llm_client.py`** — Phase-1-compatible `LLMClient` facade over `create_client`;
  exposes `.call(system, messages, tools, max_tokens)` and `.total_usage`.
- **Callers:** everything that talks to an LLM: `agent_loop` (via shim), `search/agent.py`
  and `knowledge.py` (via `create_client` / a passed client).
- **Extension:** a new backend = subclass `BaseLLMClient`, implement `_call_api`, and give
  `router.create_client` a way to select it (currently requires editing `router.py`).

## `agent/runtime/execution.py` and `agent/runtime/sandbox.py`

- **Purpose:** run agent-generated code. `execution.py` is the real implementation;
  `sandbox.py` is the legacy string-API facade the ReAct tools use.
- **Public API:** `ExecResult` (`output/exit_code/timed_out/exec_time`, `.ok`,
  `.as_text()`); `SubprocessBackend`/`DockerBackend` (both:
  `exec_python(code, timeout)`, `exec_shell(cmd, timeout)`, `close()`, `.name`);
  `make_backend(workspace)` (respects `SWARN_SANDBOX` force, else Docker-ping autodetect);
  process-wide `get_backend()`/`close_backend()`; `sandbox.get_sandbox()`/`close_sandbox()`.
- **Key mechanics:**
  - Subprocess: script written into the workspace as `_exec_<hex>.py`, run with
    `sys.executable`, `cwd=workspace`, `PYTHONUNBUFFERED=1`, `subprocess.run(timeout=…)`;
    stderr appended under a `[stderr]` marker; output truncated head+tail at 50,000 chars.
  - Docker: one persistent container (`tail -f /dev/null`, `auto_remove=True`,
    `mem_limit="2g"`, `cpu_count=2`, workspace bind-mounted at `/workspace`); exec runs in a
    watcher thread with `t.join(timeout)`; on timeout the whole container is **killed and
    lazily recreated** (`_recycle_container`) because Docker cannot kill a single exec.
- **Consumers:** tools `run_python/run_shell/install_package` (process-wide backend via
  `sandbox.py`); `search/runner.py` (fresh `make_backend(run_workspace)` per run).

## `agent/memory/memory.py`

- **Purpose:** structured per-run traces ("sessions") + recall.
- **Public API:** `StepKind` enum (PLAN/TOOL_CALL/TOOL_RESULT/CORRECTION/COMPLETE/ERROR);
  `Step`; `Session` (`add_step`, `duration_s`, `tool_call_counts`, `to_dict`,
  `to_markdown`, `on_step` subscriber list); `SessionStore` (`new_session`,
  `close_session`, `list_sessions`, `get_session` (prefix match), `recall_as_text`,
  `subscribe_to_all_sessions`); `get_session_store()` singleton.
- **Persistence:** only at `close_session()` — writes `sessions/<uuid>/trace.json` +
  `summary.md` and upserts `sessions/index.json` (capped at 100 entries, newest first).
  Live visibility during a run exists **only** through the in-memory `on_step` callbacks
  (dashboard). Callbacks are exception-swallowed so a broken subscriber can't crash a run.
- **Callers:** `agent_loop.py`, tools `list_sessions`/`recall_session`, REPL/CLI history
  commands, dashboard REST endpoints (which also read `store._index` directly).

## `agent/memory/knowledge.py`

- **Purpose:** cross-run self-improvement.
- **Public API:** `KnowledgeStore(root=None)` — root resolution: arg →
  `SWARN_KNOWLEDGE_DIR` → `<repo>/knowledge`; `playbook()`, `add_lessons(list) -> int`,
  `index_run(...)`, `search_runs(query, k=3)`, `get_run_code(run_id)`,
  `context_for_task(task) -> str`; module fn `reflect_on_run(task, journal, feedback_llm,
  store, run_id) -> list[str]`.
- **Internals:** playbook is a markdown bullet list; `add_lessons` dedupes
  case-insensitively, caps each lesson at 300 chars, and drops the **oldest** lessons until
  the rendered file fits `PLAYBOOK_MAX_CHARS = 6000`. The archive is an FTS5 virtual table
  `runs(run_id, task, summary, code, metric UNINDEXED, ts UNINDEXED)`; `search_runs` builds
  an `OR` query from up to 24 word tokens of the task. `reflect_on_run` sends a run digest
  to the feedback LLM with a forced `submit_lessons` tool, stores ≤5 lessons.
- **Error posture:** every method catches `sqlite3.Error`/`OSError`/broad exceptions and
  returns empty values — "no DB, no playbook, no API key → empty strings, never an
  exception into the search loop".

## `agent/core/self_correction.py`, `agent/core/doom_loop.py`, `agent/observability/observability.py`

Covered in depth in [14_Error_Handling.md](14_Error_Handling.md) and
[16_Security.md](16_Security.md). Summary:

- `SelfCorrectionPolicy.assess(tool, result) -> (is_error, enriched)`;
  detection via traceback markers, `Error…` prefixes, `[exit N]` regex, timeout phrases, and
  a list of Python exception names; classification into `ErrorKind`; per-kind instruction
  `HINTS`; `should_abort()` when `consecutive_errors >= max_consecutive` (3).
- `DoomLoopDetector.record(...) -> bool`; md5 signature over tool+canonical-args+result
  head; window 30; triggers on 3 identical consecutive sigs or an A,B,A,B pair cycle.
- `GuardrailPolicy.scan_tool_result(...)` — 6 compiled case-insensitive regexes
  (`INJECTION_PATTERNS`); prepends a warning banner, never strips content; accumulates
  `InjectionFinding`s. `BenchmarkHarness.run()` — 5 canned cases (3 should-flag, 2
  benign) against a fresh policy. `ObservabilityHooks` — lazy OTel init (console exporter
  default, OTLP gRPC if endpoint given); `llm_call_span`/`tool_call_span` context managers;
  no-op when OTel missing.

## `agent/memory/context_engine.py` and `agent/memory/multimodal_rag.py`

- `ContextEngine`: lazy init of sentence-transformers `all-MiniLM-L6-v2` + ChromaDB
  `PersistentClient(.chroma/)` collection `"codebase"` (cosine). `index_directory` walks,
  filters (`INDEXABLE_EXT`, `SKIP_DIRS`, ≤300KB), chunks Python via AST (module header +
  one chunk per function/class) and other text via 60-line windows with 10-line overlap,
  embeds in batches of 64, **upserts** with md5 ids over `(file,start,end,name)`.
  `search(query, n_results)` returns formatted results with file/lines/type/similarity.
- `MultiModalIndexer`: adds PDF (pdfplumber text chunks of 40 lines/page + one
  pipe-delimited chunk per detected table), image (pytesseract OCR chunk + optional caption
  chunk + optional direct CLIP embedding via a lazily-attached `engine._clip_embedder`),
  and audio (whisper transcription grouped into ~60s windows keyed by timestamp). All chunks
  reuse `ContextEngine._make_chunk`'s shape with `start_line/end_line` repurposed as
  page/timestamp; all land in the **same collection**, so `search_codebase` returns a blend.
  `_embed_and_upsert` deliberately duplicates the engine's batching loop (comment says the
  5-line duplication is preferred over coupling).

## ML pipeline modules

- **`data_pipeline.py`** — `DataPipeline.datasets: dict[str, DataFrame]`; loaders return
  registration messages listing shape/columns; `validate_dataset` builds a text report
  (dtypes, nulls with %, duplicate count, per-column z>3 outliers, optional pandera
  inferred-schema check); `save_dataset` writes CSV/Parquet by extension.
- **`feature_engineering.py`** — `profile_dataset` infers per-column roles (constant/ID
  drop, datetime decompose, numeric, boolean, low-card OHE ≤20 uniques, high-card
  frequency-encode) and task type from the target; `engineer_features` executes those
  roles: datetime decomposition into 5 derived columns, manual frequency-encoding, then a
  `ColumnTransformer` (median-impute+standard-scale numerics; most-frequent-impute+OHE
  low-card cats), outputs a dense DataFrame `<name>_features` with the target appended
  unchanged; stores the fitted transformer on the singleton (`_fitted_transformer`).
  Note: a method `apply_saved_transform()` is referenced in the class docstring but **does
  not exist in the implementation**.
- **`model_training.py`** — task detection (numeric & >20 uniques → regression; 2 →
  binary; 3–20 → multiclass; fallback regression); candidates trained via
  `_fit_and_eval` (sklearn/xgb/lgbm) or `_fit_torch_mlp` (fixed 64→32 MLP, Adam, 100
  epochs, no batching); metrics rmse/mae/r2 or accuracy/f1; **only the best candidate** is
  stored as an artifact `{model, task_type, metrics, feature_columns, target_col,
  candidate, X_test, y_test}` under key `<name>__<candidate>` (or supplied `run_id`);
  `tune_hyperparameters` runs Optuna (25 trials default) over fixed spaces for
  rf/xgb/lgbm, refits best params, stores `<name>__<candidate>_tuned`.
  Caveat: the tuning objective evaluates on the *test* split (see
  [22_Technical_Debt.md](22_Technical_Debt.md)).
- **`evaluation.py`** — `ModelEvaluator` reads artifacts through `get_model_trainer()`;
  `evaluate_model` re-predicts on the stored split (torch models routed through a manual
  no-grad path selected by `type(model).__module__.startswith("torch")`);
  plots saved as PNG (`Agg` backend forced) under `workspace/plots/`; `compare_models`
  groups artifacts by task type.
- **`deployment.py`** — `package_model` serializes (joblib always; ONNX only for
  `sklearn.linear_model`/`sklearn.ensemble` modules via skl2onnx, silent fallback with
  explanation), renders a FastAPI `app.py` from the artifact's `feature_columns`
  (sanitized to Python identifiers), `requirements.txt`, `Dockerfile`, `metadata.json`
  under `workspace/deployments/<safe-id>/`.
- **`finetuning.py`** — three-step FineTuner: `prepare_dataset` (validates
  prompt/completion dicts, JSONL train/val split only when ≥10 examples),
  `fine_tune` (LoRA via PEFT on `q_proj`/`v_proj`, QLoRA=4-bit requires CUDA else explicit
  error, prompt+completion concatenated, max_length 512, batch 1 × grad-accum 4, no
  checkpoints), `merge_and_export` (merged standalone dir or adapter-only). Run state in
  `self._runs` (in-memory only).

## `agent/integrations/mcp_integration.py`

- **Purpose:** MCP *client* — consume external MCP servers as tools.
- **Design:** one background daemon thread runs a persistent asyncio loop
  (`_ensure_loop_running`). Per server, **one owner task** (`_server_task_main`) opens the
  stdio transport + `ClientSession` in an `AsyncExitStack`, lists tools, registers each in
  `TOOL_REGISTRY` as `mcp_<server>_<tool>` (closure → `call_mcp_tool`), then services a
  request queue of `(tool, args, response_future)` until a `None` sentinel; it closes the
  exit stack itself — required because anyio cancel scopes must be exited by the task that
  entered them (documented extensively in the module).
- **Public API:** `connect_server`, `call_mcp_tool`, `list_mcp_servers`, `list_mcp_tools`,
  `disconnect_server`, `shutdown` (never called by any entry point), `get_mcp_manager()`.
- **Timeouts:** `DEFAULT_CALL_TIMEOUT_S = 60` per tool call; 30s connect; 15s disconnect.

## `agent/core/orchestrator.py`, `agent/core/roles.py`, `agent/utils/ui.py`, `agent/web/dashboard.py`, `agent/integrations/mcp_server.py`, `agent/cli.py`, `main.py`

Covered in [04_Agent_Lifecycle.md](04_Agent_Lifecycle.md), [03_Startup_Sequence.md](03_Startup_Sequence.md), and [13_APIs.md](13_APIs.md).
`ui.py` specifics: module-level Rich `Console(highlight=False)`; role accent colors
(planner cyan, coder green, reviewer magenta, tester yellow); truncation constants
(tool args preview 120 chars, result preview 400 chars); degrades to plain text when stdout
is not a TTY; honors `NO_COLOR`.
