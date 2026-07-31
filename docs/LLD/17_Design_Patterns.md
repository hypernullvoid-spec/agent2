# 17 — Design Patterns

Only patterns actually present in the code, each with its concrete instance.

## Registry + Decorator (the load-bearing pattern)

- **Where:** `tools.py` — `@tool(description, schema)` populates `TOOL_REGISTRY` at import;
  `mcp_integration.py` adds/removes entries at runtime.
- **Why it matters:** it is the system's single extension axis ("the tool registry is the
  only thing that grows"). Dispatch (`run_tool`) and export (`get_tool_definitions`) are
  the only consumers.

## Singleton (module-level, lazy)

- **Where:** 13 accessors — `get_session_store`, `get_data_pipeline`,
  `get_feature_engine`, `get_model_trainer`, `get_model_evaluator`,
  `get_deployment_packager`, `get_fine_tuner`, `get_context_engine`,
  `get_multimodal_indexer`, `get_mcp_manager`, `get_sandbox`/`get_backend`,
  `get_guardrail_policy`/`get_benchmark_harness`/`get_observability_hooks`, plus the
  router's client cache.
- **Form:** `_x: Optional[X] = None; def get_x(): global _x; if _x is None: _x = X(); return _x`.
  Not thread-safe; per-process (explicitly leveraged/documented by `dashboard.py`).

## Template Method

- **Where:** `llm/base.py` — `BaseLLMClient.call()` fixes the retry/accounting algorithm;
  subclasses implement `_call_api()` (`OpenAICompatClient`, `MockLLMClient`).

## Adapter

- **Where:** `openai_client.py` — adapts the codebase's Anthropic-style
  messages/tools/tool_choice to OpenAI wire format and back ("this client translates in
  both directions so nothing else needs to know").
- Also `llm_client.LLMClient` and `sandbox.Sandbox` — thin **Facade/Adapter shims**
  preserving Phase-1/2 APIs over the newer layers.

## Strategy

- **Where:** `execution.py` — `SubprocessBackend` vs `DockerBackend` behind an identical
  three-method interface; selected at runtime by `make_backend()` (env force or
  autodetect). Type alias `ExecutionBackend = SubprocessBackend | DockerBackend`.
- Also the search stages (`draft`/`improve`/`debug` prompt builders dispatched by stage
  name in `SearchAgent.propose`).

## Observer (publish/subscribe)

- **Where:** `memory.py` — `SessionStore.subscribe_to_all_sessions(cb)` +
  `Session.on_step` list; `Session.add_step` notifies synchronously, exception-swallowing.
  Subscriber: `dashboard.ConnectionManager.on_step`. Also the search runner's `on_step`
  callback (single-observer variant used by the MCP server).

## Policy objects (Strategy applied to cross-cutting behavior)

- **Where:** `SelfCorrectionPolicy`, `GuardrailPolicy`, `ObservabilityHooks`,
  `DoomLoopDetector` — optional constructor-injected collaborators of `AgentLoop`
  ("passed the same optional way correction_policy is"). Null = feature off; this is
  **dependency injection by constructor**, done manually (no container/framework).

## Blackboard

- **Where:** `orchestrator.py` — `BlackboardState` (role history + revision counter)
  explicitly named and documented as such; roles communicate only via written summaries
  plus this small shared state.

## Command-ish tool dispatch

- Tool_use blocks are effectively serialized commands `{name, input}` dispatched through
  the registry. (No undo/queue machinery — noted for accuracy; it's registry dispatch, not
  full Command.)

## Facade

- `sandbox.Sandbox` (string API over structured backends), `llm_client.LLMClient`,
  `tools.py`'s thin wrappers over subsystem singletons (each tool body is a one-line
  facade call), `run_search()` over the whole search subsystem.

## Bridge (sync ⇄ async), hand-built

- `mcp_integration.py`: background event loop + per-server owner task + queue/future
  hand-off — documented as "the standard bridge pattern for 'sync codebase needs to call
  into an async-only library'".
- `dashboard.ConnectionManager`: the mirror image (sync producer, async consumer) via
  `run_coroutine_threadsafe` + `asyncio.Queue`.

## Memento / Snapshot persistence

- `Journal.save()/load()` — full-state JSON snapshot after every step enabling
  checkpoint/resume; `Session.to_dict()` similarly (write-once at close).

## Factory (simple)

- `make_backend(workspace)` and `llm/router.create_client(spec)` — parameterized creation
  functions with caching (router) / autodetection (backend). No abstract factory classes.

## Patterns notably *not* used

- No inheritance hierarchies beyond `BaseLLMClient` (capability classes are flat).
- No event bus/message queue; no plugin discovery via entry points (extension = edit the
  package, except MCP tools).
- No repository pattern over storage — persistence is inline JSON/SQLite.
- No async/await in core logic.
