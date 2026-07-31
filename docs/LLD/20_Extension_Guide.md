# 20 — Extension Guide

How to extend each subsystem, following the conventions the codebase already uses.

## Add a new tool (the primary extension point)

1. In `agent/tools.py` (or a new module imported by it):

```python
@tool(
    description="One paragraph the model reads to decide when to use this.",
    schema={
        "type": "object",
        "properties": {"arg": {"type": "string", "description": "…"}},
        "required": ["arg"],
    },
)
def my_tool(arg: str) -> str:
    from agent.my_subsystem import get_my_subsystem   # lazy import
    return get_my_subsystem().do(arg)                  # return str; never raise
```

2. Conventions to follow (all load-bearing):
   - **Return strings; never raise.** Errors as `"Error: …"` so the correction policy
     classifies them.
   - **Lazy-import** the subsystem inside the function body.
   - Resolve user paths with `_safe_path` if the tool touches workspace files.
   - Heavy subsystems get a module with a class + `get_xxx()` singleton accessor
     (copy the pattern at the bottom of any capability module).
3. Optionally document the tool + workflow in `prompts.SYSTEM_PROMPT`'s catalogue (the
   model behaves better with the workflow hints) and add it to the relevant role lists in
   `roles.py` if team mode should see it.
4. Nothing else: `AgentLoop` picks it up automatically via `get_tool_definitions()`.

## Add a new role to the team pipeline

1. `agent/roles.py`: define `<ROLE>_TOOLS` (allow-list; typos are silently dropped, so
   double-check names) and `<ROLE>_PROMPT = _SHARED_CORE + "━━━ Your role: … ━━━ …"`,
   then add the entry to `ROLES`.
2. `agent/orchestrator.py`: wire the role into `run()`'s pipeline — role sequencing is
   explicit code, not configuration. Follow the existing shape: build task string from
   blackboard summaries, `self._run_role("myrole", task, state)`, branch on
   outcome/summary.
3. If the role emits verdicts, keep the `APPROVED/NEEDS_CHANGES/PASS/FAIL` keyword
   convention so `_verdict_is_approval` works, or extend that method.

## Add a new LLM backend

1. `agent/llm/my_client.py`: subclass `BaseLLMClient`, implement
   `_call_api(system, messages, tools, max_tokens, temperature, tool_choice) -> LLMResponse`.
   Convert from/to the Anthropic-style message shape (`base.py` docstring documents it);
   `openai_client.py` is the reference implementation, `mock_client.py` the minimal one.
2. `agent/llm/router.py`: add a branch in `create_client()` (the current router
   intentionally hard-routes everything but `mock:*` to the deployed endpoint — restoring
   spec-based selection means re-introducing parsing here). Keep the client cache keyed on
   your spec.
3. Tool-choice forcing (`{"type":"tool","name":…}`) must be supported if the backend will
   serve the search reviewer / knowledge reflection.

## Change the deployed endpoint

No code: set `SWARN_DEPLOYED_MODEL`, `SWARN_DEPLOYED_BASE_URL`, `SWARN_DEPLOYED_API_KEY`
(env or `.env`). Or edit the three defaults under the "PRODUCTION ENDPOINT — CHANGE HERE"
banner in `agent/llm/router.py`.

## Add a new execution backend

1. `agent/execution.py`: new class with `name`, `exec_python(code, timeout) -> ExecResult`,
   `exec_shell(command, timeout) -> ExecResult`, `close()`.
2. Extend the `ExecutionBackend` union and `make_backend()` selection (plus a
   `SWARN_SANDBOX` value for forcing).
3. Keep `ExecResult.as_text()` semantics (`[exit N]`, "timed out after") — the correction
   policy pattern-matches those strings.

## Add a search-stage or policy tweak

- Tree policy knobs are all in `SearchConfig` — prefer adding a field with an env-backed
  default (pattern: `field(default_factory=lambda: os.environ.get(...))`).
- New stage (beyond draft/debug/improve): add a prompt builder + branch in
  `SearchAgent.propose`, extend `choose_action`, and ensure `Node.stage` values flow into
  `report.py`/`journal.render_tree` (they render any stage string as-is).
- New static-gate rules: extend `static_check()` — only reject *guaranteed* failures
  (documented invariant: "must never veto a script that could have worked").

## Add a memory/knowledge implementation

- **Session storage:** `SessionStore` is instantiable with a custom `sessions_dir`; for a
  different backend entirely, keep `new_session/close_session/list_sessions/get_session/
  recall_as_text/subscribe_to_all_sessions` — that's the surface `AgentLoop`, tools,
  REPL/CLI, and dashboard use. Swap the singleton in `get_session_store()`.
- **Knowledge:** `KnowledgeStore(root=...)` already supports relocation
  (`SWARN_KNOWLEDGE_DIR`). A different store must keep `context_for_task`, `index_run`,
  `add_lessons`, `playbook` (used by runner, CLI, dashboard).

## Add middleware around tool execution

There is no formal middleware chain; the insertion point is `AgentLoop.run()`'s per-tool
block (order matters — see the Phase 4/15 ordering comment) or, for *all* callers
including roles, a wrapper around `tools.run_tool`. Follow the policy-object pattern:
optional constructor arg, `None` = off.

## Add a dashboard endpoint

`agent/dashboard.py` — plain FastAPI. Read-only data should come from disk
(`sessions/`, `runs/`, `knowledge/`) or `get_session_store()`; anything triggering agent
work must run in an executor to keep the loop responsive (copy `/api/run`).

## Add an MCP-exposed capability

`agent/mcp_server.py` — add a `@mcp.tool()` function. For long work, follow the
`_TaskRecord` + daemon-thread pattern so the tool returns immediately and is pollable.

## Add configuration

- Env-var: read via `os.environ.get` at *use* time if it should be changeable per call, or
  at import for constants; document it in `.env.example`.
- Search-related: add a `SearchConfig` field + CLI flag mapping in `cli.solve`.

## Testing your extension

- Use `mock:*` clients: `create_client("mock:my-test")` returns a `MockLLMClient` you can
  script (`script=[...]`, `fallback=...`); `tool_response(name, input)` fabricates forced
  tool calls (see `tests/test_e2e_search.py` for the full offline-e2e pattern).
- Add a `test_*.py` with plain `test_*` functions and register the module in
  `tests/run_tests.py::MODULES` (the runner does not auto-discover files).
