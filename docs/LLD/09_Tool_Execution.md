# 09 — Tool Execution

Files: `agent/tools.py` (registry + all built-ins), `agent/execution.py` + `agent/sandbox.py`
(code execution), `agent/mcp_integration.py` (dynamic remote tools).

## The registry

```python
TOOL_REGISTRY: dict[str, dict] = {}   # name -> {"description", "schema", "func"}

@tool(description=..., schema={...json schema...})
def my_tool(arg: str) -> str: ...
```

The `@tool` decorator (`tools.py:85–94`) registers the function under `func.__name__` at
import time. Two consumers:

- `get_tool_definitions(names=None)` → Anthropic-format list
  `[{"name", "description", "input_schema"}]`. Allow-list semantics:
  - `names is None` → all tools (single-agent default).
  - `"connect_mcp_server" in names` → the named tools **plus** every tool whose name starts
    with `mcp_` (so a role trusted to open MCP connections can also use the tools those
    connections register mid-run — the documented Phase 11×12 interaction, `tools.py:115–133`).
  - otherwise → exactly the named tools; unknown names silently skipped (a role config typo
    shortens the list rather than crashing).
- `run_tool(name, tool_input)` → dispatch with `**tool_input`; returns
  `"Error: unknown tool '<name>'"` or `"Error running '<name>': <Type>: <msg>"` on any
  exception. **Never raises** — the foundation of the self-correction loop.

## Tool invocation flow

```mermaid
sequenceDiagram
    participant LLM as Deployed LLM
    participant AL as AgentLoop
    participant TR as tools.run_tool
    participant SUB as Subsystem singleton
    participant EX as Execution backend

    LLM-->>AL: tool_use {name, input}
    AL->>AL: session.add_step(TOOL_CALL) — before execution
    AL->>TR: run_tool(name, input)
    TR->>SUB: lazy import + get_xxx().method(**input)
    alt code-execution tool
        SUB->>EX: get_sandbox().exec_python/shell → get_backend()
        EX-->>SUB: ExecResult.as_text()
    end
    SUB-->>TR: result str (or raises)
    TR-->>AL: result str ("Error…" if raised)
    AL->>AL: correction → guardrail → doom-loop enrichment
    AL->>AL: session.add_step(TOOL_RESULT, raw[:3000])
    AL-->>LLM: tool_result {tool_use_id, content: enriched str}
```

## Complete built-in tool catalog (registration order in `tools.py`)

| Phase | Tools | Backed by |
|---|---|---|
| 1 | `list_files`, `read_file`, `write_file`, `finish_task` | direct fs + `_safe_path` |
| 2 | `run_python`, `run_shell`, `install_package` | `sandbox.get_sandbox()` → process-wide backend |
| 3 | `index_project`, `search_codebase` | `context_engine` |
| 5 | `list_sessions`, `recall_session` | `memory` |
| 6 | `load_csv`, `load_excel`, `load_parquet`, `load_sql`, `load_cloud_data`, `validate_dataset`, `preview_dataset`, `list_datasets`, `save_dataset` | `data_pipeline` |
| 7 | `profile_features`, `engineer_features` | `feature_engineering` |
| 8 | `train_models`, `tune_hyperparameters`, `list_trained_models` | `model_training` |
| 9 | `evaluate_model`, `plot_confusion_matrix`, `plot_roc_curve`, `plot_residuals`, `compare_models` | `evaluation` |
| 10 | `package_model` | `deployment` |
| 12 | `connect_mcp_server`, `list_mcp_servers`, `list_mcp_tools`, `disconnect_mcp_server` | `mcp_integration` |
| 13 | `index_pdf`, `index_image`, `index_audio` | `multimodal_rag` |
| 14 | `prepare_finetune_dataset`, `fine_tune`, `merge_and_export_model`, `list_finetune_runs` | `finetuning` |
| 15 | `run_guardrail_benchmark`, `get_guardrail_findings` | `observability` |
| V2 | `solve_ml_task` | `agent.search.run_search` |
| dynamic | `mcp_<server>_<tool>` … | `MCPManager.call_mcp_tool` closures |

`finish_task(summary)` is special only by convention: it returns `"TASK_COMPLETE: …"` like
any tool, but `AgentLoop` checks `block.name == "finish_task"` to end the run and record the
summary.

## The workspace path guard

`_safe_path(path)` (`tools.py:171–176`, duplicated in `data_pipeline.py`):
`abspath(join(WORKSPACE_DIR, path))` must equal `WORKSPACE_DIR` or start with
`WORKSPACE_DIR + os.sep`, else `ValueError` ("escapes the workspace directory"). Applied to
all Phase-1 file tools and Phase-6 file loaders. **Not** applied to: `index_project` /
`index_pdf` / `index_image` / `index_audio` (absolute paths outside the workspace are an
explicit feature there), `load_sql`/`load_cloud_data` (remote), or shell/python execution
(the sandbox is the boundary there, not the path guard).

## Code execution backends

Selection (`execution.make_backend`): `SWARN_SANDBOX=subprocess|docker` forces; otherwise
Docker if `docker.from_env().ping()` succeeds, else subprocess. Two instantiations:

- **Process-wide** (`get_backend()`), used by `run_python`/`run_shell`/`install_package`
  through the `Sandbox` facade; workspace = repo `workspace/`.
- **Per-search-run** (`make_backend(runs/<id>/workspace)`), closed in the runner's
  `finally`.

| Property | SubprocessBackend | DockerBackend |
|---|---|---|
| Isolation | none (documented trade-off) | container; mem 2g, 2 CPUs; workspace bind-mount rw |
| Python | `sys.executable` (Windows-safe) | `python3` inside `python:3.11-slim` (override `SWARN_SANDBOX_IMAGE`) |
| Timeout | `subprocess.run(timeout=…)`; partial output preserved from `TimeoutExpired` | watcher thread `join(timeout)`; on expiry the **container is killed and recreated** (a single exec can't be killed) — the V3 fix for runaway processes |
| Result | `ExecResult(output ≤50k head+tail, exit_code, timed_out, exec_time)` | same |
| Shell | `bash -c` (POSIX) / `cmd /c` (Windows) | `bash -c` in container |
| Package install | pip into the *host env* | pip into the container (persists for container lifetime) |

`ExecResult.as_text()` produces the legacy display strings the correction policy pattern-
matches: `"Error: command timed out after Ns"`, `"[exit N]"` prefix, `"(no output)"`.

## Dynamic MCP tools

`connect_mcp_server(server_name, command, args)`:

1. `MCPManager` ensures the background asyncio loop thread is running.
2. Creates a request `asyncio.Queue` + readiness future on that loop.
3. Schedules `_server_task_main(...)` — the **single owner task** that:
   opens `stdio_client(StdioServerParameters(command, args, env))` + `ClientSession` in an
   `AsyncExitStack`, calls `initialize()` and `list_tools()`, registers each remote tool in
   `TOOL_REGISTRY` as `mcp_<safe_server>_<safe_tool>` with description prefix
   `[MCP: <server>]` and the server's `inputSchema`, resolves the readiness future, then
   loops servicing `(tool, args, response_future)` requests until a `None` sentinel, and
   finally closes the exit stack *in the same task* (anyio cancel-scope requirement,
   documented at length in the module).
4. `connect_server` blocks (≤30s) on readiness and records
   `{request_queue, task_future, tools, command}` in `_servers`.

Calls: the registered closure → `call_mcp_tool` → `_run_coro(_submit_call(...))` (60s
timeout) → owner task → `session.call_tool` → `_extract_text` flattens text blocks
(images/resources become placeholders). Errors and timeouts return strings, matching the
registry contract.

`disconnect_server`: sentinel → wait ≤15s for owner task → pop each `mcp_*` name from
`TOOL_REGISTRY` → remove server entry.

Because `AgentLoop` re-fetches tool definitions each iteration, tools registered by a
`connect_mcp_server` call become callable on the **very next** LLM turn.

## `solve_ml_task` — the search engine as a tool

`tools.py:1224–1243`: builds `SearchConfig(steps, time_limit_secs)` (all other config
defaults apply, including `use_knowledge=True`, `reflect=False`, `parallel_workers` from
env), wraps `run_search` in try/except → error string, and formats the result (best metric,
solution path, report path). This is the bridge that lets the ReAct agent delegate whole ML
tasks to the tree search.
