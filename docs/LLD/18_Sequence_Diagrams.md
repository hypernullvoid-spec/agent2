# 18 — Sequence Diagrams

## 1. Single-agent task (REPL / `swarn run`)

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant AL as AgentLoop.run
    participant SS as SessionStore
    participant LLM as LLMClient→OpenAICompatClient
    participant EP as Deployed endpoint
    participant RT as tools.run_tool
    participant P as Policies (corr/guard/doom)

    U->>AL: task
    AL->>SS: new_session(task, model)
    loop ≤ MAX_ITERATIONS (30)
        AL->>AL: compact_messages() if >400k chars
        AL->>LLM: call(SYSTEM_PROMPT, messages, get_tool_definitions())
        LLM->>EP: POST /chat/completions (converted messages+tools)
        EP-->>LLM: choice (content, tool_calls, usage)
        LLM-->>AL: LLMResponse (TextBlock/ToolUseBlock)
        AL->>SS: add_step(PLAN, text) per text block
        alt no tool_use blocks
            AL->>SS: close_session (outcome=no_tool_use)
        else tool calls
            loop each tool_use
                AL->>SS: add_step(TOOL_CALL) — pre-exec
                AL->>RT: run_tool(name, input)
                RT-->>AL: raw result str
                AL->>P: assess → scan → record (fixed order)
                P-->>AL: enriched result (+abort?)
                AL->>SS: add_step(TOOL_RESULT, raw[:3000])
                AL->>AL: collect tool_result block
            end
            AL->>AL: messages += assistant blocks + tool_results
            alt finish_task seen
                AL->>SS: add_step(COMPLETE) + close (outcome=complete)
            else abort (3 consecutive errors)
                AL->>SS: add_step(ERROR) + close (outcome=max_corrections)
            end
        end
    end
    AL-->>U: {outcome, summary, session_id}
```

## 2. `run_python` tool with Docker sandbox

```mermaid
sequenceDiagram
    autonumber
    participant RT as run_tool
    participant SB as Sandbox (facade)
    participant BE as DockerBackend
    participant W as watcher thread
    participant C as container

    RT->>SB: exec_python(code)
    SB->>BE: get_backend().exec_python(code, timeout=None→300)
    BE->>BE: _ensure_container() (create once, bind-mount workspace)
    BE->>BE: write workspace/_exec_<hex>.py
    BE->>W: start thread: container.exec_run(python3 /workspace/_exec_….py)
    W->>C: exec
    alt finishes in time
        C-->>W: exit_code, stdout/stderr
        W-->>BE: slot filled
        BE-->>SB: ExecResult(output≤50k, exit_code, exec_time)
    else timeout
        BE->>C: kill container (_recycle_container)
        BE-->>SB: ExecResult(timed_out=True, exit_code=-1)
    end
    BE->>BE: delete _exec file
    SB-->>RT: ExecResult.as_text()
```

## 3. Tree search — one parallel run (`swarn solve -w N`)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as cli.solve
    participant RS as run_search
    participant KS as KnowledgeStore
    participant SA as SearchAgent
    participant TP as ThreadPool worker
    participant BE as backend (per-run)
    participant J as Journal

    CLI->>RS: task, data_dir, SearchConfig
    RS->>RS: _prepare_run → runs/<id>/workspace/input (copy data)
    RS->>BE: make_backend(workspace)
    RS->>RS: data_preview.generate(input/)
    RS->>KS: context_for_task(task) → playbook + similar runs
    RS->>SA: SearchAgent(task, cfg, journal, preview, knowledge)
    loop until steps/budget done
        RS->>SA: (lock) choose_action(reserved, pending_drafts)
        RS->>TP: submit _work_one(stage, parent)
        TP->>SA: propose → code LLM call → Node
        TP->>TP: static_check(code) — maybe short-circuit buggy
        TP->>BE: exec_python(code, budget.node_timeout())
        TP->>SA: review(node) — forced submit_review + metric regex
        TP-->>RS: node
        RS->>J: (lock) append + save journal.json
        RS->>RS: on_step callback, log line
    end
    RS->>BE: close() (finally)
    RS->>J: write best_solution.py + final save
    RS->>RS: write_report(report.md)
    RS->>KS: index_run(...); reflect_on_run → add_lessons (if reflect)
    RS-->>CLI: SearchResult
```

## 4. Team pipeline (`swarn team` / REPL `team`)

```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator.run
    participant PL as AgentLoop(planner)
    participant CO as AgentLoop(coder)
    participant RE as AgentLoop(reviewer)
    participant TE as AgentLoop(tester)

    O->>PL: run(task) [read-only tools]
    PL-->>O: {complete, plan summary}
    O->>CO: run(task + plan) [full toolset]
    CO-->>O: {complete, work summary}
    O->>RE: run(task + coder summary) [read/eval tools]
    RE-->>O: {complete, "APPROVED…" | "NEEDS_CHANGES…"}
    alt NEEDS_CHANGES and revisions < 3
        O->>CO: run(task + prev summary + reviewer findings)
        Note over O,RE: re-review loop
    else APPROVED
        O->>TE: run(task + summaries) [exec/eval tools]
        TE-->>O: {"PASS…" | "FAIL…"}
        alt FAIL and revisions < 3
            O->>CO: run(task + tester findings)
        else PASS
            O-->>O: _finish("complete") → report markdown
        end
    end
```

## 5. `connect_mcp_server` and first remote tool call

```mermaid
sequenceDiagram
    autonumber
    participant LLM as Model
    participant AL as AgentLoop
    participant MM as MCPManager (sync side)
    participant EL as mcp-event-loop thread
    participant OT as owner task (_server_task_main)
    participant SRV as MCP server subprocess

    LLM-->>AL: tool_use connect_mcp_server{name, command, args}
    AL->>MM: connect_server(...)
    MM->>EL: ensure loop thread; create queue + ready future
    MM->>EL: schedule _server_task_main
    EL->>OT: run task
    OT->>SRV: spawn subprocess (stdio) + ClientSession.initialize()
    OT->>SRV: list_tools()
    OT->>OT: TOOL_REGISTRY["mcp_<srv>_<tool>"] = closure per tool
    OT-->>MM: ready future → (tool_names, command)
    MM-->>AL: "Connected… Registered N tool(s)" (string)
    Note over AL: next iteration: get_tool_definitions() now includes mcp_* tools
    LLM-->>AL: tool_use mcp_<srv>_<tool>{...}
    AL->>MM: closure → call_mcp_tool(server, tool, kwargs)
    MM->>EL: _submit_call → queue.put((tool, args, future))
    OT->>SRV: session.call_tool(tool, args)
    SRV-->>OT: content blocks
    OT-->>MM: future.set_result(_extract_text(...))
    MM-->>AL: result string (≤60s or timeout error string)
```

## 6. Dashboard live run

See [13_APIs.md](13_APIs.md) — browser connects `/ws/live`, POSTs `/api/run`; the run
executes in the executor thread of the same process; each `Session.add_step` crosses into
the event loop via `run_coroutine_threadsafe`; `broadcast_loop` fans out to sockets; the
POST resolves with the final outcome.

## 7. MCP-server task submission (external client driving Swarn)

```mermaid
sequenceDiagram
    autonumber
    participant CC as MCP client (e.g. Claude Code)
    participant FS as FastMCP (stdio)
    participant TH as worker thread
    participant CORE as run_search | AgentLoop

    CC->>FS: swarn_submit_task(task, data_dir, mode)
    FS->>FS: mode auto → solve iff data_dir else agent
    FS->>TH: Thread(_run_task, rec).start()
    FS-->>CC: "Task <id> submitted"
    TH->>CORE: run_search(on_step→messages) | AgentLoop.run (stdout captured)
    CC->>FS: swarn_task_status(id)  (poll)
    FS-->>CC: running + latest message
    CORE-->>TH: result
    TH->>TH: rec.status=complete, rec.result=…
    CC->>FS: swarn_get_messages(id)
    FS-->>CC: transcript + result
```

## 8. Startup flows

Covered in [03_Startup_Sequence.md](03_Startup_Sequence.md) (REPL diagram included there).
