# 16 — Security

## Threat model implied by the implementation

The code defends against three things: (1) the agent writing/reading outside its
workspace, (2) agent-generated code consuming the host, and (3) prompt injection embedded
in data the agent reads. It does **not** implement user authentication, network policy, or
multi-tenant isolation — this is single-user local tooling.

## 1. Filesystem containment — `_safe_path`

`tools.py:171` and `data_pipeline.py:51` (duplicated):

```python
full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
if not (full == WORKSPACE_DIR or full.startswith(WORKSPACE_DIR + os.sep)):
    raise ValueError(...)
```

Applies to `list_files`/`read_file`/`write_file` and the Phase-6 file loaders. It blocks
`../` traversal and absolute paths (an absolute `path` makes `os.path.join` return it, and
the prefix check then fails).

**Boundary exceptions (by design):** `index_project`, `index_pdf/image/audio` accept
absolute paths anywhere; `run_python`/`run_shell` code can touch anything the executing
backend can (see next section); `save_dataset`/plots/deployments write only inside the
workspace.

## 2. Code-execution sandboxing

- **DockerBackend:** real isolation — container filesystem, `mem_limit=2g`, `cpu_count=2`,
  only the workspace bind-mounted (rw). Network is **not** restricted (default bridge).
- **SubprocessBackend:** *no isolation* — documented as a "deliberate trade-off so the
  agent works on any machine without Docker" (`execution.py:69–70`). Agent code runs as the
  invoking user with full host access, constrained only by wall-clock timeout and output
  truncation. The startup message warns: "no container isolation".
- Timeouts are the universal backstop; the Docker timeout path actually kills the runaway
  container (V3 fix).
- The tree search additionally pre-filters code with `static_check` (a correctness gate,
  not a security gate — it does not scan for dangerous operations).

## 3. Prompt-injection guardrails (`observability.py`)

- `GuardrailPolicy.scan_tool_result` scans **tool results only** (never user messages —
  the comment notes a real user may legitimately say "ignore previous instructions").
- Six case-insensitive regex patterns: classic override, system-prompt disregard,
  developer/admin/DAN mode, prompt extraction, covert-action ("do not tell the user"),
  fake authorization.
- Policy is **flag, never block**: a warning banner is prepended; content is not altered —
  rationale documented: the model must *see* the attack to resist it, and the system prompt
  teaches it how to treat the banner.
- Findings accumulate per policy instance; surfaced via the `guardrails` REPL command,
  `get_guardrail_findings` tool, and `ui.warn` lines.
- `BenchmarkHarness` tests the detector itself (3 true positives + 2 benign cases) and is
  honest about its limits: it validates detection logic, not live agent resistance.
- Explicitly heuristic: "a deliberately small, illustrative set … not a research-grade
  classifier."

## 4. Secrets & credentials

| Secret | Handling |
|---|---|
| `SWARN_DEPLOYED_API_KEY` | Env/.env; default `"dummy"` (test endpoint unsecured). Never logged |
| `.env` | Gitignored (`.env`, `.env.*` except `.env.example`) |
| DB connection strings (`load_sql`) | **Passed as a tool argument by the LLM** — they appear in tool inputs, hence in `trace.json` session logs and the dashboard feed. No redaction anywhere (verified: no masking code exists) |
| Cloud credentials | Read from environment by boto3/gcsfs; never handled by Swarn code |
| MCP server env | `connect_server(env=...)` parameter exists; the registered tool schema doesn't expose it to the LLM (tool schema has only server_name/command/args) |

## 5. Network exposure

- Dashboard: binds `127.0.0.1:8420` by default; `--host` can widen it. **No auth, no CORS
  config, no TLS.** `POST /api/run` executes arbitrary agent tasks — anyone who can reach
  the port can drive the agent (and thus, via `run_shell` on the subprocess backend, the
  host). Treat `--host 0.0.0.0` as remote code execution exposure.
- MCP server: stdio only — inherits the client's process boundary.
- `connect_mcp_server` lets the *LLM* launch arbitrary subprocesses (`command`, `args`) —
  a deliberate capability of the design; the trust boundary is the tool allow-list (only
  the Coder role gets it in team mode; the single agent always has it).

## 6. Injection surfaces reviewed

| Surface | Status |
|---|---|
| SQL | `load_sql` passes the LLM-authored query verbatim to `pd.read_sql` — by design (it's a query tool), no sanitization |
| Shell | `run_shell` is arbitrary `bash -c` — by design, sandbox-bounded |
| Path traversal | Blocked by `_safe_path` for file tools |
| Generated FastAPI app | Column names sanitized to identifiers (`_py_identifier`); `api_title`/`artifact_id` interpolated into the generated source with only basic quoting — a hostile artifact_id could break the generated file's syntax (low risk: ids originate from dataset names) |
| FTS5 query | Task words are wrapped in double quotes (`"word"`) and OR-joined; the token regex `[A-Za-z][A-Za-z0-9_]{2,}` excludes quote characters, preventing FTS syntax injection |
| Dashboard HTML | Session/step data injected into the DOM via `innerHTML`/template literals without escaping (only `report_markdown` is `<`-escaped) — a task string containing HTML executes in the viewer's browser (self-XSS on localhost; relevant if `--host` is widened) |

## 7. Human-in-the-loop as a control (`agent/core/approval_policy.py`)

Not a security boundary against a hostile model, but the control that keeps destructive
*data* operations from running unreviewed:

- `approval_policy.py` gates tool calls behind an approval callback. The REPL passes an
  `_Approver`; **headless mode passes none**, so every tool call runs unprompted — headless
  exists to be scriptable, and a prompt written to a stdin nobody is watching would hang.
- `apply_cleaning` and `ask_human` block on the human inside the tool itself, independent of
  the policy. `apply_cleaning` applies only the ops the human names and writes a new
  `<name>_clean` dataset rather than mutating the source.
- `SWARN_AUTO_APPROVE=1` and the REPL's `/yolo` bypass the gate — and with it the guarantee
  that nothing destructive ran unreviewed. Both are explicit opt-ins.

## 8. What does not exist (verified)

- No authentication or authorization anywhere.
- No encryption at rest (sessions, journals, knowledge DB are plaintext JSON/SQLite).
- No audit log beyond the session traces themselves.
- No rate limiting.
- No egress restrictions on sandboxed code (Docker default network).
- No approval gate in headless mode (documented above; deliberate, not an oversight).
