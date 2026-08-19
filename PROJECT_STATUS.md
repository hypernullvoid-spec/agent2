# Swarn — Project Status Report

**Autonomous ML / document-intelligence engineering agent**
Version `2.0.0` · Python ≥3.10 · Status as of **2026-08-18**

> One-line summary: Swarn is a complete, working autonomous engineering agent —
> 16 foundation phases + a V2 tree-search rewrite + a V3 performance/learning pass
> + a document-intelligence suite — shipping as an installable CLI (`swarn`), a
> web dashboard, and an MCP server, with **283/283 tests passing offline**.

---

## 1. Completion at a glance

| Dimension | Value |
| --- | --- |
| Overall state | **Feature-complete and running.** All advertised commands execute; full suite green. |
| Python source | **21,625 lines** across 68 files (excl. `workspace/` scratch) |
| Core agent (`agent/`) | 11,651 lines · 28 modules |
| Capability packages (`swarn/`) | 4,860 lines · 5 modules |
| Tests | **5,010 lines · 283 tests · 283 passing, 0 failing** |
| Documentation | 2,010 lines of Markdown (`README.md` + `README-phases-1-16.md`) + design docx + leadership deck |
| Agent-callable tools | **51** registered in `TOOL_REGISTRY` |
| CLI commands | **15** (`swarn <cmd>`) |
| HTTP/WS endpoints | 8 (dashboard + JSON API + live websocket) |
| External runtime deps | 33 pinned packages (`requirements.txt`) |
| Offline capability | Full test suite, PDF extraction, doc Q&A and search all run with **no network and no API key** |

**Verified on 2026-08-19:** `python tests/run_tests.py` → `283 passed, 0 failed`.

---

## 2. What is built — by delivery stage

### Stage 1 · Foundation (Phases 1–5) — ✅ Done
| Phase | Capability | Module |
| --- | --- | --- |
| 1 | Core ReAct agent loop with tool dispatch | `agent/agent_loop.py` (347) |
| 2 | Code execution sandbox | `agent/execution.py` (280), `agent/sandbox.py` shim |
| 3 | Project context engine / repo-RAG | `agent/context_engine.py` (385) |
| 4 | Self-correction loop on tool errors | `agent/self_correction.py` (210) |
| 5 | Memory + structured session traces | `agent/memory.py` (402) |

### Stage 2 · ML pipeline (Phases 6–10) — ✅ Done
| Phase | Capability | Module |
| --- | --- | --- |
| 6 | Data ingestion & validation (CSV/Excel/Parquet/SQL/cloud, profiling, outliers, pandera schema) | `agent/data_pipeline.py` (284) |
| 7 | Automated feature engineering (numeric/categorical/high-card/datetime branches) | `agent/feature_engineering.py` (291) |
| 8 | Model training + Optuna hyperparameter tuning (sklearn, XGBoost, LightGBM) | `agent/model_training.py` (479) |
| 9 | Evaluation & visualization (confusion matrix, ROC, residuals, model comparison) | `agent/evaluation.py` (393) |
| 10 | Deployment automation (pickle/ONNX export, FastAPI service scaffolding) | `agent/deployment.py` (387) |

### Stage 3 · Advanced agentic capabilities (Phases 11–15) — ✅ Done
| Phase | Capability | Module |
| --- | --- | --- |
| 11 | Multi-agent orchestration + role definitions | `agent/orchestrator.py` (299), `agent/roles.py` (272) |
| 12 | Tool ecosystem & MCP client integration | `agent/mcp_integration.py` (440) |
| 13 | Multi-modal RAG — PDF text/tables, image OCR + captioning, audio transcription | `agent/multimodal_rag.py` (1,081) |
| 14 | LLM fine-tuning (PEFT/LoRA, dataset prep, merge & export) | `agent/finetuning.py` (406) |
| 15 | Guardrails, benchmarking, OpenTelemetry observability | `agent/observability.py` (382) |

### Stage 4 · Interface (Phase 16) — ✅ Done
| Phase | Capability | Module |
| --- | --- | --- |
| 16 | Typer CLI + FastAPI web dashboard with live websocket feed | `agent/cli.py` (695), `agent/dashboard.py` (505) |

### V2 · Search engine rewrite — ✅ Done
AIDE-style **solution tree search** replacing single-pass ReAct for "build the best model" tasks.

| Component | File | Lines |
| --- | --- | --- |
| Search policy (draft → debug → improve, epsilon-greedy) | `agent/search/agent.py` | 240 |
| Parallel scheduler + reservations | `agent/search/runner.py` | 299 |
| Crash-safe solution-tree journal | `agent/search/journal.py` | 165 |
| Data preview injection (file tree, dtypes, heads) | `agent/search/data_preview.py` | 91 |
| Run report (ASCII tree, metric history, failure analysis) | `agent/search/report.py` | 74 |
| Config | `agent/search/config.py` | 56 |
| LLM layer — single deployed OpenAI-compatible endpoint, retries, usage accounting | `agent/llm/` | 405 |

### V3 · Competitive pass — ✅ Done
| # | Feature | Where |
| --- | --- | --- |
| 1 | **Parallel tree search** (`--workers N`, reservation-aware, no duplicate debugging) | `agent/search/runner.py` |
| 2 | **Cross-run self-improvement** — reflection → capped playbook + SQLite FTS5 run archive, injected as prior art | `agent/knowledge.py` (232) |
| 3 | **Checkpoint / resume** (`--resume <run_id>`) from `journal.json` | `agent/search/` |
| 4 | **Static gate** — AST/syntax/metric-print/hang checks before burning a sandbox run | `agent/search/static_check.py` (55) |
| 5 | **Budgets** — `--token-budget`, time budgets shrink per-node timeouts | `agent/search/config.py` |
| 6 | **Doom-loop detection + context compaction** (result-hash aware) | `agent/doom_loop.py` (74) |
| 7 | **MCP server** — exposes the platform to Claude Code / Cursor / Windsurf / Zed | `agent/mcp_server.py` (175) |
| 8 | Docker timeout fix (timed-out exec actually killed, container recycled) | `agent/execution.py` |
| 9 | Dashboard upgrades — search runs, tree, report, live playbook | `agent/dashboard.py` |

### Document intelligence suite (`swarn/capabilities/`) — ✅ Done
The largest and most recently completed body of work: **4,043 lines of implementation, 2,826 lines of tests, 142 tests.**

| Capability | File | Lines | What it does |
| --- | --- | --- | --- |
| **Visual document intelligence** | `doc_intelligence.py` | 2,263 | PDF/image → typed fields + normalized bounding boxes + confidence-coloured annotated PNG. Four backends (`text` / `ocr` / `vlm` / `mock`) with an `auto` router that **never** silently substitutes mock data. Entity engine over positioned words: GSTIN, IFSC, PAN, email, phone, invoice/PO numbers, dates, amounts, `Label: value` pairs, stacked captions, column-segmented lines. Optional pydantic `target_schema` narrowing + validation. |
| **Grounded document Q&A** | `doc_qa.py` | 1,051 | Question → answer + verified evidence boxes + locally re-evaluated arithmetic. Three enforced defences: model never sees the document (only a column-aware line-id transcript), every quote token-exactly verified against line/multiline/table spans, arithmetic re-checked by a restricted AST walker (no `eval`). Unsupported answers reported as unsupported. Exit codes 0/2/1. |
| **Parse-once document store** | `doc_store.py` | 780 | `swarn ingest` writes a structured JSON copy (words, boxes, per-word confidence, line ids, page dims, tables) keyed `<stem>-<sha256[:12]>`; later questions load it in ~10 ms instead of re-parsing (~2.0 s measured on a 6-page deck). Optional cached page rasters make the store self-contained. |

| **PDF tables → CSV** | `doc_csv.py` | 299 | Writes a folder per PDF holding one CSV per table plus a `tables.json` manifest. Ruled tables come from the store; borderless ones from text alignment, which recovered 2,657 rows from a 54-page dataset that yielded **zero** files before. Three gates refuse prose that the strategy carved into a fake grid — row fill, column presence, and word integrity (a boundary cutting `Section 9.3` into `Sectio` \| `n9.3` scores 28% invented tokens against a data page's 0-4%). |
| **Stored-data document tree** | `doc_structure.py` | 259 | Converts a `StoredDocument` into the reading-order element list the document-tree builder consumes, so `swarn extract-pdf` and `swarn ask` work from **one parse**. Replaced a second, independent pdfplumber pipeline that had no column handling and produced a different (worse) answer from the same file. Every element carries a bounding box, which the old path had none of. |

Plus a **PDF → structured data** extractor (`swarn extract-pdf`) producing a full document tree — `fields` / `sections` / typed blocks (paragraph, list, key_values, table with `rows` **and** `records`) — inferred purely from layout, with **no LLM call and no network**.

---

## 3. Surface area shipped

### CLI — 15 commands

`swarn run` is the **universal entry point**. It takes an arbitrary task and,
optionally, the documents that task is about, then routes
([task_router.py](agent/task_router.py)):

| Invocation | Goes to | Why |
| --- | --- | --- |
| `swarn run "what is the total GST" invoice.pdf` | **fast path** → `doc_qa` | one question, one document, no other work — identical output and guarantees to `swarn ask`, one LLM call |
| `swarn run "read invoice.pdf and plot the line items"` | ReAct agent | needs `run_python` too; fast-pathing would answer and silently never plot |
| `swarn run "which has the higher total" a.pdf b.pdf` | ReAct agent | `doc_qa` answers about one document at a time |
| `swarn run "train the best model on churn.csv"` | ReAct agent | no document named |

Paths are picked up from trailing arguments *or* from the task text itself
(`swarn run "what is the total in invoice.pdf"`), and only when the file
actually exists on disk.

The fast path exists for correctness, not only speed: `doc_qa` verifies every
quote and re-evaluates the arithmetic, and an agent paraphrase of that JSON
would sit *after* all three defences — reintroducing exactly the unverified
claim they exist to catch. A bare document question therefore skips the
paraphrase entirely. On the agent path the document tool's own verified
evidence is printed before the agent's summary, and the system prompt forbids
restating a tool's figures loosely. `--ask` / `--agent` force either way.

Exit codes on the fast path match `swarn ask`: `0` answered, `2` not
answerable from this document, `1` error.
`run` (universal) · `team` · `solve` · `sessions` · `recall` · `index` · `extract-pdf` · `to-csv` · `doc-inspect` · `ingest` · `ask` · `mcp-serve` · `playbook` · `guardrail-benchmark` · `serve`
(installed as the bare `swarn` command via `pip install -e .`)

### Agent tools — 51 registered
| Group | Tools |
| --- | --- |
| Core / filesystem / exec (7) | `read_file`, `write_file`, `list_files`, `run_python`, `run_shell`, `install_package`, `finish_task` |
| Context & memory (4) | `index_project`, `search_codebase`, `list_sessions`, `recall_session` |
| Data (10) | `load_csv`, `load_excel`, `load_parquet`, `load_sql`, `load_cloud_data`, `validate_dataset`, `preview_dataset`, `list_datasets`, `save_dataset`, `profile_features` |
| Features & training (5) | `engineer_features`, `train_models`, `tune_hyperparameters`, `list_trained_models`, `solve_ml_task` |
| Evaluation & deploy (6) | `evaluate_model`, `plot_confusion_matrix`, `plot_roc_curve`, `plot_residuals`, `compare_models`, `package_model` |
| MCP (4) | `connect_mcp_server`, `list_mcp_servers`, `list_mcp_tools`, `disconnect_mcp_server` |
| Multimodal / documents (9) | `index_pdf`, `index_image`, `index_audio`, `extract_pdf_structured`, `extract_pdf_document`, `swarn_pdf_to_csv`, `swarn_doc_inspect`, `swarn_doc_ingest`, `swarn_doc_ask` |
| Fine-tuning (4) | `prepare_finetune_dataset`, `fine_tune`, `merge_and_export_model`, `list_finetune_runs` |
| Guardrails (2) | `run_guardrail_benchmark`, `get_guardrail_findings` |

### HTTP / WebSocket API — 8 endpoints
`GET /` (dashboard UI) · `GET /api/sessions` · `GET /api/sessions/{id}` · `GET /api/runs` · `GET /api/runs/{id}` · `GET /api/playbook` · `POST /api/run` · `WS /ws/live`

### MCP server — 4 exposed tools
`swarn_submit_task` · `swarn_task_status` · `swarn_get_messages` · `swarn_list_tasks`

---

## 4. Test coverage — 283/283 passing

| Suite | Tests | Covers |
| --- | --- | --- |
| `test_doc_intelligence.py` | 60 | Backends, backend routing (`auto` never picks mock), entity patterns, box normalization/coercion, confidence model, schema narrowing, annotation |
| `test_table_evidence.py` | 23 | Table-span quote resolution, token-exact matching, cross-row rejection, wrapped cells |
| `test_doc_store.py` | 22 | Parse-once caching, content-hash ids, re-parse on edit, page rasters, listing |
| `test_doc_qa.py` | 22 | Verification strategies, arithmetic AST re-evaluation, unsupported-answer handling, exit codes |
| `test_pdf_extract.py` | 20 | Section/heading inference, block typing, table `rows`+`records`, field mining, hyphenation repair |
| `test_column_layout.py` | 15 | Column banding, reading order, borderless-table joins, two-signal row/gutter rule, line-id stability |
| `test_search_agent.py` | 11 | Policy transitions, draft/debug/improve selection, review parsing |
| `test_journal.py` | 8 | Tree ops, persistence, crash-safety |
| `test_llm_layer.py` | 8 | Deployed-endpoint routing, message/tool conversion, retry policy |
| `test_knowledge.py` | 7 | Playbook capping, reflection, FTS5 archive |
| `test_static_check.py` | 7 | Syntax errors, missing metric prints, guaranteed hangs |
| `test_execution.py` | 6 | Subprocess backend, timeouts, truncation, `ExecResult` |
| `test_doom_loop.py` | 6 | Repetition detection, result-hash awareness, context compaction |
| `test_parallel_resume.py` | 4 | Parallel reservations, resume, token budgets |
| `test_task_router.py` | 25 | `swarn run` routing — path discovery, question vs. other-work signals, fast-path vs. agent, `--ask`/`--agent` overrides |
| `test_doc_structure.py` | 14 | Document tree derived from stored data — element/box round-trip, reading order, `\| ` cell segmentation, bilingual labels |
| `test_doc_csv.py` | 24 | PDF → CSV — borderless dataset recovery, table-boundary rules, per-PDF folder + manifest, prose refusal, word-integrity gate, fused-cell splitting, automatic export at ingest |
| `test_e2e_search.py` | 1 | Full offline end-to-end search on synthetic data (mock LLM, real execution) |

Runner: `python tests/run_tests.py` (zero external dependencies) or `pytest tests/`.

---

## 5. Engineering properties worth citing

- **Runs offline.** The entire 283-test suite, PDF structuring, document Q&A verification and search execution work with no network and no API key (`mock:*` scripted LLM client + real subprocess execution).
- **No silent fabrication.** `auto` backend selection can never fall back to synthetic sample data; mock results carry `synthetic: true`. Unverified quotes are boxed red and marked `NOT FOUND IN DOCUMENT`; an answer with nothing verified behind it is reported as *unsupported*, not as an answer.
- **No execution sink.** LLM-produced arithmetic is re-evaluated by a restricted AST walker accepting only numeric literals and operators — never `eval`.
- **Falsifiable output.** Every extracted field carries a normalized (0.0–1.0, DPI-independent) bounding box and a confidence, rendered as a colour-coded annotated PNG — the failure mode this suite exists to catch is a confident wrong number.
- **Crash-safe and resumable.** Search journals persist after every node; `--resume <run_id>` continues a killed run.
- **Portable execution.** Docker when the daemon is up (memory/CPU caps, bind-mounted workspace), a Windows-safe subprocess backend when it isn't; timed-out Docker execs are actually killed.
- **Single-endpoint LLM routing.** Every call — agent, team, search, dashboard, MCP — routes through `agent/llm/router.py`; production swap is three env vars, no code change.
- **Bounded cost.** `--token-budget` stops a run before spend runs away; time budgets shrink per-node timeouts.

---

## 6. Known limitations (stated, by design)

1. **Tables are found by visual structure** — ruled lines or consistent column gaps. Data laid out as unruled positioned text lands in the surrounding prose instead of being detected as a table.
2. **Structuring is heuristic** — a PDF with no font-size variation (or bold body text) under-structures into fewer sections rather than inventing wrong ones.
2b. **A text layer with no space glyphs cannot be un-fused.** Some producers emit a whole clause as one word object (`'है/Debitrepresentstheadditionalamountchargedtothe'` is a single word in a UPPCL bill's text layer). Word boundaries the file does not encode are not recoverable from it — that text needs OCR of a rasterized page.
2c. **Narrow two-column layouts vs. borderless tables.** A page gutter measured 4-6% of page width on that same bill, *narrower* than the borderless-table cells (2-5%) the distance rule exists to rescue; left-edge banding and baselines did not separate them either. The two are now told apart by content length — both sides being prose means columns, since a table cell is a value and a page column is a sentence. A borderless table whose cells are full sentences will split, which loses a row grouping but never fabricates an adjacency.
3. **Scanned/image-only PDFs have no text layer** — they need OCR (`index_image` on a rasterized page, or the `ocr` backend).
4. **BYO-LLM was removed in V3** — `--model`, `SWARN_CODE_MODEL` and `SWARN_FEEDBACK_MODEL` are accepted but display-only; routing is hard-pinned to the deployed endpoint (except `mock:*`).
5. **Evidence images still read source pixels** — an image cannot be reconstructed from cached word geometry; `swarn ingest --render-pages` caches rasters to close this.
6. **Git history is a single squashed commit** — per-phase history is documented in `README-phases-1-16.md` rather than in the log.

---

## 7. How to verify this report

```bash
cd swarn/agent2
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python tests/run_tests.py          # → 283 passed, 0 failed
swarn --help                       # → 15 commands
python examples/demo_doc_inspector.py   # zero-config, no key: generates and annotates an invoice
swarn serve                        # dashboard + JSON API
```

Regenerate the numbers in this report at any time:

```bash
python scripts/project_stats.py
```

---

## 8. Repository map

```
agent/                       11,501 lines — core platform
  agent_loop.py  prompts.py  tools.py (50 tools, 1,526)
  context_engine.py  memory.py  self_correction.py
  data_pipeline.py  feature_engineering.py  model_training.py
  evaluation.py  deployment.py  orchestrator.py  roles.py
  multimodal_rag.py (1,081)  finetuning.py  observability.py
  knowledge.py  doom_loop.py  execution.py
  cli.py  dashboard.py  mcp_server.py  mcp_integration.py
  task_router.py               swarn run: fast path vs. agent
  llm/      base · openai_client · router · mock_client
  search/   config · journal · agent · runner · report
            data_preview · static_check
swarn/                        4,860 lines — capability packages
  capabilities/ doc_intelligence (2,266) · doc_qa (1,051) · doc_store (726)
                doc_structure (259)  stored data -> document tree
                doc_csv (299)        PDF tables -> per-PDF CSV folder
tests/                        5,010 lines — 283 tests + zero-dep runner
examples/                     demo_doc_inspector.py (runnable, no key)
docs/                         design document (.docx) + leadership deck (.pptx)
artifacts/                    annotated PNGs + parsed document JSON store
sessions/                     structured run traces
README.md                     V2/V3 + document-intelligence reference
README-phases-1-16.md         detailed phase-by-phase engineering guide
swarn-architecture.png/.excalidraw   architecture diagram
```
