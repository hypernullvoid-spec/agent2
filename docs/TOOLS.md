# Swarn — Tools Reference

Every capability the agent can call, and every way you can reach it yourself.

There are two surfaces:

| Surface | What it is | How you use it |
|---|---|---|
| **Agent tools** | 75 functions in `TOOL_REGISTRY` the LLM calls during a run | You describe a task; the agent picks tools. You can also call them directly from Python. |
| **CLI commands** | `swarn <command>` (typer app in [agent/cli.py](agent/cli.py)) | You type them in a shell, or at the REPL prompt. |

---

## 1. Getting started

```bash
pip install -e .          # installs the `swarn` console script

swarn                     # interactive REPL
swarn run "build me a churn model"       # headless one-shot
swarn team "clean and analyse sales.csv" # multi-agent pipeline
python main.py            # identical to `swarn` (thin shim)
python -m agent.cli       # source checkout, nothing installed
```

Files the agent reads/writes live under `workspace/`. Paths passed to tools are
relative to it and are guarded by `safe_path` in [agent/paths.py](agent/paths.py) —
you cannot escape the workspace.

### Calling a tool directly from Python

```python
from agent.runtime.tools import TOOL_REGISTRY, run_tool, get_tool_definitions

run_tool("load_csv", {"path": "sales.csv", "name": "sales"})
run_tool("describe_dataset", {"name": "sales"})

len(TOOL_REGISTRY)                                 # every registered tool
get_tool_definitions(["read_file", "write_file"])  # Anthropic tools-API format
```

`run_tool` never raises — failures come back as a string starting with `Error:`,
which is exactly what the agent's self-correction loop reads.

Tools in [agent/data_cleaner.py](agent/data_cleaner.py),
[agent/data_analysis.py](agent/data_analysis.py),
[agent/data_report.py](agent/data_report.py) and
[agent/workbook.py](agent/workbook.py) register themselves lazily; if you are
driving the registry by hand, call each module's `register_*` function first
(the CLI does this for you).

---

## 2. Agent tools

### 2.1 Files & task control — [agent/runtime/tools.py](agent/runtime/tools.py)

| Tool | Arguments | What it does |
|---|---|---|
| `list_files` | `path` | List files/subdirs inside the workspace. `.` for root. |
| `read_file` | `path` | Read a file's full text. |
| `write_file` | `path`, `content` | Write a file, creating parent dirs. Overwrites. |
| `finish_task` | `summary` | Ends the run. Called only when work is complete. |

### 2.2 Code execution

| Tool | Arguments | What it does |
|---|---|---|
| `run_python` | `code` | Runs Python in a Docker sandbox (workspace mounted at `/workspace`), falling back to subprocess if Docker is down. **Loaded datasets are already available here as pandas DataFrames under their registered names.** Timeout via `SWARN_EXEC_TIMEOUT` (default 300s). |
| `run_shell` | `command` | Any shell command in the sandbox (`bash -c`). Requires Docker. |
| `install_package` | `packages` | pip-install space-separated packages into the sandbox, e.g. `"numpy pandas"`. Persists for the session. |

### 2.3 Semantic search over code and documents

| Tool | Arguments | What it does |
|---|---|---|
| `index_project` | `directory` | Chunk + embed a directory into ChromaDB (Python chunked by AST function/class). Call **before** `search_codebase`. Re-calling updates the index. |
| `search_codebase` | `query`, `n_results` | Natural-language search over the index; returns chunks with file paths and line numbers. |
| `index_pdf` | `path` | Index a PDF's prose *and* tables into the same index; citations become (file, page). |
| `index_image` | `path`, `caption` | OCR + (if available) CLIP embedding, so visual content is searchable too. |
| `index_audio` | `path`, `model_size` | Whisper-transcribe and index; citations are timestamps (`12:34`). |

Typical flow: `index_project(".")` → `search_codebase("where is auth handled")` → `read_file(...)`.

### 2.4 Session memory

| Tool | Arguments | What it does |
|---|---|---|
| `list_sessions` | `n` | Recent runs with outcome, duration, tool-call and self-correction counts. |
| `recall_session` | `session_id` | Full ordered tool-call log for an 8-char session ID prefix. |

### 2.5 Loading data

| Tool | Arguments | What it does |
|---|---|---|
| `load_csv` | `path`, `name` | CSV into the dataset registry. |
| `load_excel` | `path`, `name`, `sheet_name` | Excel. Omit `sheet_name` on a multi-sheet file and it returns the sheet list instead of loading. |
| `load_parquet` | `path`, `name` | Parquet. |
| `load_sql` | `connection_string`, `query`, `name` | SQLAlchemy URL + query into a dataset. |
| `load_cloud_data` | `uri`, `name` | `s3://` or `gs://` CSV/Parquet. Needs credentials already in the environment. |
| `inspect_workbook` | `path` | **Use this first on any .xlsx.** Describes *every* sheet without loading data: shapes, columns, types, sample rows, formulas, which sheets feed which, shared columns. A few hundred tokens at any workbook size. |
| `load_workbook` | `path`, `prefix`, `sheets` | Loads **all** sheets at once as `<file>_<sheet>` datasets. Use instead of repeated `load_excel`. |
| `list_datasets` | — | Everything currently in the registry, with shapes. |
| `preview_dataset` | `name`, `n` | First rows. |
| `validate_dataset` | `name` | Dtypes, null counts, duplicates, z-score outliers, pandera schema check. Call right after loading. |
| `describe_dataset` | `name` | Analyst summary: per-column dtype, nulls, cardinality, quartiles / top values. |
| `save_dataset` | `name`, `path` | Write a registry dataset back out (CSV or Parquet, by extension). |

### 2.6 Cleaning — human-in-the-loop ([agent/data_cleaner.py](agent/data_cleaner.py))

| Tool | Arguments | What it does |
|---|---|---|
| `clean_dataset` | `name`, `target_col` | Diagnoses missing values, redundancy, structural errors, types/formats, anomalies. Returns a numbered **cleaning plan**. Changes nothing. |
| `apply_cleaning` | `name`, `operations`, `params`, `output_name`, `target_col` | **Blocks and asks the human to approve** — `all`, `none`, `op1 op3 op5`, or `op4: factor=83.2` — then applies only what was approved. Writes `<name>_clean`; the original is never modified. |
| `ask_human` | `question`, `options` | Generic approval prompt for any destructive step. Called alone, never batched with other calls. |

The contract: load → `clean_dataset` → `apply_cleaning`. Don't call `ask_human`
for cleaning approval; `apply_cleaning` does it itself.

### 2.7 Analysis & charts ([agent/data_analysis.py](agent/data_analysis.py))

Charts are saved as PNG under `workspace/plots/`; each tool also returns the
reading **in words**, because the agent cannot see the image.

| Tool | Arguments | What it does |
|---|---|---|
| `analyze_dataset` | `name` | Overall shape/quality pass. |
| `plot_column` | `name`, `column`, `bins`, `top`, `log_scale` | One column: histogram + spread (numbers), bar + shares (categories), timeline (dates). Chart type chosen automatically. `log_scale` matters for revenue/vote-count style columns. |
| `plot_relationship` | `name`, `x`, `y` | Two columns: scatter + trend + correlation, box plot per group, time trend, or counts grid — picked from the column types. |
| `analyze_correlations` | `name`, `target`, `method` | Ranked correlations + heatmap. Pass `target` to answer "what drives X". Association, never causation. |
| `check_subgroups` | `name`, `x`, `y`, `by`, `min_rows`, `top` | Recomputes the x~y relation inside each group and flags reversals. **Run before reporting any correlation** — a pooled correlation can be true overall and wrong for every group in it. |
| `compare_groups` | `name`, `value_col`, `group_col` | Is the gap real? Welch t-test (2 groups) / one-way ANOVA (3+), translated into plain language. |
| `rank_by` | `name`, `group_col`, `value_col`, `how`, `top`, `ascending` | The league table. Use for **every** "top N" claim — near-ties reorder when read off a bar chart. The ranking is recorded, and a report naming a different top N is refused. |
| `analyze_missing` | `name`, `min_frac`, `top` | Which columns go blank *in the same rows* — the signal that blanks mark a structurally different record set. Call before approving imputes. |
| `analyze_multivalue` | `name`, `column`, `value_col`, `how`, `sep`, `min_rows`, `top` | For list-valued cells ("Action, Adventure, Comedy"). Splits and analyses individual values. Use **instead of** `plot_column`/`compare_groups` on such a column. |
| `pivot_dataset` | `name`, `rows`, `columns`, `values`, `how`, `percent`, `output_name` | Cross-tab grid; registers `<name>_pivot`. |
| `group_dataset` | `name`, `keys`, `aggregations` | SQL-style GROUP BY; registers `<name>_grouped`. |
| `analyze_over_time` | `name`, `date_col`, `freq`, `value_col`, `how` | Period totals with % change, biggest rise and fall, line chart. `freq`: `D`, `W`, `ME` (default), `QE`, `YE`. |
| `measure_duration` | `name`, `start_col`, `end_col`, `unit`, `by` | Elapsed time between two date columns. Handles the two columns being written in different date formats, and warns when an end date precedes its start. |

### 2.8 Reporting ([agent/data_report.py](agent/data_report.py))

| Tool | Arguments | What it does |
|---|---|---|
| `write_report` | `dataset`, `situation`, `complication`, `takeaways`, `next_steps`, `title`, `dashboard_url`, `output_name` | The end-of-analysis report: Background → Key figures → Key takeaways → Methodology → Appendix. **You** write the narrative; the numbers, charts, cleaning record and limitations are generated from recorded evidence and cannot be edited. If the narrative contradicts what was measured, the report is **refused** with the reasons. |

### 2.9 Machine learning

**Feature pipeline**

| Tool | Arguments | What it does |
|---|---|---|
| `profile_features` | `name`, `target_col` | Column roles: numeric / low- vs high-cardinality categorical / datetime / drop-me ID columns. Infers task type from the target. Run before `engineer_features`. |
| `engineer_features` | `name`, `target_col`, `drop_cols`, `output_name` | Impute + scale numerics, one-hot low-cardinality, frequency-encode high-cardinality, decompose datetimes into year/month/day/dayofweek/is_weekend. Registers `<name>_features`. |

**Training**

| Tool | Arguments | What it does |
|---|---|---|
| `train_models` | `name`, `target_col`, `candidates`, `test_size`, `cv_folds` | Auto-detects regression / binary / multiclass; tries linear-logistic, random forest, XGBoost, LightGBM and a small PyTorch MLP. Returns a ranked leaderboard. Input must be all-numeric — run `engineer_features` first. |
| `tune_hyperparameters` | `name`, `target_col`, `candidate`, `n_trials` | Optuna search (25 trials default) over `random_forest`, `xgboost` or `lightgbm`. Registers `<name>__<candidate>_tuned`. |

**Inspecting models**

| Tool | Arguments | What it does |
|---|---|---|
| `list_trained_models` | — | Artifacts in memory, their metrics, and how many are persisted. |
| `evaluate_model` | `artifact_id` | Full diagnostics on the stored held-out split: per-class precision/recall, residual distribution stats. |
| `compare_models` | — | Bar-chart comparison of every artifact, grouped by task type. |
| `feature_importance` | `artifact_id` | Ranked features + bar chart (tree `feature_importances_`, or absolute coefficients). |
| `plot_confusion_matrix` | `artifact_id` | Classification confusion matrix PNG. |
| `plot_roc_curve` | `artifact_id` | ROC + AUC for binary models supporting `predict_proba`. |
| `plot_residuals` | `artifact_id` | Residuals-vs-predicted and residual distribution for regressors. |

**Using and shipping models**

| Tool | Arguments | What it does |
|---|---|---|
| `predict` | `artifact_id`, `rows` | In-session inference. `rows` are dicts keyed by the model's feature column names. |
| `save_model` | `artifact_id` | Re-persist to `workspace/artifacts/`. Training and tuning already save automatically. |
| `delete_model` | `artifact_id` | Remove from memory and from disk. |
| `package_model` | `artifact_id`, `export_format`, `api_title` | Pickle (or ONNX for sklearn-native models) plus a generated FastAPI service with a typed `/predict`, OpenAPI docs, `requirements.txt` and Dockerfile, written to `workspace/deployments/<artifact_id>/`. |

**Autonomous solving**

| Tool | Arguments | What it does |
|---|---|---|
| `solve_ml_task` | `task`, `data_dir`, `steps`, `time_limit_secs` | Solution-tree search: drafts several independent scripts, executes each in the sandbox, reviews the results, then debugs failures and improves the best until the budget is spent. The tool form of `swarn solve`. |

### 2.10 Fine-tuning a local open-weight model

Not for fine-tuning Claude — these train a small HuggingFace model locally.

| Tool | Arguments | What it does |
|---|---|---|
| `prepare_finetune_dataset` | `examples`, `run_id`, `validation_split` | Validates `{prompt, completion}` pairs and splits off validation at 10+ examples. Catches malformed data before training starts. |
| `fine_tune` | `run_id`, `base_model_id`, `use_qlora`, `num_epochs`, `learning_rate`, `lora_r` | LoRA (or QLoRA, needs a CUDA GPU) training. Adapter lands in `workspace/finetune/<run_id>/adapter/`. Downloads the base model on first use — budget real time. |
| `merge_and_export_model` | `run_id`, `export_mode` | `merged` (default) folds the adapter into the base model as a standalone directory; `adapter` just reports the adapter path. |
| `list_finetune_runs` | — | Runs in this session and their status. |

### 2.11 Documents ([swarn/capabilities/](swarn/capabilities/))

| Tool | Arguments | What it does |
|---|---|---|
| `extract_pdf_structured` | `path`, `tables_only`, `page` | Per page: prose plus every table as real rows and `records`. Use when you need the **data**, not searchability. No embedding. |
| `extract_pdf_document` | `path`, `page` | The whole PDF as a document tree — nested sections with headings, typed blocks (paragraph, list, key_values, table) in reading order, plus title, metadata, and a flat `fields` map of every label→value pair found anywhere. |
| `swarn_pdf_to_csv` | `path`, `out_path`, `out_dir`, `split_fused` | Tables to CSV files. Handles **borderless** tables by inferring columns from text alignment, which is why it succeeds on files that yield no tables otherwise. Consecutive same-width pages become one table. |
| `swarn_doc_ingest` | `path`, `backend`, `force` | Parse once into stored JSON (word-level text, bounding boxes, per-word confidence, line ids, page dimensions, tables). Optional — but it saves a full OCR pass per later question on scans. |
| `swarn_doc_ask` | `path`, `question`, `page`, `backend`, `include_raw` | Answer a question grounded in the document, returning the evidence: each cited figure with page, verbatim quote and bounding box, an annotated image, and the computation when arithmetic was needed. |
| `swarn_doc_inspect` | `path`, `page`, `annotate`, `backend`, `include_raw` | Extract key entities **with the box each value was read from**, and write an annotated page colour-coded by confidence (green >0.85, amber 0.60–0.85, red <0.60). |

`backend` is one of `text` (PDF text layer), `ocr` (local tesseract), `vlm`
(vision model API — needs `SWARN_VLM_API_KEY`), or `mock` (synthetic sample data,
never auto-selected for a real document). Omit it to auto-select.

### 2.12 MCP — external tool servers ([agent/integrations/mcp_integration.py](agent/integrations/mcp_integration.py))

| Tool | Arguments | What it does |
|---|---|---|
| `connect_mcp_server` | `server_name`, `command`, `args` | Launch an MCP server as a subprocess and register **all** its tools as `mcp_<server>_<tool>`. Same config shape as Claude Desktop, e.g. `command="npx", args=["-y", "@modelcontextprotocol/server-github"]`. |
| `list_mcp_servers` | — | Connected servers and how many tools each exposes. |
| `list_mcp_tools` | `server_name` | The registered remote tool names — call them directly, e.g. `mcp_github_create_issue`. |
| `disconnect_mcp_server` | `server_name` | Drop the server and all its tools. Call before reconnecting the same name. |

Remote tools go into the *same* `TOOL_REGISTRY` and dispatch through the same
`run_tool`. A role whose allow-list includes `connect_mcp_server` automatically
sees every `mcp_*` tool registered mid-run — see `get_tool_definitions`.

Swarn also runs *as* an MCP server: `swarn mcp-serve`.

### 2.13 Guardrails ([agent/observability/](agent/observability/))

| Tool | Arguments | What it does |
|---|---|---|
| `run_guardrail_benchmark` | — | Canned suite checking the prompt-injection guardrail flags known injections **and** does not flag benign content. |
| `get_guardrail_findings` | — | Patterns the live guardrail flagged in this session while scanning real tool results. |

---

## 3. CLI commands

```
swarn                              interactive REPL
swarn run <task> [paths...]        one-shot; paths make it a document question
swarn team <task>                  multi-agent pipeline (Planner→Coder→Reviewer→Tester)
swarn solve <task> --data <dir>    autonomous solution-tree search
swarn sessions / recall <id>       run history
swarn index <path>                 build the semantic index
swarn extract-pdf <pdf>            structured extraction
swarn to-csv <pdf>                 PDF tables to CSV
swarn doc-inspect <doc>            annotated entity extraction
swarn ingest <doc> / --list        pre-parse a document
swarn ask <question> <doc>         grounded document Q&A
swarn playbook [--clear]           learned cross-run lessons
swarn guardrail-benchmark          guardrail self-test
swarn serve [--port 8420]          observability dashboard
swarn config [--path]              show configuration
swarn mcp-serve                    expose Swarn's tools over MCP
```

Global flags (before the subcommand): `--no-banner`, `--model/-m`,
`--max-iterations`, `--no-stream`, `--sandbox-tools`, `--version/-v`.

Notable per-command flags:

- `run` — `--ask` / `--agent` force the document fast path or the ReAct agent; plus `--page`, `--backend`, `--no-annotate`, `--json`, `--no-progress`.
- `team` — `--no-tester`, `--no-report`, `--no-progress`.
- `solve` — `--steps/-s` (node budget, default 20), `--drafts` (default 4), `--time-limit/-t`, `--workers/-w` (parallel nodes), `--token-budget`, `--exec-timeout`, `--resume <run_id>` (adds `--steps` more nodes), `--no-learn`.
- `extract-pdf` — `--mode document|pages`, `--markdown`, `--tables-only`, `--page`, `--out/-o`, `--csv-dir`.
- `to-csv` — `--out/-o`, `--dir/-d`, `--page` (repeatable), `--split-fused`, `--quiet`.
- `doc-inspect` — `--page`, `--backend`, `--all-pages`, `--no-annotate`, `--out/-o`, `--quiet`. Omit the path entirely to inspect a generated mock invoice.
- `ingest` — `--backend`, `--render-pages`, `--force`, `--list`.
- `ask` — `--page`, `--backend`, `--no-annotate`, `--json`.

Add `--help` to any of them for the authoritative list.

### 3.1 REPL commands

Type a task at the prompt to run it, or use:

| Command | Effect |
|---|---|
| `/help` | Command list. |
| `/plan` | Show the current plan. |
| `/new` | Fresh conversation. |
| `/compact` | Compact the context. |
| `/undo` | Undo the last workspace change. |
| `/model [name]` | Show or set the model. |
| `/effort [level]` | Show or set reasoning effort. |
| `/status` | Session status. |
| `/resume [id]` | Resume a past session. |
| `/share-traces [on\|off]` | Toggle trace sharing. |
| `/yolo` | Auto-approve mode — skips approval prompts. |
| `history`, `recall <id>` | Session history. |
| `index <path>` | Index a directory for semantic search. |
| `report`, `team <task>`, `guardrails` | Report, multi-agent run, guardrail findings. |
| `ask`, `ingest`, `inspect`, `to-csv`, `extract-pdf` | The document subcommands, accepting every flag they take in the shell — they dispatch through the identical Click command. |

---

## 4. Configuration

Environment variables (all optional):

**Execution & sandbox** — `SWARN_SANDBOX` (`docker`/`subprocess`), `SWARN_SANDBOX_IMAGE`,
`SWARN_SANDBOX_READY_IMAGE` (default `swarn-sandbox:ready`), `SWARN_SANDBOX_PACKAGES`
(default `pandas numpy scipy scikit-learn matplotlib openpyxl`; set empty to skip),
`SWARN_SANDBOX_INSTALL_TIMEOUT` (900), `SWARN_EXEC_TIMEOUT` (300),
`SWARN_MAX_ITERATIONS`, `SWARN_AUTO_APPROVE`.

**Models** — `SWARN_DEPLOYED_MODEL`, `SWARN_DEPLOYED_BASE_URL`, `SWARN_DEPLOYED_API_KEY`,
`SWARN_CODE_MODEL`, `SWARN_FEEDBACK_MODEL`.

**Vision / documents** — `SWARN_VLM_MODEL`, `SWARN_VLM_BASE_URL`, `SWARN_VLM_API_KEY`,
`SWARN_DOC_BACKEND`.

**Search & budgets** — `SWARN_SEARCH_WORKERS`, `SWARN_KNOWLEDGE_DIR`,
`SWARN_CONTEXT_CHAR_BUDGET`, `SWARN_BUDGET_WARN_AT` (8), `SWARN_BUDGET_FINAL_AT` (3).

**Cleaning & reporting thresholds** — `SWARN_CLEAN_OUTLIER_Z` (3.0),
`SWARN_CLEAN_ROW_NULL_DROP` (0.5), `SWARN_CLEAN_COL_NULL_DROP` (0.5),
`SWARN_PLACEHOLDER_SHARE` (0.75), `SWARN_STALE_DATA_DAYS` (365),
`SWARN_THIN_ROWS_PER_LEVEL` (20), `SWARN_SEASONALITY_MIN_DAYS` (400),
`SWARN_CATEGORY_LEAK_SHARE` (0.5), `SWARN_CATEGORY_SAME_DOMAIN` (0.6),
`SWARN_MONEY_TOKENS` / `SWARN_COST_TOKENS` (comma-separated column-name hints).

**Paths & UI** — `SWARN_ARTIFACTS_DIR`, `SWARN_THEME` (`classic` default, or `lain`),
`SWARN_NO_BOOT_ANIM`, `SWARN_ENABLE_TRACING`, `SWARN_CORS_ORIGINS`.

`swarn config` prints the resolved configuration; `swarn config --path` prints the file location.

---

## 5. Two worked flows

**Analyse a spreadsheet and write it up**

```
inspect_workbook("sales.xlsx")           # see every sheet first
load_workbook("sales.xlsx")              # all sheets at once
validate_dataset("sales_Orders")
clean_dataset("sales_Orders")            # plan only, changes nothing
apply_cleaning("sales_Orders", [...])    # asks you; writes sales_Orders_clean
analyze_missing / rank_by / compare_groups / check_subgroups
write_report(dataset="sales_Orders_clean", situation=..., takeaways=[...])
finish_task("...")
```

**Train and ship a model**

```
load_csv("train.csv", "train") -> validate_dataset -> profile_features(target_col="churn")
engineer_features("train", target_col="churn", drop_cols=["customer_id"])
train_models("train_features", "churn")            # leaderboard
tune_hyperparameters("train_features", "churn", candidate="xgboost")
evaluate_model / feature_importance / plot_roc_curve
package_model(<artifact_id>)                       # FastAPI + Dockerfile
```
