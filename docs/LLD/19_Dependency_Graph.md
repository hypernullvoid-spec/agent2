# 19 — Dependency Graph

## Package-level

```mermaid
graph TD
    agent_pkg[agent/] --> llm_pkg[agent.llm]
    agent_pkg --> search_pkg[agent.search]
    search_pkg --> llm_pkg
    search_pkg -->|runner → execution, knowledge| agent_pkg
```

`agent.llm` depends on nothing else in the project (leaf package).
`agent.search` depends on `agent.llm`, `agent.execution`, and (lazily) `agent.knowledge`.

## Module import graph (static imports; dashed = lazy/function-level import)

```mermaid
graph TD
    main[main.py] --> ui & agent_loop & sandbox & self_correction & tools
    main -.-> memory & observability & orchestrator & llm

    cli --> llm
    cli -.-> agent_loop & orchestrator & search & memory & tools & observability & knowledge & mcp_server & dashboard

    agent_loop --> ui & llm & llm_client & tools & prompts & memory & self_correction & doom_loop

    orchestrator --> ui & agent_loop & llm & roles & self_correction
    roles --> prompts

    llm_client --> llm
    llm[agent.llm __init__] --> llm_base[llm.base] & llm_router[llm.router]
    llm_router --> llm_base
    llm_router -.-> llm_openai[llm.openai_client] & llm_mock[llm.mock_client]
    llm_openai --> llm_base
    llm_mock --> llm_base

    search_init[agent.search] --> s_config[search.config] & s_journal[search.journal] & s_runner[search.runner]
    s_runner --> execution & s_agent[search.agent] & s_config & s_journal & s_report[search.report] & s_static[search.static_check] & s_preview[search.data_preview]
    s_runner -.-> knowledge
    s_agent --> llm & s_config & s_journal
    s_config -.-> llm
    s_report --> s_config & s_journal

    sandbox --> execution
    tools -.-> sandbox & context_engine & memory & data_pipeline & feature_engineering & model_training & evaluation & deployment & mcp_integration & multimodal_rag & finetuning & observability & search_init

    feature_engineering --> data_pipeline
    model_training --> data_pipeline
    evaluation --> model_training
    deployment --> model_training
    multimodal_rag --> context_engine
    mcp_integration --> tools

    dashboard --> llm & memory
    dashboard -.-> agent_loop & self_correction & observability & s_journal & knowledge
    mcp_server --> llm
    mcp_server -.-> search_init & agent_loop & self_correction & observability
    knowledge -.-> llm_stub[（uses passed feedback client）]
```

## The one import cycle, and how it's broken

`tools.py ⇄ mcp_integration.py`:
- `mcp_integration.py` imports `TOOL_REGISTRY` from `tools.py` **at module top** (needs to
  write into it).
- `tools.py` imports `mcp_integration` **lazily inside the four MCP tool bodies**.

Because `tools.py` only imports `mcp_integration` at call time, the cycle never bites at
import. Similarly, `search/config.py` gets `DEFAULT_MODEL` through a lazy `_default_model()`
function explicitly commented "lazy import avoids cycles at module load".

## Lazy-import policy (systemic)

Function-level imports are used everywhere for three documented reasons:
1. **Startup speed / optional deps** — heavy libs (torch, chromadb, transformers, whisper,
   docker, matplotlib, optuna) load only when their tool runs; missing ones become error
   strings, not crashes.
2. **Cycle avoidance** — tools ↔ mcp_integration, search.config ↔ llm.
3. **CLI hygiene** — `swarn --help` shouldn't construct clients (`cli.py` comment).

## Third-party dependency layers

| Layer | Direct deps |
|---|---|
| Always needed | python-dotenv, openai, rich, typer |
| ReAct extras | (none additional — file tools are stdlib) |
| RAG | sentence-transformers, chromadb; pdfplumber, pytesseract, pillow (multimodal); openai-whisper (on demand) |
| ML pipeline | pandas, numpy, scikit-learn, xgboost, lightgbm, torch, optuna, matplotlib, joblib, openpyxl, pyarrow, sqlalchemy, pandera |
| Fine-tuning | peft, datasets, accelerate (+bitsandbytes on demand) |
| MCP | mcp |
| Observability | opentelemetry-api/sdk (+otlp exporter on demand) |
| Dashboard | fastapi, uvicorn, websockets |
| Sandbox | docker (commented out — install to enable) |

## Fan-in / fan-out hotspots

- **Highest fan-in:** `agent/llm` (imported by loop, shim, search, router consumers,
  cli, dashboard, mcp_server, orchestrator) and `tools.py` (loop, roles via names,
  mcp_integration, main, cli).
- **Highest fan-out:** `tools.py` (lazily touches 12 subsystems) and `cli.py` (touches
  everything by design).
- **Leaf utilities:** `ui.py` (imports only rich + stdlib), `doom_loop.py`,
  `static_check.py`, `data_preview.py`, `prompts.py`.

## Layering violations observed

- `evaluation.compare_models` reads `trainer._trained_models` directly (private attribute,
  self-acknowledged "read-only access, same module family").
- `multimodal_rag` reaches into `ContextEngine._collection/_embedder/_ensure_ready` and
  attaches `_clip_embedder` onto the engine instance (documented as deliberate reuse).
- `dashboard` reads `store._index` directly.
These are the only cross-module private accesses found; each is commented in code.
