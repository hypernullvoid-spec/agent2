# Swarn — Low-Level Design (LLD) Documentation

This documentation set is a **reverse-engineered, implementation-accurate** description of the
Swarn autonomous ML-engineering agent as it exists in this repository. Every statement is
derived from reading the actual source code across the `agent/` and `swarn/` packages,
`main.py`, and `tests/`. Where behavior could not be determined from the implementation, the
docs say so explicitly.

Companion document: [../TOOLS.md](../TOOLS.md) — the user-facing catalog of all 75 agent
tools with their arguments, the `swarn` CLI, the REPL commands, and the `SWARN_*` env vars.

**Audience:** engineers joining the project who need to understand, maintain, debug, and
extend the system without first reading the entire codebase.

## Reading order

New to the project? Read in this order:

1. [00_Executive_Summary.md](00_Executive_Summary.md) — what the system is, in one page
2. [01_Repository_Structure.md](01_Repository_Structure.md) — where everything lives
3. [02_System_Architecture.md](02_System_Architecture.md) — the big picture and component map
4. [04_Agent_Lifecycle.md](04_Agent_Lifecycle.md) — the two agent paradigms (ReAct loop + tree search)
5. [09_Tool_Execution.md](09_Tool_Execution.md) — the tool registry, the heart of the system
6. [../TOOLS.md](../TOOLS.md) — what the 75 tools actually do, and how to call them
7. [21_New_Developer_Guide.md](21_New_Developer_Guide.md) — onboarding walkthrough

## Full index

| # | Document | Covers |
|---|----------|--------|
| 00 | [Executive Summary](00_Executive_Summary.md) | System purpose, capabilities, key design decisions |
| 01 | [Repository Structure](01_Repository_Structure.md) | Every directory and file, with purpose and relationships |
| 02 | [System Architecture](02_System_Architecture.md) | Components, layering, dependency graph |
| 03 | [Startup Sequence](03_Startup_Sequence.md) | All four entry points, initialization order, config loading |
| 04 | [Agent Lifecycle](04_Agent_Lifecycle.md) | ReAct loop, tree search loop, multi-agent pipeline lifecycles |
| 05 | [Module Design](05_Module_Design.md) | Per-module deep dive: purpose, APIs, callers, internals |
| 06 | [Class Design](06_Class_Design.md) | Every important class: fields, methods, collaborators, lifecycle |
| 07 | [Data Flow](07_Data_Flow.md) | Where data originates, transforms, and lands |
| 08 | [State Management](08_State_Management.md) | All state: in-memory singletons, on-disk artifacts, ownership |
| 09 | [Tool Execution](09_Tool_Execution.md) | TOOL_REGISTRY, dispatch, sandboxing, MCP dynamic tools |
| 10 | [Prompt Construction](10_Prompt_Construction.md) | System prompts, role prompts, search prompts, knowledge injection |
| 11 | [Configuration](11_Configuration.md) | Env vars, `.env`, SearchConfig, defaults, precedence |
| 12 | [External Integrations](12_External_Integrations.md) | LLM endpoint, Docker, ChromaDB, SQLite, MCP, OTel, HF, cloud storage |
| 13 | [APIs](13_APIs.md) | Dashboard REST/WS API, MCP server tools, CLI commands, Python API |
| 14 | [Error Handling](14_Error_Handling.md) | "Errors as strings" contract, retries, correction policy, budgets |
| 15 | [Concurrency](15_Concurrency.md) | Parallel search, MCP event-loop thread, dashboard bridge, locks |
| 16 | [Security](16_Security.md) | Workspace path guard, sandboxing, secrets, injection guardrails |
| 17 | [Design Patterns](17_Design_Patterns.md) | Patterns actually present in the code, with evidence |
| 18 | [Sequence Diagrams](18_Sequence_Diagrams.md) | Step-by-step traces of every major workflow |
| 19 | [Dependency Graph](19_Dependency_Graph.md) | Module-level and package-level import graphs |
| 20 | [Extension Guide](20_Extension_Guide.md) | How to add tools, roles, backends, prompts, endpoints |
| 21 | [New Developer Guide](21_New_Developer_Guide.md) | Onboarding: how it starts, flows, and where code goes |
| 22 | [Technical Debt](22_Technical_Debt.md) | Observed debt: coupling, duplication, potential bugs |
| 23 | [Recommendations](23_Recommendations.md) | Practical, behavior-preserving improvements |

## Ground rules used while writing these docs

- **Traceability:** every architectural claim cites the implementing file (and usually the
  class/function). Line numbers refer to the state of the repo at the time of writing.
- **No invention:** features that are mentioned in docstrings/READMEs but do not exist in
  code (or the reverse) are called out explicitly.
- **Versioning context:** the codebase evolved through "Phases 1–16" plus "V2"/"V3" upgrades;
  module docstrings retain that vocabulary. These docs describe the *current* code, using
  phase numbers only as cross-references to the in-code comments.
