"""Configuration for the solution tree search engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _default_model() -> str:
    """Deployed model name (lazy import avoids cycles at module load)."""
    from agent.llm import DEFAULT_MODEL
    return DEFAULT_MODEL


@dataclass
class SearchConfig:
    # budgets
    steps: int = 20                      # max nodes to expand (additional nodes when resuming)
    time_limit_secs: Optional[int] = None  # wall-clock budget for the whole search
    exec_timeout: int = 600              # per-node code execution timeout (s)
    token_budget: Optional[int] = None   # max total LLM tokens (in+out) for the run

    # parallelism — how many draft/debug/improve nodes run concurrently.
    parallel_workers: int = field(default_factory=lambda: int(os.environ.get("SWARN_SEARCH_WORKERS", "1")))

    # tree policy
    num_drafts: int = 4                  # independent initial solutions
    debug_prob: float = 0.5              # chance to debug a buggy leaf vs. improve best
    max_debug_depth: int = 3             # give up on a branch after this many fix attempts
    improve_topk: int = 2                # improve is epsilon-greedy over the top-k good nodes

    # models — every non-"mock:" spec is hard-routed to the DEPLOYED endpoint
    # configured in agent/llm/router.py ("PRODUCTION ENDPOINT — CHANGE HERE"),
    # so these fields are display/log values plus the mock switch for tests.
    code_model: str = field(default_factory=lambda: os.environ.get("SWARN_CODE_MODEL", _default_model()))
    feedback_model: str = field(default_factory=lambda: os.environ.get("SWARN_FEEDBACK_MODEL", _default_model()))
    code_temperature: float = 0.7
    feedback_temperature: float = 0.2

    # environment
    runs_dir: str = field(default_factory=lambda: os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "runs")))
    copy_data: bool = True               # copy data_dir into the run workspace as ./input

    # cross-run knowledge (agent/knowledge.py)
    use_knowledge: bool = True           # inject playbook + similar past runs into prompts
    reflect: bool = False                # post-run reflection call -> playbook lessons
    knowledge_dir: Optional[str] = None  # default: <repo>/knowledge

    # static gate — reject guaranteed-broken code before the sandbox runs it
    static_gate: bool = True

    # context sizing
    max_term_out_chars: int = 6000       # execution output shown to the feedback model
    max_memory_nodes: int = 12           # journal entries summarized into draft/improve prompts
