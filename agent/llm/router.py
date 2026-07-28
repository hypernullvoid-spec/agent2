"""
LLM routing — hard-wired to the deployed model endpoint.

The former BYO-LLM layer (Anthropic / OpenAI / Ollama / vLLM / Gemini / Groq
presets + "[provider:]model[@base_url]" spec parsing) has been removed.
EVERY LLM call in the codebase — agent loop, team pipeline, search engine,
dashboard, MCP server — now routes through this one deployed,
OpenAI-compatible endpoint.

The only exception is "mock:*" specs, kept so the unit-test suite and
offline e2e runs never touch the network.

╔══════════════════════════════════════════════════════════════════════════╗
║  PRODUCTION ENDPOINT — CHANGE HERE                                       ║
║                                                                          ║
║  The DEPLOYED_* values below currently point at a TEST deployment        ║
║  (Qwen 3.5 9B served on Modal). When the production deployed model is    ║
║  ready, swap in its endpoint in ONE of two ways:                         ║
║    1. Edit the three DEPLOYED_* defaults below, or                       ║
║    2. Set env vars (they take precedence, no code change needed):        ║
║         SWARN_DEPLOYED_MODEL     — served model name                     ║
║         SWARN_DEPLOYED_BASE_URL  — OpenAI-compatible /v1 base URL        ║
║         SWARN_DEPLOYED_API_KEY   — auth key (use "dummy" if unsecured)   ║
║  Nothing else in the codebase needs to change.                           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
from typing import Optional

from agent.llm.base import BaseLLMClient

# ── deployed model configuration (single source of truth) ────────────────
# TODO(production): replace these test-deployment defaults with the
# production deployed model endpoint (see banner above).
DEPLOYED_MODEL_NAME = os.environ.get("SWARN_DEPLOYED_MODEL", "qwen3.5-9b")
DEPLOYED_BASE_URL = os.environ.get(
    "SWARN_DEPLOYED_BASE_URL",
    "https://hypernullvoid5869-55--qwen35-9b-serve.modal.run/v1",  # TEST: Qwen on Modal
)
DEPLOYED_API_KEY = os.environ.get("SWARN_DEPLOYED_API_KEY", "dummy")

# Kept as the canonical default-model string other modules import for
# display/logging defaults (journal, dashboard, CLI help).
DEFAULT_MODEL = DEPLOYED_MODEL_NAME


_client_cache: dict[str, BaseLLMClient] = {}
_ignored_spec_notices: set[str] = set()


def create_client(spec: Optional[str] = None, cache: bool = True) -> BaseLLMClient:
    """
    Return the client for the deployed model endpoint.

    `spec` is accepted for backward compatibility with old call sites and
    CLI flags, but — apart from "mock:*" (unit tests / offline runs) — it is
    IGNORED: all traffic is hard-routed to the deployed endpoint configured
    at the top of this file.
    """
    # mock passthrough — tests and offline e2e runs, never hits the network
    if spec and (spec == "mock" or spec.startswith("mock:")):
        from agent.llm.mock_client import MockLLMClient
        model = spec.split(":", 1)[1] if ":" in spec else "mock"
        key = f"mock:{model}"
        if cache and key in _client_cache:
            return _client_cache[key]
        client: BaseLLMClient = MockLLMClient(model=model)
        if cache:
            _client_cache[key] = client
        return client

    # one-time notice when a caller asked for some other model — it still
    # gets the deployed endpoint (that's the point of this mode)
    if spec and spec != DEPLOYED_MODEL_NAME and spec not in _ignored_spec_notices:
        _ignored_spec_notices.add(spec)
        print(f"[llm] model spec '{spec}' ignored — all calls route to the "
              f"deployed endpoint ({DEPLOYED_MODEL_NAME} @ {DEPLOYED_BASE_URL})")

    key = f"deployed:{DEPLOYED_MODEL_NAME}@{DEPLOYED_BASE_URL}"
    if cache and key in _client_cache:
        return _client_cache[key]

    from agent.llm.openai_client import OpenAICompatClient
    client = OpenAICompatClient(
        model=DEPLOYED_MODEL_NAME,
        base_url=DEPLOYED_BASE_URL,
        api_key=DEPLOYED_API_KEY,
    )
    if cache:
        _client_cache[key] = client
    return client
