"""
Back-compat shim — the original Phase 1 `LLMClient` now routes through
agent/llm/. Existing imports keep working, but note: every call is
hard-routed to the DEPLOYED model endpoint configured in
agent/llm/router.py ("PRODUCTION ENDPOINT — CHANGE HERE" banner there);
the `model` argument is kept only for display/log compatibility.
"""

from agent.llm import DEFAULT_MODEL, create_client


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._client = create_client(model)
        self.model = self._client.model  # actual served model, not the requested spec

    def call(self, system: str, messages: list, tools: list, max_tokens: int = 8192):
        return self._client.call(system, messages, tools=tools, max_tokens=max_tokens)

    @property
    def total_usage(self):
        return self._client.total_usage
