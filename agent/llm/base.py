"""
Provider-agnostic LLM layer — normalized types shared by every backend.

The agent loop only ever sees these types. Each provider client converts
its native wire format to/from them, so swapping Anthropic for OpenAI,
Ollama, vLLM, or Gemini requires zero changes anywhere else.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ─────────────────────────────────────────────── normalized content blocks

@dataclass
class TextBlock:
    text: str
    type: str = "text"

    def to_dict(self) -> dict:
        return {"type": "text", "text": self.text}


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

    def to_dict(self) -> dict:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


Block = TextBlock | ToolUseBlock


def block_to_dict(block: Any) -> dict:
    """Serialize any block (ours, a provider's native object, or a raw dict)."""
    if isinstance(block, dict):
        return block
    if hasattr(block, "to_dict"):
        return block.to_dict()
    # Anthropic SDK native blocks
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    raise ValueError(f"Cannot serialize block: {block!r}")


# ──────────────────────────────────────────────────── usage / cost tracking

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.calls += other.calls

    def summary(self) -> str:
        return (
            f"{self.calls} calls · {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out tokens"
            + (f" · {self.cache_read_tokens:,} cached" if self.cache_read_tokens else "")
        )


@dataclass
class LLMResponse:
    """Normalized response: what agent_loop.py and the search engine consume."""
    content: list  # list[Block]
    stop_reason: Optional[str] = None
    model: str = ""
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        """All text blocks joined — convenience for non-tool-use calls."""
        return "\n".join(b.text for b in self.content if getattr(b, "type", "") == "text")

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if getattr(b, "type", "") == "tool_use"]


# ─────────────────────────────────────────────────────────── base client

class LLMError(Exception):
    """Raised after all retries are exhausted."""


RETRYABLE_MARKERS = (
    "overloaded", "rate_limit", "rate limit", "429", "500", "502", "503",
    "529", "timeout", "timed out", "connection", "temporarily", "404",
    "provider returned error",
)

# Failures that are permanent: retrying re-sends the same bad request and can
# only fail again. A context-length overflow is the classic one — it wastes
# five attempts AND five times the tokens.
PERMANENT_MARKERS = (
    "maximum context length", "context_length_exceeded", "too many tokens",
    "reduce the length", "invalid_api_key", "authentication", "unauthorized",
    "permission denied", "billing", "insufficient_quota",
)

MAX_RETRY_DELAY = 60.0      # a model being restarted takes minutes, not seconds
_RETRY_AFTER_RE = re.compile(r"retry[-_ ]?after[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)", re.I)


def _retry_after_seconds(error: Exception) -> Optional[float]:
    """Honour the provider's own Retry-After hint when it sends one — it knows
    how long the wait actually is, and guessing shorter just wastes attempts."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset-after"):
            try:
                value = headers.get(key)
            except AttributeError:
                value = None
            if value:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    pass
    match = _RETRY_AFTER_RE.search(str(error))
    return float(match.group(1)) if match else None


class BaseLLMClient:
    """
    Common retry / accounting shell. Subclasses implement _call_api().

    call() signature is stable across providers:
        call(system, messages, tools=None, max_tokens=..., temperature=...,
             tool_choice=None) -> LLMResponse

    `messages` is Anthropic-style: [{"role": "user"|"assistant", "content":
    str | list[block|dict]}] where user-role list content may hold
    {"type": "tool_result", ...} dicts. Providers convert as needed.
    """

    # 6 attempts with the widened backoff below covers roughly two minutes.
    # That is deliberate: an upstream model being restarted takes minutes, and
    # the old ~17-second window guaranteed failure. Permanent errors now exit
    # immediately, so a long window costs nothing when the request is hopeless.
    MAX_RETRIES = 6

    def __init__(self, model: str):
        self.model = model
        self.total_usage = Usage()

    # subclasses override
    def _call_api(self, system, messages, tools, max_tokens, temperature, tool_choice) -> LLMResponse:
        raise NotImplementedError

    def call(
        self,
        system: str,
        messages: list,
        tools: Optional[list] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        tool_choice: Optional[dict] = None,
    ) -> LLMResponse:
        last_err: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._call_api(system, messages, tools or [], max_tokens, temperature, tool_choice)
                self.total_usage.add(resp.usage)
                return resp
            except Exception as e:  # noqa: BLE001 — provider SDKs raise many types
                last_err = e
                msg = str(e).lower()
                permanent = next((m for m in PERMANENT_MARKERS if m in msg), None)
                if permanent:
                    raise LLMError(
                        f"LLM call failed permanently ({permanent}) — retrying cannot help: {e}"
                    ) from e
                if not any(m in msg for m in RETRYABLE_MARKERS) or attempt == self.MAX_RETRIES - 1:
                    break
                if "404" in msg:
                    reason = self._diagnose_404()
                    if reason:
                        raise LLMError(f"LLM call failed permanently: {reason}") from e
                # exponential backoff, but wide enough to outlast a model being
                # restarted upstream; the provider's own hint always wins
                delay = _retry_after_seconds(e)
                if delay is None:
                    delay = min(4 * 2 ** attempt + random.random(), MAX_RETRY_DELAY)
                delay = min(delay, MAX_RETRY_DELAY)
                print(f"[llm] transient error ({type(e).__name__}); "
                      f"retry {attempt + 1}/{self.MAX_RETRIES} in {delay:.1f}s")
                time.sleep(delay)
        raise LLMError(f"LLM call failed after retries: {last_err}") from last_err

    def _diagnose_404(self) -> Optional[str]:
        """A 404 means either 'no such model' (permanent) or 'cannot reach it
        right now' (temporary). Ask the endpoint which, so a typo fails in one
        second instead of after several minutes of pointless waiting.

        Returns a message when the failure is permanent, else None.
        """
        client = getattr(self, "client", None)
        if client is None:
            return None
        try:
            available = [m.id for m in client.models.list().data]
        except Exception:  # noqa: BLE001 — cannot tell; assume temporary
            return None
        if not available or self.model in available:
            return None
        close = [m for m in available if self.model.split("/")[-1][:12] in m][:5]
        hint = f" Did you mean: {close}?" if close else f" {len(available)} models are available."
        return (f"model '{self.model}' does not exist at this endpoint.{hint} "
                f"Fix SWARN_DEPLOYED_MODEL in .env.")

    # convenience: plain text completion (no tools)
    def complete(self, system: str, prompt: str, **kw) -> str:
        return self.call(system, [{"role": "user", "content": prompt}], **kw).text
