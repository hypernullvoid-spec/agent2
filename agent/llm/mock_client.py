"""
Mock LLM — scripted responses for tests and offline end-to-end runs.

Feed it a list of responses; each call pops the next one. A response can be:
  - a str                      → one TextBlock
  - an LLMResponse             → returned as-is
  - a callable(system, messages, tools) → any of the above (dynamic scripting)

If the script runs out, `fallback` (default: echo text) is used.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

from agent.llm.base import BaseLLMClient, LLMResponse, TextBlock, ToolUseBlock, Usage


def text_response(text: str) -> LLMResponse:
    return LLMResponse(content=[TextBlock(text=text)], stop_reason="end_turn",
                       model="mock", usage=Usage(calls=1))


def tool_response(name: str, tool_input: dict, text: str = "") -> LLMResponse:
    content: list = []
    if text:
        content.append(TextBlock(text=text))
    content.append(ToolUseBlock(id=f"mock_{uuid.uuid4().hex[:12]}", name=name, input=tool_input))
    return LLMResponse(content=content, stop_reason="tool_use", model="mock", usage=Usage(calls=1))


class MockLLMClient(BaseLLMClient):
    def __init__(self, model: str = "mock", script: Optional[list] = None,
                 fallback: Optional[Callable] = None):
        super().__init__(model)
        self.script: list = list(script or [])
        self.fallback = fallback
        self.calls: list[dict] = []  # recorded for assertions

    def _call_api(self, system, messages, tools, max_tokens, temperature, tool_choice) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "tools": tools,
                           "tool_choice": tool_choice})
        item: Any
        if self.script:
            item = self.script.pop(0)
        elif self.fallback:
            item = self.fallback
        else:
            item = "mock response"

        if callable(item) and not isinstance(item, LLMResponse):
            item = item(system, messages, tools)
        if isinstance(item, str):
            item = text_response(item)
        return item
