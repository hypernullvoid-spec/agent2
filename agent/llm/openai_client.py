"""
OpenAI-compatible client for the DEPLOYED model endpoint (anything speaking
/chat/completions — the test Qwen deployment on Modal today, the production
deployed model later; endpoint configured in agent/llm/router.py).

The whole codebase talks Anthropic-style messages/tools; this client
translates in both directions so nothing else needs to know.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Optional

from agent.llm.base import BaseLLMClient, LLMResponse, TextBlock, ToolUseBlock, Usage


class OpenAICompatClient(BaseLLMClient):
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        super().__init__(model)
        from openai import OpenAI  # lazy import — only needed for this backend
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",  # local servers ignore it
            base_url=base_url,
        )
    # ── translation: Anthropic style → OpenAI style ────────────────────

    @staticmethod
    def _convert_tools(tools: list) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _convert_tool_choice(tool_choice: Optional[dict]):
        if not tool_choice:
            return None
        t = tool_choice.get("type")
        if t == "tool":
            return {"type": "function", "function": {"name": tool_choice["name"]}}
        if t == "any":
            return "required"
        return "auto"

    @staticmethod
    def _convert_messages(system: str, messages: list) -> list:
        out: list = [{"role": "system", "content": system}]
        for m in messages:
            role, content = m["role"], m["content"]
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            if role == "assistant":
                text_parts, tool_calls = [], []
                for b in content:
                    btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                    if btype == "text":
                        text_parts.append(b["text"] if isinstance(b, dict) else b.text)
                    elif btype == "tool_use":
                        bid = b["id"] if isinstance(b, dict) else b.id
                        name = b["name"] if isinstance(b, dict) else b.name
                        binput = b["input"] if isinstance(b, dict) else b.input
                        tool_calls.append({
                            "id": bid,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(binput)},
                        })
                msg: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
            else:  # user turn — may contain tool_result dicts
                text_parts = []
                for b in content:
                    btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                    if btype == "tool_result":
                        out.append({
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": str(b.get("content", "")),
                        })
                    elif btype == "text":
                        text_parts.append(b["text"] if isinstance(b, dict) else b.text)
                if text_parts:
                    out.append({"role": "user", "content": "\n".join(text_parts)})
        return out

    # ── the call ────────────────────────────────────────────────────────

    def _call_api(self, system, messages, tools, max_tokens, temperature, tool_choice) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=self._convert_messages(system, messages),
        )
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            tc = self._convert_tool_choice(tool_choice)
            if tc:
                kwargs["tool_choice"] = tc

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        content: list = []
        if msg.content:
            content.append(TextBlock(text=msg.content))
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            content.append(ToolUseBlock(id=tc.id or f"call_{uuid.uuid4().hex[:12]}",
                                        name=tc.function.name, input=args))

        stop = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}.get(
            choice.finish_reason, choice.finish_reason)
        u = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            calls=1,
        )
        return LLMResponse(content=content, stop_reason=stop, model=self.model, usage=usage)