"""LLM layer: deployed-endpoint routing, normalization, message conversion, mock client."""

from agent.llm.base import TextBlock, ToolUseBlock, block_to_dict
from agent.llm.mock_client import MockLLMClient, text_response, tool_response
from agent.llm.openai_client import OpenAICompatClient
from agent.llm.router import (
    DEPLOYED_BASE_URL, DEPLOYED_MODEL_NAME, create_client,
)


def test_create_client_routes_to_deployed_endpoint():
    """No spec → the deployed endpoint client (test Qwen deployment for now)."""
    c = create_client(cache=False)
    assert isinstance(c, OpenAICompatClient)
    assert c.model == DEPLOYED_MODEL_NAME
    assert str(c.client.base_url).rstrip("/") == DEPLOYED_BASE_URL.rstrip("/")


def test_create_client_ignores_other_specs():
    """Old BYO-LLM specs are hard-routed to the deployed endpoint too."""
    for spec in ("claude-sonnet-4-6", "openai:gpt-4o", "ollama:llama3.1",
                 "vllm:meta-llama/Llama-3.1-70B@http://gpu:8000/v1"):
        c = create_client(spec, cache=False)
        assert isinstance(c, OpenAICompatClient)
        assert c.model == DEPLOYED_MODEL_NAME


def test_create_client_mock_passthrough():
    """'mock:*' still returns the offline mock client for tests."""
    c = create_client("mock:unit", cache=False)
    assert isinstance(c, MockLLMClient)
    assert c.model == "unit"


def test_block_serialization_roundtrip():
    t = TextBlock(text="hi")
    tu = ToolUseBlock(id="x1", name="run", input={"a": 1})
    assert block_to_dict(t) == {"type": "text", "text": "hi"}
    assert block_to_dict(tu)["name"] == "run"
    assert block_to_dict({"type": "tool_result", "tool_use_id": "x1", "content": "ok"})["type"] == "tool_result"


def test_mock_client_script_and_recording():
    mock = MockLLMClient(script=["hello", tool_response("t", {"k": "v"}, text="plan")])
    r1 = mock.call("sys", [{"role": "user", "content": "hi"}])
    assert r1.text == "hello"
    r2 = mock.call("sys", [{"role": "user", "content": "again"}])
    assert r2.tool_uses()[0].name == "t"
    assert len(mock.calls) == 2
    assert mock.total_usage.calls == 2


def test_openai_message_conversion():
    messages = [
        {"role": "user", "content": "solve it"},
        {"role": "assistant", "content": [
            TextBlock(text="thinking"),
            ToolUseBlock(id="c1", name="run_python", input={"code": "print(1)"}),
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "1"},
        ]},
    ]
    out = OpenAICompatClient._convert_messages("SYS", messages)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "solve it"}
    asst = out[2]
    assert asst["role"] == "assistant" and asst["tool_calls"][0]["function"]["name"] == "run_python"
    tool_msg = out[3]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"


def test_openai_tool_conversion():
    tools = [{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    conv = OpenAICompatClient._convert_tools(tools)
    assert conv[0]["type"] == "function" and conv[0]["function"]["name"] == "f"
    assert OpenAICompatClient._convert_tool_choice({"type": "tool", "name": "f"}) == {
        "type": "function", "function": {"name": "f"}}
    assert OpenAICompatClient._convert_tool_choice({"type": "any"}) == "required"


def test_retry_gives_up_on_non_retryable():
    class Boom(MockLLMClient):
        def _call_api(self, *a, **k):
            raise ValueError("invalid api key")
    c = Boom()
    try:
        c.call("s", [{"role": "user", "content": "x"}])
        assert False, "should have raised"
    except Exception as e:
        assert "invalid api key" in str(e)


# ─────────────────────────────────────────────── retry policy

def _client_that_always_raises(message, models=None):
    """A client whose API call always fails with `message`."""
    from agent.llm.base import BaseLLMClient

    class _Fake(BaseLLMClient):
        def __init__(self):
            super().__init__("some/model-v1")
            self.attempts = 0
            if models is not None:
                lister = type("_M", (), {"list": lambda _s: type(
                    "_R", (), {"data": [type("_X", (), {"id": m})() for m in models]})()})
                self.client = type("_C", (), {"models": lister()})()

        def _call_api(self, *a, **k):
            self.attempts += 1
            raise RuntimeError(message)

    return _Fake()


def test_context_overflow_is_permanent_and_never_retried():
    """Retrying a too-long request re-sends the same payload and burns tokens."""
    from agent.llm.base import LLMError
    c = _client_that_always_raises(
        "Error code: 400 - maximum context length is 1000000 tokens, "
        "your messages resulted in 1835298 tokens")
    try:
        c.call("sys", [])
    except LLMError as e:
        assert "permanently" in str(e) and "maximum context length" in str(e)
    assert c.attempts == 1, "a permanent failure must not be retried"


def test_missing_model_404_fails_immediately_with_a_suggestion():
    from agent.llm.base import LLMError
    c = _client_that_always_raises("Error code: 404 - not found",
                                   models=["other/model-a", "other/model-b"])
    try:
        c.call("sys", [])
    except LLMError as e:
        assert "does not exist at this endpoint" in str(e)
        assert "SWARN_DEPLOYED_MODEL" in str(e)
    assert c.attempts == 1


def test_404_for_a_model_that_does_exist_is_treated_as_temporary(monkeypatch=None):
    """The model is in the catalogue, so 404 means 'cannot reach it now'."""
    import agent.llm.base as base
    from agent.llm.base import LLMError
    c = _client_that_always_raises("Error code: 404 - not found",
                                   models=["some/model-v1", "other/model-a"])
    slept = []
    real_sleep = base.time.sleep
    base.time.sleep = lambda s: slept.append(s)
    try:
        c.call("sys", [])
    except LLMError as e:
        assert "after retries" in str(e)
    finally:
        base.time.sleep = real_sleep
    assert c.attempts == base.BaseLLMClient.MAX_RETRIES
    assert sum(slept) > 60, f"retry window too short to outlast a restart: {slept}"


def test_retry_after_hint_is_honoured():
    from agent.llm.base import _retry_after_seconds
    assert _retry_after_seconds(RuntimeError("rate limited; retry-after: 45")) == 45.0
    assert _retry_after_seconds(RuntimeError("plain failure")) is None
