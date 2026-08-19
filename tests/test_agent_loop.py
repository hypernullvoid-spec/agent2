"""
Tests for agent/agent_loop.py exit behaviors.
Run via:  python tests/run_tests.py
"""

import contextlib
import io

from agent.core.agent_loop import AgentLoop
from agent.llm.mock_client import MockLLMClient, text_response, tool_response


def _run(script) -> dict:
    loop = AgentLoop()
    loop.llm = MockLLMClient(model="mock", script=script)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return loop.run("test task")


def test_plain_text_answer_marks_complete():
    res = _run([text_response("The dataset needs cleaning: drop 3 columns.")])
    assert res["outcome"] == "complete"
    assert "drop 3 columns" in res["summary"]


def test_finish_task_still_complete():
    res = _run([tool_response("finish_task", {"summary": "done"})])
    assert res["outcome"] == "complete"
    assert res["summary"] == "done"


def test_blank_no_tool_still_no_tool_use():
    res = _run([text_response("   ")])
    assert res["outcome"] == "no_tool_use"


def test_tool_then_plain_text_answer_complete():
    res = _run([
        tool_response("list_files", {}, "listing"),
        text_response("Found dataset1.csv."),
    ])
    assert res["outcome"] == "complete"
    assert "dataset1.csv" in res["summary"]


# ─────────────────────────────────────────────── step budget

def test_budget_notice_stays_silent_early_and_escalates_late():
    """Hitting the cap delivers no answer at all, so the model must be warned
    while it still has calls left to write a summary."""
    from agent.core.agent_loop import _budget_notice
    cap = 30
    assert _budget_notice(1, cap) == ""            # no nagging early on
    assert _budget_notice(15, cap) == ""
    mid = _budget_notice(22, cap)                  # 8 left
    assert "[BUDGET]" in mid and "finish_task" in mid
    assert "NOW" not in mid                        # not panicking yet
    late = _budget_notice(28, cap)                 # 2 left
    assert "NOW" in late and "NO answer delivered" in late
    assert _budget_notice(cap, cap) == ""          # nothing useful to say at the cap


def test_budget_notice_scales_with_a_custom_cap():
    from agent.core.agent_loop import _budget_notice
    assert _budget_notice(1, 100) == ""
    assert "[BUDGET]" in _budget_notice(95, 100)


def test_system_prompt_explains_the_step_budget():
    from agent.messaging.prompts import SYSTEM_PROMPT
    assert "Step budget" in SYSTEM_PROMPT
    assert "finish_task" in SYSTEM_PROMPT
    assert "[BUDGET]" in SYSTEM_PROMPT              # the notice is not a surprise
