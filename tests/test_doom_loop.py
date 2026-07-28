"""Doom-loop detection + context compaction (the ReAct-loop V3 upgrades)."""

from agent.agent_loop import compact_messages
from agent.doom_loop import DoomLoopDetector


def test_three_identical_calls_trip():
    d = DoomLoopDetector()
    assert d.record("read_file", {"path": "a.txt"}, "same content") is False
    assert d.record("read_file", {"path": "a.txt"}, "same content") is False
    assert d.record("read_file", {"path": "a.txt"}, "same content") is True


def test_polling_with_changing_results_never_trips():
    d = DoomLoopDetector()
    for i in range(10):
        assert d.record("job_status", {"id": "42"}, f"progress {i}%") is False


def test_pair_cycle_trips():
    d = DoomLoopDetector()
    d.record("a", {}, "ra")
    d.record("b", {}, "rb")
    d.record("a", {}, "ra")
    assert d.record("b", {}, "rb") is True


def test_different_args_do_not_trip():
    d = DoomLoopDetector()
    for i in range(6):
        assert d.record("read_file", {"path": f"f{i}.txt"}, "content") is False


def test_compaction_truncates_only_old_results():
    big = "x" * 50_000
    messages = [{"role": "user", "content": "task"}]
    for _ in range(12):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "thinking"}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": big}]})

    n = compact_messages(messages)
    assert n > 0
    # old results shrunk
    old_result = messages[2]["content"][0]["content"]
    assert len(old_result) < 5_000 and "compacted" in old_result
    # the most recent messages stay verbatim
    recent = messages[-1]["content"][0]["content"]
    assert recent == big


def test_compaction_noop_under_budget():
    messages = [{"role": "user", "content": "small task"},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "tiny"}]}]
    assert compact_messages(messages) == 0
    assert messages[1]["content"][0]["content"] == "tiny"
