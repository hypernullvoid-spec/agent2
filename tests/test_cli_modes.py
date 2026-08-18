"""
Tests for the two CLI modes: headless (auto-approve, one-shot) and
interactive (approval-gated, conversation carries across turns).

Everything runs against the mock LLM, so nothing here touches the network.
"""

import json
import tempfile
from pathlib import Path

from agent.core.agent_loop import AgentLoop
from agent.config import CLIConfig, load_config, save_config
from agent.core.approval_policy import describe_operation, requires_approval
from agent.core.plan import clear_plan, get_current_plan, set_current_plan
from agent.llm.mock_client import text_response, tool_response


def _agent(script, **kwargs) -> AgentLoop:
    """An AgentLoop whose LLM is a scripted mock."""
    from agent.llm.mock_client import MockLLMClient

    agent = AgentLoop(model="mock", **kwargs)
    agent.llm = MockLLMClient(model="mock", script=list(script))
    return agent


# --- approval policy -------------------------------------------------------


def test_side_effecting_tools_need_approval():
    assert requires_approval("run_shell")
    assert requires_approval("write_file")
    assert requires_approval("install_package")


def test_read_only_tools_do_not_need_approval():
    assert not requires_approval("read_file")
    assert not requires_approval("list_files")
    assert not requires_approval("finish_task")


def test_describe_operation_is_human_readable():
    assert describe_operation("write_file", {"path": "a.py"}) == "write a.py"
    assert describe_operation("run_shell", {"command": "ls -la"}) == "ls -la"
    assert "pip install" in describe_operation("install_package", {"packages": "numpy"})


# --- headless: no approval callback means nothing is gated -----------------


def test_headless_runs_tools_without_approval():
    agent = _agent(
        [
            tool_response("write_file", {"path": "approval_probe.txt", "content": "hi"}),
            tool_response("finish_task", {"summary": "done"}),
        ]
    )
    result = agent.run("write a file")
    assert result["outcome"] == "complete"


# --- interactive: the approval callback gates side-effecting calls ---------


def test_approval_callback_is_consulted_for_side_effecting_tools():
    seen = []

    def approve(name, tool_input):
        seen.append(name)
        return True

    agent = _agent(
        [
            tool_response("write_file", {"path": "approval_probe.txt", "content": "hi"}),
            tool_response("finish_task", {"summary": "done"}),
        ],
        approval_callback=approve,
    )
    agent.run("write a file")
    # finish_task is read-only, so only write_file should have been offered.
    assert seen == ["write_file"]


def test_denied_tool_is_not_executed_and_run_continues():
    denied_path = Path(tempfile.gettempdir()) / "swarn_denied_probe.txt"
    if denied_path.exists():
        denied_path.unlink()

    agent = _agent(
        [
            tool_response("write_file", {"path": str(denied_path), "content": "nope"}),
            tool_response("finish_task", {"summary": "gave up"}),
        ],
        approval_callback=lambda name, inp: False,
    )
    result = agent.run("write a file")

    assert not denied_path.exists(), "denied tool call must not run"
    # A refusal is a normal tool result, not a crash: the loop keeps going.
    assert result["outcome"] == "complete"


# --- interactive: conversation continuity ----------------------------------


def test_keep_history_carries_context_across_runs():
    agent = _agent(
        [
            tool_response("finish_task", {"summary": "first"}),
            tool_response("finish_task", {"summary": "second"}),
        ],
        keep_history=True,
    )
    agent.run("first task")
    first_len = len(agent.history)
    assert first_len > 0

    agent.run("second task")
    assert len(agent.history) > first_len
    # The opening user message of the conversation is still the first task.
    assert agent.history[0]["content"] == "first task"


def test_one_shot_mode_keeps_no_history():
    agent = _agent([tool_response("finish_task", {"summary": "done"})])
    agent.run("a task")
    assert agent.history == []


def test_reset_conversation_clears_history():
    agent = _agent([tool_response("finish_task", {"summary": "done"})], keep_history=True)
    agent.run("a task")
    assert agent.history
    agent.reset_conversation()
    assert agent.history == []


def test_max_iterations_override_is_respected():
    # Never calls finish_task: the loop must stop at the configured cap.
    agent = _agent([text_response("thinking")] * 10, max_iterations=2)
    agent.llm.script = [tool_response("list_files", {"path": "."}) for _ in range(10)]
    result = agent.run("loop forever")
    assert result["outcome"] == "max_iterations"


# --- config persistence ----------------------------------------------------


def test_config_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cli_agent_config.json"
        cfg = CLIConfig(model_name="some/model", yolo_mode=True, reasoning_effort="high")
        save_config(cfg, path)

        loaded = load_config(path)
        assert loaded.model_name == "some/model"
        assert loaded.yolo_mode is True
        assert loaded.reasoning_effort == "high"


def test_missing_config_returns_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(Path(tmp) / "nope.json")
        assert cfg.yolo_mode is False
        assert cfg.tool_runtime == "local"


def test_corrupt_config_falls_back_to_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cli_agent_config.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_config(path).tool_runtime == "local"


def test_config_to_dict_is_json_serializable():
    json.dumps(CLIConfig().to_dict())


# --- bare-prompt argv rewrite ----------------------------------------------


def test_bare_prompt_is_rewritten_to_run():
    from agent.cli import _rewrite_bare_prompt

    assert _rewrite_bare_prompt(["do the thing"]) == ["run", "do the thing"]


def test_known_subcommands_are_left_alone():
    from agent.cli import _rewrite_bare_prompt

    assert _rewrite_bare_prompt(["run", "x"]) == ["run", "x"]
    assert _rewrite_bare_prompt(["sessions", "-n", "5"]) == ["sessions", "-n", "5"]
    assert _rewrite_bare_prompt(["mcp-serve"]) == ["mcp-serve"]


def test_option_values_are_not_mistaken_for_the_prompt():
    from agent.cli import _rewrite_bare_prompt

    assert _rewrite_bare_prompt(["--model", "mock", "team", "x"]) == ["--model", "mock", "team", "x"]
    assert _rewrite_bare_prompt(["-m", "mock", "do it"]) == ["-m", "mock", "run", "do it"]


def test_flags_only_argv_is_untouched():
    from agent.cli import _rewrite_bare_prompt

    assert _rewrite_bare_prompt(["--help"]) == ["--help"]
    assert _rewrite_bare_prompt([]) == []


# --- plan store (the /plan view) -------------------------------------------


def test_plan_store_normalizes_and_clears():
    clear_plan()
    assert get_current_plan() == []

    set_current_plan([{"content": "step one"}, {"content": "step two", "status": "bogus"}])
    plan = get_current_plan()
    assert [t["id"] for t in plan] == [1, 2]
    assert plan[1]["status"] == "pending"  # unknown status falls back

    clear_plan()
    assert get_current_plan() == []


def test_plan_display_renders_without_error():
    from agent.utils.terminal_display import format_plan_display

    set_current_plan([{"content": "a", "status": "completed"}, {"content": "b"}])
    out = format_plan_display()
    assert "1/2 done" in out
    clear_plan()
