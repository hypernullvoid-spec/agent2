"""Execution backends: subprocess correctness, timeouts, output shaping."""

import tempfile

from agent.execution import SubprocessBackend, _truncate


def _backend():
    return SubprocessBackend(workspace=tempfile.mkdtemp(prefix="swarn_test_ws_"))


def test_exec_python_captures_stdout_and_exit_code():
    r = _backend().exec_python("print('hello swarn')")
    assert r.ok and "hello swarn" in r.output and r.exit_code == 0


def test_exec_python_captures_errors():
    r = _backend().exec_python("raise RuntimeError('boom')")
    assert not r.ok and r.exit_code != 0 and "boom" in r.output


def test_exec_python_timeout():
    r = _backend().exec_python("import time; time.sleep(30)", timeout=2)
    assert r.timed_out and not r.ok
    assert "timed out" in r.as_text()


def test_exec_shell_runs():
    r = _backend().exec_shell("echo shell_ok")
    assert r.ok and "shell_ok" in r.output


def test_workspace_is_cwd():
    b = _backend()
    r = b.exec_python("open('marker.txt','w').write('x'); print('done')")
    assert r.ok
    import os
    assert os.path.isfile(os.path.join(b.workspace, "marker.txt"))


def test_truncation_keeps_head_and_tail():
    text = "A" * 1000 + "MIDDLE" + "B" * 1000
    out = _truncate(text, limit=200)
    assert out.startswith("A") and out.endswith("B") and "truncated" in out
