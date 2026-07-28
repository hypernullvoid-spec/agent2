"""Static gate — the pre-execution diagnostics that save sandbox runs."""

from agent.search.static_check import static_check

GOOD = """
import numpy as np
x = np.arange(10)
print(f"Final Validation Metric: {x.mean()}")
"""


def test_valid_code_passes():
    assert static_check(GOOD) is None


def test_syntax_error_caught():
    err = static_check("def broken(:\n    pass\nprint('Final Validation Metric: 1')")
    assert err is not None and "SyntaxError" in err


def test_empty_code_caught():
    err = static_check("   \n  ")
    assert err is not None and "empty" in err


def test_missing_metric_print_caught():
    err = static_check("x = 1\nprint(x)")
    assert err is not None and "Final Validation Metric" in err


def test_missing_metric_ok_when_not_required():
    assert static_check("x = 1\nprint(x)", require_metric_print=False) is None


def test_input_call_caught():
    err = static_check("name = input('who? ')\nprint('Final Validation Metric: 1')")
    assert err is not None and "input()" in err


def test_never_vetoes_runtime_uncertainty():
    # runtime errors (bad import, wrong column) must NOT be gated —
    # the gate only rejects *guaranteed* failures
    code = "import nonexistent_module_xyz\nprint('Final Validation Metric: 1')"
    assert static_check(code) is None
