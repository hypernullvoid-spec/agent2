"""
Tests for agent/data_bridge.py — reaching a loaded dataset from the sandbox.

The behaviour under test is a handover between two processes, so the tests come
in two halves: the selection/serialisation logic, which is pure and cheap, and a
few genuine round trips through the real sandbox, which are the only way to
prove the bootstrap actually executes there. The pure half also pins the
NEGATIVE cases — a name in a comment is not a reference, and a dataset the code
never mentions must not be materialised, because "bind everything" would quietly
turn every call into a full copy of the session.

Run via:  python tests/run_tests.py
"""

import os

import numpy as np
import pandas as pd

from agent.data_bridge import (
    SCRATCH_DIRNAME,
    _abs,
    _materialised,
    build_bootstrap,
    cleanup,
    referenced_datasets,
    reset,
    run_with_datasets,
)
from agent.ml.data_pipeline import get_data_pipeline
from agent.runtime.tools import TOOL_REGISTRY


def _put(name: str, df: pd.DataFrame) -> str:
    get_data_pipeline().datasets[name] = df
    return name


def _drop(*names: str) -> None:
    pipe = get_data_pipeline()
    for n in names:
        pipe.datasets.pop(n, None)
        _materialised.pop(n, None)


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "n": rng.integers(0, 100, n),
        "f": rng.normal(0, 1, n),
        "s": rng.choice(["a", "b", "c"], n),
        "d": pd.date_range("2024-01-01", periods=n, freq="D"),
        "b": rng.choice([True, False], n),
    })


# ────────────────────────────── which datasets are needed ─────────────────────


def test_a_bare_name_counts_as_a_reference():
    assert referenced_datasets("print(orders.shape)", ["orders", "products"]) == ["orders"]


def test_a_string_literal_counts_so_odd_names_still_work():
    assert referenced_datasets("df = load_dataset('sales-2024')", ["sales-2024"]) == ["sales-2024"]


def test_a_name_only_in_a_comment_is_not_a_reference():
    assert referenced_datasets("# products is irrelevant here\nprint(1)", ["products"]) == []


def test_unmentioned_datasets_are_not_selected():
    code = "print(orders.head())"
    assert referenced_datasets(code, ["orders", "products", "customers"]) == ["orders"]


def test_several_datasets_are_all_found():
    code = "m = orders.merge(products, on='k')"
    assert set(referenced_datasets(code, ["orders", "products", "x"])) == {"orders", "products"}


def test_broken_code_still_gets_its_datasets():
    # A syntax error is reported by the sandbox, not by refusing to bind data.
    assert referenced_datasets("print(orders.shape", ["orders"]) == ["orders"]


def test_nothing_is_selected_when_nothing_is_loaded():
    assert referenced_datasets("print(orders)", []) == []


# ────────────────────────────── materialising ─────────────────────────────────


def test_only_the_referenced_dataset_is_written_to_disk():
    reset()
    _put("br_a", _frame())
    _put("br_b", _frame())
    build_bootstrap(["br_a"])
    written = os.listdir(_abs(SCRATCH_DIRNAME))
    assert any(f.startswith("br_a") for f in written)
    assert not any(f.startswith("br_b") for f in written)
    cleanup(); _drop("br_a", "br_b"); reset()


def _data_file(base: str) -> str:
    """The frame's file, whichever format the bridge chose for it.

    Parquet when both sides of the handover can read it, CSV otherwise — an
    implementation detail these tests deliberately do not pin, so that changing
    it stays a one-line change in data_bridge rather than a test rewrite.
    """
    for ext in (".parquet", ".csv"):
        if os.path.exists(_abs(base + ext)):
            return _abs(base + ext)
    raise AssertionError(f"no data file written for {base!r}")


def _read_back(base: str):
    path = _data_file(base)
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def test_an_unchanged_frame_is_not_serialised_twice():
    reset()
    df = _frame()
    _put("br_cache", df)
    build_bootstrap(["br_cache"]); first = _materialised["br_cache"][1]
    mtime = os.path.getmtime(_data_file(first))
    build_bootstrap(["br_cache"])
    assert _materialised["br_cache"][1] == first
    assert os.path.getmtime(_data_file(first)) == mtime
    cleanup(); _drop("br_cache"); reset()


def test_a_replaced_frame_is_re_serialised():
    reset()
    _put("br_new", _frame(10))
    build_bootstrap(["br_new"])
    _put("br_new", _frame(20))                 # a different object
    build_bootstrap(["br_new"])
    assert len(_read_back(_materialised["br_new"][1])) == 20
    cleanup(); _drop("br_new"); reset()


def test_a_non_identifier_name_is_not_bound_as_a_variable():
    reset()
    _put("br-odd", _frame())
    _, bound = build_bootstrap(["br-odd"])
    assert bound == []                          # unusable as `br-odd = ...`
    src = open(_abs("_swarn_bootstrap.py"), encoding="utf-8").read()
    assert "'br-odd'" in src                    # but reachable via load_dataset
    cleanup(); _drop("br-odd"); reset()


# ────────────────────────────── real round trips ──────────────────────────────


def test_a_loaded_dataset_is_usable_in_the_sandbox():
    reset()
    _put("br_use", _frame(25))
    out = run_with_datasets("print('ROWS', len(br_use))")
    assert "ROWS 25" in out
    assert "datasets available in this call" in out
    _drop("br_use"); reset()


def test_dtypes_survive_the_handover():
    reset()
    _put("br_dt", _frame(30))
    out = run_with_datasets(
        "print('INT', str(br_dt['n'].dtype).startswith('int'))\n"
        "print('FLOAT', str(br_dt['f'].dtype).startswith('float'))\n"
        "print('DATE', str(br_dt['d'].dtype).startswith('datetime'))\n"
        "print('BOOL', str(br_dt['b'].dtype) == 'bool')")
    for line in ("INT True", "FLOAT True", "DATE True", "BOOL True"):
        assert line in out, out
    _drop("br_dt"); reset()


def test_values_are_not_altered_in_transit():
    reset()
    df = _frame(50)
    _put("br_vals", df)
    out = run_with_datasets("print('SUM', int(br_vals['n'].sum()))")
    assert f"SUM {int(df['n'].sum())}" in out
    _drop("br_vals"); reset()


def test_two_datasets_can_be_joined_in_one_call():
    reset()
    _put("br_l", pd.DataFrame({"k": [1, 2, 3], "x": ["a", "b", "c"]}))
    _put("br_r", pd.DataFrame({"k": [1, 2, 3], "y": [10, 20, 30]}))
    out = run_with_datasets("print('TOTAL', br_l.merge(br_r, on='k')['y'].sum())")
    assert "TOTAL 60" in out
    _drop("br_l", "br_r"); reset()


def test_a_published_frame_comes_back_into_the_registry():
    reset()
    _put("br_pub", _frame(40))
    out = run_with_datasets(
        "agg = br_pub.groupby('s', as_index=False)['n'].sum()\n"
        "publish_dataset('br_result', agg)")
    assert "registered back into the dataset registry" in out
    pipe = get_data_pipeline()
    assert "br_result" in pipe.datasets
    assert list(pipe.datasets["br_result"].columns) == ["s", "n"]
    _drop("br_pub", "br_result"); reset()


def test_a_published_frame_is_usable_by_the_other_tools():
    reset()
    _put("br_pub2", _frame(40))
    run_with_datasets("publish_dataset('br_handoff', br_pub2.head(12))")
    out = TOOL_REGISTRY["describe_dataset"]["func"](name="br_handoff")
    assert "12 rows" in out
    _drop("br_pub2", "br_handoff"); reset()


def test_a_name_the_code_never_mentions_is_absent_from_the_sandbox():
    # The check must not NAME the absent dataset — a string literal is itself a
    # reference (that is what makes load_dataset('odd-name') work), so naming it
    # would bind the very thing the test claims is missing.
    reset()
    _put("br_here", _frame(10))
    _put("br_gone", _frame(10))
    out = run_with_datasets("print('BOUND', sorted(n for n in dir() if n.startswith('br_')))\n"
                            "print(len(br_here))")
    assert "br_here" in out
    assert "br_gone" not in out, out
    _drop("br_here", "br_gone"); reset()


def test_naming_a_dataset_in_a_string_binds_it_on_purpose():
    # The cost of this rule is over-inclusion (one extra CSV write); the benefit
    # is that datasets whose names are not valid identifiers are reachable at all.
    reset()
    _put("br-dashed", _frame(15))
    out = run_with_datasets("print('N', len(load_dataset('br-dashed')))")
    assert "N 15" in out
    _drop("br-dashed"); reset()


def test_code_that_touches_no_dataset_still_runs():
    reset()
    assert "42" in run_with_datasets("print(6 * 7)")
    reset()


def test_an_error_reports_the_line_the_model_wrote():
    reset()
    _put("br_err", _frame(10))
    out = run_with_datasets("print(len(br_err))\nraise ValueError('boom')")
    assert "boom" in out
    # the bootstrap import is line 1 of the executed file; the raise is line 2 of
    # the model's code and must be reported as such, not as line 3
    assert "line 2" in out, out
    _drop("br_err"); reset()


# ────────────────────────────── the guard still stands ────────────────────────


def test_reading_the_file_behind_a_loaded_dataset_is_still_refused():
    reset()
    pipe = get_data_pipeline()
    pipe.datasets["br_guard"] = _frame(10)
    pipe.sources["br_guard"] = "csv:br_guard.csv"
    out = TOOL_REGISTRY["run_python"]["func"](
        code="import pandas as pd\nbr_guard = pd.read_csv('br_guard.csv')")
    assert "was NOT run" in out
    _drop("br_guard"); pipe.sources.pop("br_guard", None); reset()


def test_the_refusal_now_points_at_the_working_alternative():
    reset()
    pipe = get_data_pipeline()
    pipe.datasets["br_alt"] = _frame(10)
    pipe.sources["br_alt"] = "csv:br_alt.csv"
    out = TOOL_REGISTRY["run_python"]["func"](
        code="import pandas as pd\nbr_alt = pd.read_csv('br_alt.csv')")
    assert "ALREADY bound to a variable of the same name" in out
    _drop("br_alt"); pipe.sources.pop("br_alt", None); reset()
