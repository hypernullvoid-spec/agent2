"""
Tests for agent/data_cleaner.py — human-in-the-loop data cleaning + analysis.

Run via:  python tests/run_tests.py   (or  pytest tests/test_data_cleaner.py)
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from unittest import mock

import pandas as pd

from agent.data_cleaner import (
    DataAnalyzer,
    DataCleaner,
    ask_human,
    parse_approval,
    register_into_swarn,
)

# ─────────────────────────────────────────────── fixtures


def _dirty_df() -> pd.DataFrame:
    base = pd.DataFrame({
        "city":       ["  Mumbai ", "Mumbai", "New York", "New York", "  Delhi ", "Chennai", "Chennai", "  Mumbai "],
        "price":      [100, 200, None, 200, 50, 1000000, 150, 99999],
        "rating":     ["4.5", "4.0", "3.9", "4.1", "4.2", "4.4", None, "4.3"],
        "in_stock":   ["yes", "no", "true", "1", "0", "yes", "no", "yes"],
        "joined":     ["2024/01/15", "2024-01-16", "Jan 17, 2024", "2024-01-18",
                       "2024-01-19", "2024-01-20", "2024/01/21", "2024-01-22"],
        "email":      ["a@b.com", "bad-email", "c@d.io", "e@f.org", "g@h.in",
                       "i@j.co", "k@l.com", "m@n.com"],
        "row_id":     list(range(8)),
    })
    return pd.concat([base, base.iloc[[0, 3]].copy()], ignore_index=True)


def _make_cleaner():
    c = DataCleaner(_dirty_df(), source="test")
    c.diagnose()
    return c


# ─────────────────────────────────────────────── diagnosis

def test_diagnose_finds_duplicates():
    c = _make_cleaner()
    assert any(op["action"] == "deduplicate" for op in c.ops.values())
    assert any("duplicate" in op["label"].lower() for op in c.ops.values())
    assert any(op["action"] == "impute" for op in c.ops.values())
    assert any(op["action"] == "uniform_dates" for op in c.ops.values())
    assert any(op["action"] == "rule_validate" for op in c.ops.values())
    assert any(op["action"] == "cap_outliers" for op in c.ops.values())
    assert any(op["action"] == "cast_types" for op in c.ops.values())
    assert any(op["action"] == "convert_booleans" for op in c.ops.values())


def test_diagnose_does_not_modify_data():
    df = _dirty_df().copy()
    c = DataCleaner(df.copy(), source="test")
    c.diagnose()
    pd.testing.assert_frame_equal(df, c.df)


def test_diagnose_high_null_column():
    df = pd.DataFrame({
        "good": [1, 2, 3, 4],
        "bad":  [1, None, None, None],
    })
    c = DataCleaner(df, source="test")
    c.diagnose()
    assert any(op["action"] == "drop_columns" and "bad" in op["label"] for op in c.ops.values())


def test_diagnose_less_info_rows():
    df = pd.DataFrame({
        "a": [1, None, 3, None],
        "b": [1, None, 3, None],
        "c": [1, None, 3, None],
    })
    c = DataCleaner(df, source="test")
    c.diagnose()
    drop_rows = [op for op in c.ops.values() if op["action"] == "drop_rows"]
    assert drop_rows and drop_rows[0]["extra"]["indices"] == [1, 3]


# ─────────────────────────────────────────────── apply

def test_apply_deduplicate_removes_only_exact_dups():
    c = _make_cleaner()
    dup = [op["id"] for op in c.ops.values() if op["action"] == "deduplicate"]
    before = len(c.df)
    c.apply_cleaning(dup)
    assert len(c.df) < before
    assert int(c.df.duplicated().sum()) == 0


def test_apply_impute_median_fills_nulls():
    c = _make_cleaner()
    imp = [op["id"] for op in c.ops.values() if op["action"] == "impute" and "price" in op["label"]]
    assert imp
    c.apply_cleaning(imp)
    assert c.df["price"].isnull().sum() == 0


def test_apply_flag_missing_adds_indicator():
    c = _make_cleaner()
    flag = [op["id"] for op in c.ops.values() if op["action"] == "flag_missing"]
    c.apply_cleaning(flag)
    assert any(c.endswith("_was_missing") for c in c.df.columns)


def test_apply_trim_and_case():
    c = DataCleaner(pd.DataFrame({"city": ["  Mumbai ", "Mumbai ", " New York", "NYC"]}), source="t")
    c.diagnose()
    trim = [op["id"] for op in c.ops.values() if op["action"] == "trim_text"]
    case = [op["id"] for op in c.ops.values() if op["action"] == "normalize_case"]
    c.apply_cleaning(trim + case)
    assert all(str(v) == str(v).strip() for v in c.df["city"])
    assert all(str(v) == str(v).lower() for v in c.df["city"])


def test_apply_cast_types():
    c = DataCleaner(pd.DataFrame({"rating": ["4.5", "3.0", "2", "5"]}), source="t")
    c.diagnose()
    cast = [op["id"] for op in c.ops.values() if op["action"] == "cast_types"]
    assert cast
    c.apply_cleaning(cast)
    assert pd.api.types.is_float_dtype(c.df["rating"]) or pd.api.types.is_numeric_dtype(c.df["rating"])


def test_apply_uniform_dates_iso():
    c = DataCleaner(pd.DataFrame({"joined": ["2024/01/15", "Jan 17, 2024", "2024-01-16"]}), source="t")
    c.diagnose()
    dates = [op["id"] for op in c.ops.values() if op["action"] == "uniform_dates"]
    assert dates
    c.apply_cleaning(dates)
    values = c.df["joined"].tolist()
    assert all(v == str(v) and len(v.split("-")[0]) == 4 for v in values if pd.notna(v))


def test_apply_convert_booleans():
    c = DataCleaner(pd.DataFrame({"flag": ["yes", "no", "1", "0"]}), source="t")
    c.diagnose()
    conv = [op["id"] for op in c.ops.values() if op["action"] == "convert_booleans"]
    assert conv
    c.apply_cleaning(conv)
    assert c.df["flag"].dropna().isin([True, False]).all()


def test_apply_cap_outliers_clips():
    c = DataCleaner(pd.DataFrame({"x": list(range(1, 100)) + [100000]}), source="t")
    c.diagnose()
    cap = [op["id"] for op in c.ops.values() if op["action"] == "cap_outliers"]
    assert cap
    c.apply_cleaning(cap)
    assert c.df["x"].max() < 100000


def test_apply_drop_columns_and_rows():
    df = pd.DataFrame({
        "keep":  [1, 2, 3, 4],
        "const": ["a", "a", "a", "a"],
        "part":  [1, 2, None, None],
    })
    c = DataCleaner(df, source="t")
    c.diagnose()
    drops = [op["id"] for op in c.ops.values() if op["action"] in ("drop_columns", "drop_rows")]
    assert drops
    c.apply_cleaning(drops)
    assert "const" not in c.df.columns
    assert "part" not in c.df.columns


def test_apply_ignores_unapproved_ops():
    c = _make_cleaner()
    before_cols = set(c.df.columns)
    c.apply_cleaning([])
    pd.testing.assert_frame_equal(c.df, c.df)
    assert set(c.df.columns) == before_cols


def test_apply_unknown_op_ids_safe():
    c = _make_cleaner()
    before = c.df.copy()
    result = c.apply_cleaning(["op999"])
    assert "skipped unknown" in result
    pd.testing.assert_frame_equal(c.df, before)


def test_cross_field_and_rule_validate_flag_only():
    df = pd.DataFrame({
        "start_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "end_date":   ["2024-06-01", "2024-01-15", "2024-09-01"],
        "email":      ["a@b.com", "not-an-email", "c@d.io"],
    })
    c = DataCleaner(df, source="t")
    c.diagnose()
    ops = [op["id"] for op in c.ops.values() if op["action"] in ("cross_field", "rule_validate")]
    before_rows = len(c.df)
    c.apply_cleaning(ops)
    assert len(c.df) == before_rows
    assert any(c.endswith("_valid") or "_vs_" in c for c in c.df.columns)


# ─────────────────────────────────────────────── swarn registry

def test_register_into_swarn_registers_tools():
    from agent.runtime.tools import TOOL_REGISTRY
    for name in ("clean_dataset", "apply_cleaning", "ask_human", "describe_dataset", "group_dataset"):
        assert name in TOOL_REGISTRY


def test_swarn_apply_keeps_original():
    from agent.ml.data_pipeline import get_data_pipeline
    from agent.runtime.tools import run_tool
    pipe = get_data_pipeline()
    original = _dirty_df().copy()
    pipe.datasets["swarn_src"] = original.copy()
    plan = run_tool("clean_dataset", {"name": "swarn_src"})
    assert "CLEANING PLAN" in plan
    with mock.patch("sys.stdin.isatty", return_value=False), \
         mock.patch("sys.stdout.isatty", return_value=False), \
         mock.patch.dict(os.environ, {"SWARN_AUTO_APPROVE": "1"}, clear=False):
        run_tool("apply_cleaning", {"name": "swarn_src", "operations": []})
    assert "swarn_src_clean" in pipe.datasets
    pd.testing.assert_frame_equal(pipe.datasets["swarn_src"], original)
    del pipe.datasets["swarn_src"]
    del pipe.datasets["swarn_src_clean"]


def test_swarn_apply_waits_for_approval():
    from agent.ml.data_pipeline import get_data_pipeline
    from agent.runtime.tools import run_tool
    pipe = get_data_pipeline()
    pipe.datasets["swarn_src"] = _dirty_df().copy()
    run_tool("clean_dataset", {"name": "swarn_src"})
    with mock.patch("sys.stdin.isatty", return_value=False), \
         mock.patch("sys.stdout.isatty", return_value=False), \
         mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SWARN_AUTO_APPROVE", None)
        result = run_tool("apply_cleaning", {"name": "swarn_src", "operations": ["op1"]})
    assert "No operations approved" in result
    assert "swarn_src_clean" not in pipe.datasets
    del pipe.datasets["swarn_src"]


# ─────────────────────────────────────────────── analysis

def test_describe_output():
    c = DataCleaner(_dirty_df(), source="t")
    c.diagnose()
    summary = DataAnalyzer(c.df).describe()
    assert "DATA SUMMARY" in summary
    assert "city" in summary
    assert "median" in summary


def test_group_dataset_aggregates():
    df = pd.DataFrame({
        "region": ["north", "south", "north", "south", "north"],
        "month":  ["jan", "jan", "feb", "jan", "feb"],
        "sales":  [10, 20, 30, 40, 50],
    })
    result = DataAnalyzer(df).group(["region", "month"], {"sales": "sum"})
    assert len(result) == 3  # south/feb never occurs
    assert result["sales"].sum() == 150


# ─────────────────────────────────────────────── approval parsing / ask_human

def test_parse_approval_variants():
    op_ids = ["op1", "op2", "op3", "op4"]
    assert parse_approval("all", op_ids) == (["op1", "op2", "op3", "op4"], {})
    assert parse_approval("none", op_ids) == ([], {})
    assert parse_approval("", op_ids) == ([], {})
    assert parse_approval("approve all", op_ids) == (["op1", "op2", "op3", "op4"], {})
    approved, params = parse_approval("op1 op3", op_ids)
    assert approved == ["op1", "op3"] and params == {}
    approved, params = parse_approval("op4:factor=83.2", op_ids)
    assert approved == ["op4"] and params == {"op4": {"factor": "83.2"}}
    # 'all' combined with per-op params approves everything, not just the named op
    approved, params = parse_approval("all op4:factor=83.2", op_ids)
    assert approved == op_ids and params == {"op4": {"factor": "83.2"}}


def test_ask_human_noninteractive_default_none():
    with mock.patch("sys.stdin.isatty", return_value=False), \
         mock.patch("sys.stdout.isatty", return_value=False), \
         mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SWARN_AUTO_APPROVE", None)
        assert ask_human("drop?") == "none"


def test_ask_human_noninteractive_auto_approve():
    with mock.patch("sys.stdin.isatty", return_value=False), \
         mock.patch("sys.stdout.isatty", return_value=False), \
         mock.patch.dict(os.environ, {"SWARN_AUTO_APPROVE": "1"}, clear=False):
        assert ask_human("drop?") == "approve all"


# ─────────────────────────────────────────────── CLI smoke

def test_cli_smoke_auto_approve():
    df = _dirty_df()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "data.csv")
        df.to_csv(src, index=False)
        out = os.path.join(tmp, "data_clean.csv")
        env = dict(os.environ)
        proc = subprocess.run(
            [sys.executable, "-m", "agent.data_cleaner", src, "-o", out, "--auto-approve"],
            capture_output=True, text=True, env=env, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "CLEANING PLAN" in proc.stdout
        assert "DATA SUMMARY" in proc.stdout
        assert os.path.exists(out)
        cleaned = pd.read_csv(out)
        assert cleaned.shape[1] <= df.shape[1] + 8  # flag cols added, dup rows removed


# ─────────────────────────────────────────────── robustness regressions

def test_op_targeting_dropped_column_is_skipped_not_fatal():
    """'approve all' where drop_columns removes a column a later op targets."""
    df = pd.DataFrame({"row_id": list(range(1, 100)) + [100000], "y": ["a"] * 100})
    c = DataCleaner(df, source="t")
    c.diagnose()
    result = c.apply_cleaning(c.op_ids())          # must not raise
    assert "removed by an earlier operation" in result
    assert "row_id" not in c.df.columns


def test_failing_op_does_not_discard_the_other_ops():
    df = pd.DataFrame({"x": list(range(1, 100)) + [100000], "y": ["a"] * 100})
    c = DataCleaner(df, source="t")
    c.diagnose()
    boom = mock.patch.object(DataCleaner, "_op_cap_outliers",
                             side_effect=RuntimeError("boom"), autospec=True)
    with boom:
        result = c.apply_cleaning(c.op_ids())
    assert "FAILED" in result and "boom" in result
    assert "y" not in c.df.columns                  # the constant-column drop still ran
    assert c.df["x"].max() == 100000                # the failed op changed nothing


def test_bad_bounds_param_is_a_clean_skip():
    df = pd.DataFrame({"x": list(range(1, 100)) + [100000]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    cap = [o["id"] for o in c.ops.values() if o["action"] == "cap_outliers"][0]
    result = c.apply_cleaning([cap], params={cap: {"bounds": "not-a-pair"}})
    assert "expected two numbers" in result
    assert "could not run" in result and "FAILED" not in result
    assert c.df["x"].max() == 100000


def test_unique_integer_measure_is_not_dropped_as_id():
    df = pd.DataFrame({
        "region":     ["n", "s", "n", "s", "n", "s"],
        "units_sold": [11, 27, 34, 48, 52, 63],     # all distinct, but real data
    })
    c = DataCleaner(df, source="t")
    c.diagnose()
    dropped = [col for o in c.ops.values() if o["action"] == "drop_columns" for col in o["columns"]]
    assert "units_sold" not in dropped


def test_named_ids_and_row_counters_are_still_dropped():
    df = pd.DataFrame({
        "order_id": [905, 118, 447, 232],           # distinct + name says id
        "seq":      [1, 2, 3, 4],                   # perfect row counter
        "amount":   [10.5, 20.5, 30.5, 40.5],
    })
    c = DataCleaner(df, source="t")
    c.diagnose()
    dropped = [col for o in c.ops.values() if o["action"] == "drop_columns" for col in o["columns"]]
    assert "order_id" in dropped and "seq" in dropped
    assert "amount" not in dropped


def test_zero_padded_codes_are_not_cast_to_numbers():
    df = pd.DataFrame({"zipcode": ["07001", "08002", "01003", "09004"], "v": [1.0, 2.0, 3.0, 4.0]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    assert not [o for o in c.ops.values() if o["action"] == "cast_types"]
    c.apply_cleaning(c.op_ids())
    assert c.df["zipcode"].tolist() == ["07001", "08002", "01003", "09004"]


def test_drop_columns_is_one_op_per_column():
    df = pd.DataFrame({"keep": [1.5, 2.5, 3.5], "const": ["a"] * 3, "uid": [7, 8, 9]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    drops = [o for o in c.ops.values() if o["action"] == "drop_columns"]
    assert len(drops) == 2 and all(len(o["columns"]) == 1 for o in drops)
    const_op = [o["id"] for o in drops if o["columns"] == ["const"]]
    c.apply_cleaning(const_op)                       # approve one, reject the other
    assert "const" not in c.df.columns and "uid" in c.df.columns


def test_missing_target_col_fails_loudly():
    df = pd.DataFrame({"f": [1.0, 2.0, None, 4.0], "price": [1.0, None, 3.0, 4.0]})
    c = DataCleaner(df, source="t", target_col="prce")   # typo
    plan = c.diagnose()
    assert plan.startswith("Error:") and "prce" in plan
    assert c.ops == {}


def test_reapplying_does_not_destroy_missing_indicators():
    df = pd.DataFrame({"x": [1.0, None, 3.0, 4.0, 5.0, 6.0]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    ops = [o["id"] for o in c.ops.values() if o["action"] in ("flag_missing", "impute")]
    c.apply_cleaning(ops)
    assert c.df["x_was_missing"].tolist() == [0, 1, 0, 0, 0, 0]
    result = c.apply_cleaning(ops)                   # second run must be a no-op
    assert "already applied" in result
    assert c.df["x_was_missing"].tolist() == [0, 1, 0, 0, 0, 0]


def test_capping_marks_the_rows_it_rewrote():
    df = pd.DataFrame({"revenue": [500.0 + i for i in range(20)] + [99999.0]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    original = df["revenue"].copy()
    cap = [o["id"] for o in c.ops.values() if o["action"] == "cap_outliers"]
    result = c.apply_cleaning(cap)
    marked = c.df["revenue_was_capped"].astype(bool)
    changed = c.df["revenue"] != original
    assert marked.equals(changed)                # every rewritten row is marked, and only those
    assert marked.loc[20]                        # the 99999 among them
    assert "original values" in result           # the summary is explicit about it


def test_capping_actually_removes_every_outlier_it_reported():
    """Detection used the IQR fence but capping used the 1st/99th percentile, so
    sentinel values defined their own bound and survived a clip that claimed to
    have handled them."""
    df = pd.DataFrame({"revenue": [700.0 + i for i in range(60)]
                                  + [99999.0, 120000.0, 88888.0, -5000.0]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    cap = [o for o in c.ops.values() if o["action"] == "cap_outliers"]
    reported = int(re.search(r"\((\d+) values", cap[0]["label"]).group(1))
    c.apply_cleaning([cap[0]["id"]])
    assert int(c.df["revenue_was_capped"].sum()) == reported
    assert c.df["revenue"].max() < 5000 and c.df["revenue"].min() > 0


def test_cross_field_does_not_pair_columns_by_accidental_substring():
    """'customer', 'history' and 'total' all contain 'to' — they are not end-dates."""
    df = pd.DataFrame({
        "start_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "end_date":   ["2024-06-01", "2024-01-15", "2024-09-01"],
        "customer":   ["Acme", "Globex", "Initech"],
        "total":      [10.0, 20.0, 30.0],
    })
    c = DataCleaner(df, source="t")
    c.diagnose()
    pairs = [tuple(o["columns"]) for o in c.ops.values() if o["action"] == "cross_field"]
    assert ("start_date", "end_date") in pairs
    assert all("customer" not in p and "total" not in p for p in pairs)


def test_capping_can_null_instead_of_rewrite():
    df = pd.DataFrame({"revenue": [500.0 + i for i in range(20)] + [99999.0]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    cap = [o["id"] for o in c.ops.values() if o["action"] == "cap_outliers"]
    c.apply_cleaning(cap, params={cap[0]: {"mode": "null"}})
    assert pd.isna(c.df.loc[20, "revenue"])      # blanked, not silently rewritten
    assert c.df.loc[20, "revenue_was_capped"] == 1
    assert c.df["revenue"].isna().sum() == int(c.df["revenue_was_capped"].sum())


def test_outliers_found_in_numbers_stored_as_text():
    df = pd.DataFrame({"revenue": ["500", "520", "480", "510", "495",
                                   "505", "515", "99999", "490", "502"]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    assert any(o["action"] == "cap_outliers" for o in c.ops.values())
    c.apply_cleaning(c.op_ids())
    assert c.df["revenue"].max() < 99999


def test_duplicates_hidden_by_spacing_are_offered():
    df = pd.DataFrame({"city": ["Mumbai", "  mumbai ", "Delhi", "Delhi "],
                       "v": [1.0, 1.0, 2.0, 2.0]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    dedup = [o for o in c.ops.values() if o["action"] == "deduplicate"]
    assert dedup, "rows that duplicate after standardisation must still be offered"
    c.apply_cleaning(c.op_ids())
    assert len(c.df) == 2


def test_category_spelling_variants_merge_but_distinct_labels_do_not():
    df = pd.DataFrame({"customer": ["Acme Corp", "Acme Corp.", "ACME  Corp", "Acme Corp",
                                    "Store 1", "Store 2"]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    std = [o["id"] for o in c.ops.values() if o["action"] == "standardize_categories"]
    assert std
    c.apply_cleaning(std)
    values = c.df["customer"].tolist()
    assert values[:4] == ["Acme Corp"] * 4          # most common spelling wins
    assert "Store 1" in values and "Store 2" in values   # never fuzzy-merged


def test_group_aware_imputation_beats_global_median():
    df = pd.DataFrame({
        "region": ["n"] * 5 + ["s"] * 5,
        "sales":  [10.0, 10.0, 10.0, 10.0, None, 1000.0, 1000.0, 1000.0, 1000.0, None],
    })
    c = DataCleaner(df, source="t")
    c.diagnose()
    imp = [o["id"] for o in c.ops.values() if o["action"] == "impute"]
    c.apply_cleaning(imp, params={imp[0]: {"by": "region"}})
    assert c.df.loc[4, "sales"] == 10.0            # filled from its own region
    assert c.df.loc[9, "sales"] == 1000.0


def test_fuzzy_merge_folds_entity_suffixes():
    df = pd.DataFrame({"customer": ["Acme Corp", "Acme Corp", "Acme Corporation",
                                    "Globex Ltd", "Globex Limited", "Initech"]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    fz = [o["id"] for o in c.ops.values() if o["action"] == "fuzzy_merge"]
    assert fz
    c.apply_cleaning(fz, params={fz[0]: {"apply": "1"}})
    values = c.df["customer"].tolist()
    assert values.count("Acme Corp") == 3          # 'Corporation' folded into 'Corp'
    assert values.count("Globex Ltd") == 2
    assert "Initech" in values


def test_fuzzy_merge_is_not_run_by_blanket_approval():
    df = pd.DataFrame({"customer": ["Acme Corp", "Acme Corporation", "Globex"]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    result = c.apply_cleaning(c.op_ids())          # plain 'approve all', no params
    assert "needing parameters" in result
    # other ops may lower-case it, but the two labels must remain distinct
    assert c.df["customer"].nunique() == 3


def test_fuzzy_merge_never_merges_labels_whose_digits_differ():
    df = pd.DataFrame({"site": ["Store 1", "Store 2", "Store 3", "Store 1"]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    fz = [o["id"] for o in c.ops.values() if o["action"] == "fuzzy_merge"]
    if fz:                                         # even at a reckless threshold
        c.apply_cleaning(fz, params={fz[0]: {"apply": "1", "threshold": "0.5"}})
    assert set(c.df["site"]) == {"Store 1", "Store 2", "Store 3"}


def test_fuzzy_merge_threshold_catches_typos():
    df = pd.DataFrame({"city": ["Mumbai"] * 4 + ["Mumbay", "Chennai"]})
    c = DataCleaner(df, source="t")
    c.ops = {"op1": {"id": "op1", "action": "fuzzy_merge", "label": "f",
                     "columns": ["city"], "extra": {}, "needs_param": True,
                     "recommended": False}}
    c.apply_cleaning(["op1"], params={"op1": {"threshold": "0.8"}})
    values = c.df["city"].tolist()
    assert values.count("Mumbai") == 5             # typo folded into the majority
    assert "Chennai" in values                     # unrelated label untouched


def test_impute_supports_interpolate_and_constant():
    df = pd.DataFrame({"reading": [1.0, None, 3.0, None, 5.0, 6.0, 7.0, 8.0]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    imp = [o["id"] for o in c.ops.values() if o["action"] == "impute"]
    c.apply_cleaning(imp, params={imp[0]: {"strategy": "interpolate"}})
    assert c.df["reading"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    c2 = DataCleaner(pd.DataFrame({"grade": ["a", None, "b", "a"]}), source="t")
    c2.diagnose()
    imp2 = [o["id"] for o in c2.ops.values() if o["action"] == "impute"]
    c2.apply_cleaning(imp2, params={imp2[0]: {"strategy": "constant", "value": "MISSING"}})
    assert c2.df["grade"].tolist() == ["a", "MISSING", "b", "a"]


def test_impute_rejects_an_unknown_strategy():
    df = pd.DataFrame({"x": [1.0, None, 3.0, 4.0]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    imp = [o["id"] for o in c.ops.values() if o["action"] == "impute"]
    result = c.apply_cleaning(imp, params={imp[0]: {"strategy": "magic"}})
    assert "unknown strategy" in result
    assert c.df["x"].isnull().sum() == 1            # untouched, not silently defaulted


def test_drift_report_flags_a_large_mean_shift():
    df = pd.DataFrame({"revenue": [500.0 + i for i in range(30)] + [900000.0]})
    c = DataCleaner(df, source="t")
    c.diagnose()
    cap = [o["id"] for o in c.ops.values() if o["action"] == "cap_outliers"]
    c.apply_cleaning(cap)
    report = c.drift_report()
    assert "revenue" in report and "mean" in report
    assert "⚠" in report                            # the shift is called out, not buried


def test_drift_report_notes_rows_and_columns_removed():
    c = _make_cleaner()
    c.apply_cleaning([o["id"] for o in c.ops.values()
                      if o["action"] in ("deduplicate", "drop_columns")])
    report = c.drift_report()
    assert "rows 10 → 8" in report


def test_manifest_records_what_was_applied():
    c = _make_cleaner()
    c.apply_cleaning([o["id"] for o in c.ops.values() if o["action"] == "deduplicate"])
    m = c.manifest()
    assert m["schema"] == "swarn.data_cleaner/1"
    assert m["rows_before"] == 10 and m["rows_after"] == 8
    assert [o["action"] for o in m["operations"]] == ["deduplicate"]
    assert json.loads(json.dumps(m))               # must be JSON-serialisable


def test_replay_reproduces_cleaning_on_a_fresh_extract():
    c = DataCleaner(_dirty_df(), source="jan")
    c.diagnose()
    c.apply_cleaning(c.op_ids())
    recipe = json.loads(json.dumps(c.manifest()))

    later = DataCleaner(_dirty_df(), source="feb")     # same schema, new extract
    result = later.replay(recipe)
    assert "Replayed recipe" in result
    assert list(later.df.columns) == list(c.df.columns)
    assert later.df.shape == c.df.shape


def test_replay_recomputes_row_drops_instead_of_reusing_positions():
    """Row positions from January must not be applied blindly to February."""
    jan = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1.0, None, 3.0, 4.0]})
    c = DataCleaner(jan, source="jan")
    c.diagnose()
    c.apply_cleaning([o["id"] for o in c.ops.values() if o["action"] == "drop_rows"])
    recipe = c.manifest()
    assert len(c.df) == 3

    feb = pd.DataFrame({"a": [9.0, 8.0, None, 6.0], "b": [9.0, 8.0, None, 6.0]})
    later = DataCleaner(feb, source="feb")
    later.replay(recipe)
    assert len(later.df) == 3
    assert later.df["a"].tolist() == [9.0, 8.0, 6.0]   # row 2 went, not row 1


def test_cli_writes_and_replays_a_cleaning_record():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "jan.csv")
        _dirty_df().to_csv(src, index=False)
        out = os.path.join(tmp, "jan_clean.csv")
        first = subprocess.run(
            [sys.executable, "-m", "agent.data_cleaner", src, "-o", out, "--auto-approve"],
            capture_output=True, text=True, cwd=root,
        )
        assert first.returncode == 0, first.stdout + first.stderr
        log = out + ".cleaning.json"
        assert os.path.exists(log)
        with open(log, encoding="utf-8") as fh:
            recipe = json.load(fh)
        assert recipe["operations"] and recipe["source"].endswith("jan.csv")

        src2 = os.path.join(tmp, "feb.csv")
        _dirty_df().to_csv(src2, index=False)
        out2 = os.path.join(tmp, "feb_clean.csv")
        second = subprocess.run(
            [sys.executable, "-m", "agent.data_cleaner", src2, "-o", out2, "--replay", log],
            capture_output=True, text=True, cwd=root,
        )
        assert second.returncode == 0, second.stdout + second.stderr
        assert "Replayed recipe" in second.stdout
        assert "CLEANING PLAN" not in second.stdout      # unattended: no prompting
        assert pd.read_csv(out2).shape == pd.read_csv(out).shape


def test_non_unique_index_does_not_delete_good_rows():
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1.0, None, 3.0, 4.0]},
                      index=[0, 1, 0, 1])
    c = DataCleaner(df, source="t")
    c.diagnose()
    drop = [o["id"] for o in c.ops.values() if o["action"] == "drop_rows"]
    c.apply_cleaning(drop)
    assert len(c.df) == 3                            # only the all-null row goes
    assert c.df["a"].tolist() == [1.0, 3.0, 4.0]


def test_heavy_tailed_column_is_not_recommended_for_capping():
    """Vote counts / revenue span orders of magnitude. The tail is real data;
    clipping it ties the top decile at one value and destroys the ranking."""
    rng = __import__("numpy").random.default_rng(11)
    votes = (10 ** rng.normal(3, 1.4, 400)).round() + 1          # log-normal, like real votes
    c = DataCleaner(pd.DataFrame({"VOTES": votes}), source="t")
    c.diagnose()
    caps = [o for o in c.ops.values() if o["action"] == "cap_outliers"]
    assert caps, "outliers should still be detected"
    assert caps[0]["recommended"] is False
    assert "NOT RECOMMENDED" in caps[0]["label"]
    assert "log scale" in caps[0]["label"] and "mode=null" in caps[0]["label"]
    assert caps[0]["extra"].get("heavy_tail") is True


def test_bounded_column_is_still_recommended_for_capping():
    """A rating column with a few typos is exactly what capping is for."""
    values = [7.0 + (i % 20) / 10 for i in range(200)] + [99.0, -5.0]
    c = DataCleaner(pd.DataFrame({"rating": values}), source="t")
    c.diagnose()
    caps = [o for o in c.ops.values() if o["action"] == "cap_outliers"]
    assert caps and caps[0]["recommended"] is True
    assert "NOT RECOMMENDED" not in caps[0]["label"]


def test_group_always_reports_how_many_rows_each_average_rests_on():
    """A 1-row 'yearly average' reads exactly like a 1,600-row one unless the
    count is there, so it is attached whether or not the caller asks."""
    df = pd.DataFrame({"yr": [1933, 1933, 2020, 2020, 2020],
                       "rating": [5.4, 6.0, 7.1, 7.2, 7.0]})
    result = DataAnalyzer(df).group(["yr"], {"rating": "mean"})
    assert "n_rows" in result.columns
    assert result.set_index("yr")["n_rows"].to_dict() == {1933: 2, 2020: 3}


def test_group_warns_when_a_groups_average_is_entirely_imputed():
    from agent.data_cleaner import _group_imputation_warnings
    df = pd.DataFrame({
        "yr":    [2020, 2020, 2022, 2022],
        "votes": [1500.0, 2000.0, 789.0, 789.0],
        "votes_was_missing": [0, 0, 1, 1],          # 2022 is 100% invented
    })
    notes = _group_imputation_warnings(df, ["yr"], {"votes": "mean"})
    assert notes and "100% imputed" in notes[0]
    assert "2022" in notes[0]
    assert "IS the fill value" in notes[0]
    # no markers → no noise
    assert _group_imputation_warnings(df[["yr", "votes"]], ["yr"], {"votes": "mean"}) == []


def test_group_stays_quiet_when_nothing_was_imputed():
    from agent.data_cleaner import _group_imputation_warnings
    df = pd.DataFrame({"yr": [2020, 2020], "votes": [1.0, 2.0], "votes_was_missing": [0, 0]})
    assert _group_imputation_warnings(df, ["yr"], {"votes": "mean"}) == []


def test_validation_flags_a_value_that_leaked_between_category_columns():
    """'Raksha Bandhan' — an occasion — sat in the Category column of a real
    products file and passed every check: non-null string, clean inferred schema,
    no z-score applies to text. It was then reported as an 8.4% product category.
    """
    import pandas as pd
    from agent.ml.data_pipeline import get_data_pipeline
    pipe = get_data_pipeline()
    pipe.datasets["dq"] = pd.DataFrame({
        "Category": ["Sweets", "Cake", "Raksha Bandhan", "Sweets"],
        "Occasion": ["Diwali", "Birthday", "Raksha Bandhan", "Holi"],
    })
    out = pipe.validate_dataset("dq")
    assert "category consistency" in out
    assert "Raksha Bandhan" in out and "leaked" in out
    pipe.datasets.pop("dq", None)


def test_validation_stays_quiet_when_category_columns_do_not_overlap():
    import pandas as pd
    from agent.ml.data_pipeline import get_data_pipeline
    pipe = get_data_pipeline()
    pipe.datasets["dq_ok"] = pd.DataFrame({
        "Category": ["Sweets", "Cake", "Mugs", "Sweets"],
        "Occasion": ["Diwali", "Birthday", "Holi", "Holi"],
    })
    out = pipe.validate_dataset("dq_ok")
    assert "nothing to flag" in out
    pipe.datasets.pop("dq_ok", None)


def test_validation_does_not_call_paired_columns_a_leak():
    """'origin'/'destination' share their WHOLE vocabulary by design — two roles
    of one entity type. Calling that a duplicate or a leak is a wrong assertion
    about correct data, which is worse than saying nothing."""
    import pandas as pd
    from agent.ml.data_pipeline import get_data_pipeline
    pipe = get_data_pipeline()
    codes = ["DEL", "BOM", "BLR", "MAA", "CCU"]
    pipe.datasets["legs"] = pd.DataFrame({
        "origin": codes * 4,
        "destination": (codes[2:] + codes[:2]) * 4,
    })
    out = pipe.validate_dataset("legs")
    assert "nothing to flag" in out
    assert "leaked" not in out and "duplicate copy" not in out
    pipe.datasets.pop("legs", None)


def test_validation_still_flags_identical_columns():
    import pandas as pd
    from agent.ml.data_pipeline import get_data_pipeline
    pipe = get_data_pipeline()
    pipe.datasets["dupe"] = pd.DataFrame({"a": ["x", "y", "z"] * 4, "b": ["x", "y", "z"] * 4})
    out = pipe.validate_dataset("dupe")
    assert "identical row for row" in out
    pipe.datasets.pop("dupe", None)
