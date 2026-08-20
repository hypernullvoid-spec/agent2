"""
Tests for agent/data_join.py — the guarded join.

Every test here is built around the scenario that motivated the module: four
orders worth 1,000, joined to a customer list holding one duplicate and missing
one customer. Rows in = 4, rows out = 4 — the fan-out and the orphan cancel —
and the revenue total silently becomes 700.

That is why almost nothing below asserts on row counts alone. A suite that only
checked shapes would pass on exactly the broken join this module exists to
catch, so the assertions are on the measures.
"""

import pandas as pd

import agent.data_cleaner
from agent.data_analysis import joins_for, reset_evidence
from agent.data_join import (
    _auto_keys,
    _match_counts,
    _measure_impact,
    _predicted_rows,
    _resolve_keys,
    join_datasets,
    register_into_swarn,
)
from agent.ml.data_pipeline import get_data_pipeline


# ─── harness ───────────────────────────────────────────────────────────────────

class _Answer:
    """Stand in for the human at the terminal, and remember what they were shown."""

    def __init__(self, reply="yes"):
        self.reply = reply
        self.question = ""
        self.calls = 0
        self._saved = None

    def __enter__(self):
        self._saved = agent.data_cleaner.ask_human

        def fake(question, options=None):
            self.calls += 1
            self.question = question
            return self.reply

        agent.data_cleaner.ask_human = fake
        return self

    def __exit__(self, *exc):
        agent.data_cleaner.ask_human = self._saved
        return False


def _fresh():
    pipe = get_data_pipeline()
    pipe.datasets.clear()
    pipe.sources.clear()
    reset_evidence()
    return pipe


def _cancelling():
    """The fan-out and the orphan cancel out: 4 rows in, 4 rows out, 1000 → 700."""
    pipe = _fresh()
    pipe.datasets["orders"] = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "customer_id": [10, 11, 12, 13],
        "revenue": [100.0, 200.0, 300.0, 400.0],
    })
    pipe.datasets["customers"] = pd.DataFrame({
        "customer_id": [10, 10, 11, 12, 99],      # 10 twice, 13 absent, 99 spare
        "city": ["Pune", "Pune", "Delhi", "Mumbai", "Goa"],
    })
    return pipe


def _pair(left, right):
    pipe = _fresh()
    pipe.datasets["a"] = left
    pipe.datasets["b"] = right
    return pipe


# ─── the prediction is exact, not a rule of thumb ─────────────────────────────

def test_match_counts_are_exact():
    pipe = _cancelling()
    counts = _match_counts(pipe.datasets["orders"], ["customer_id"],
                           pipe.datasets["customers"], ["customer_id"])
    assert list(counts) == [2, 1, 1, 0], list(counts)


def test_predicted_rows_per_direction():
    pipe = _cancelling()
    left, right = pipe.datasets["orders"], pipe.datasets["customers"]
    lm = _match_counts(left, ["customer_id"], right, ["customer_id"])
    rm = _match_counts(right, ["customer_id"], left, ["customer_id"])
    assert _predicted_rows("inner", lm, rm) == 4      # 2+1+1+0
    assert _predicted_rows("left", lm, rm) == 5       # the orphan is kept
    assert _predicted_rows("outer", lm, rm) == 6      # + spare customer 99


def test_measure_impact_sees_the_silent_loss():
    pipe = _cancelling()
    left = pipe.datasets["orders"]
    lm = _match_counts(left, ["customer_id"], pipe.datasets["customers"], ["customer_id"])
    impact = {c: (b, a, lost) for c, b, a, lost in
              _measure_impact(left, ["customer_id"], lm, "inner")}
    before, after, lost = impact["revenue"]
    assert before == 1000.0, before
    assert after == 700.0, after                      # 100*2 + 200 + 300 + 400*0
    assert lost == 400.0, lost


def test_identifier_columns_are_not_treated_as_measures():
    pipe = _cancelling()
    left = pipe.datasets["orders"]
    lm = _match_counts(left, ["customer_id"], pipe.datasets["customers"], ["customer_id"])
    cols = [c for c, *_ in _measure_impact(left, ["customer_id"], lm, "inner")]
    assert "revenue" in cols
    assert "order_id" not in cols                     # summing an id is noise


# ─── the human gate ────────────────────────────────────────────────────────────

def test_nothing_happens_without_approval():
    pipe = _cancelling()
    with _Answer("no"):
        out = join_datasets("orders", "customers", on="customer_id", how="inner")
    assert "cancelled" in out.lower(), out
    assert "orders_customers" not in pipe.datasets
    assert len(pipe.datasets["orders"]) == 4          # inputs untouched


def test_the_card_states_the_damage_before_asking():
    _cancelling()
    with _Answer("no") as human:
        join_datasets("orders", "customers", on="customer_id", how="inner")
    q = human.question
    assert "FAN-OUT" in q, q
    assert "DATA LOSS" in q, q
    assert "MEASURE LOSS" in q, q
    assert "1,000" in q and "700" in q, q             # the measure arithmetic


def test_approval_can_change_the_direction():
    pipe = _cancelling()
    with _Answer("yes how=left"):
        join_datasets("orders", "customers", on="customer_id", how="inner")
    assert len(pipe.datasets["orders_customers"]) == 5     # the orphan survived
    assert joins_for("orders_customers")[0]["how"] == "left"


def test_a_bare_no_is_not_read_as_approval():
    pipe = _cancelling()
    for reply in ("no", "none", "", "cancel"):
        _cancelling()
        with _Answer(reply):
            join_datasets("orders", "customers", on="customer_id")
        assert "orders_customers" not in get_data_pipeline().datasets, reply


def test_auto_approve_wording_is_accepted():
    """ask_human returns 'approve all' for non-interactive SWARN_AUTO_APPROVE runs."""
    pipe = _cancelling()
    with _Answer("approve all"):
        join_datasets("orders", "customers", on="customer_id", how="left")
    assert "orders_customers" in pipe.datasets


# ─── the join itself ───────────────────────────────────────────────────────────

def test_left_join_keeps_every_left_row():
    pipe = _cancelling()
    with _Answer("yes"):
        join_datasets("orders", "customers", on="customer_id", how="left")
    result = pipe.datasets["orders_customers"]
    assert len(result) == 5
    assert set(result["order_id"]) == {1, 2, 3, 4}    # nothing vanished
    assert result["revenue"].sum() == 1100.0          # 100 counted twice — and declared


def test_inner_join_reports_the_loss_it_caused():
    pipe = _cancelling()
    with _Answer("yes"):
        out = join_datasets("orders", "customers", on="customer_id", how="inner")
    assert "DROPPED" in out, out
    assert "revenue" in out and "700" in out, out
    assert pipe.datasets["orders_customers"]["revenue"].sum() == 700.0


def test_an_unchanged_row_count_is_not_taken_as_success():
    """4 rows in, 4 rows out — the case a row-count check waves straight through."""
    pipe = _cancelling()
    with _Answer("yes"):
        out = join_datasets("orders", "customers", on="customer_id", how="inner")
    assert len(pipe.datasets["orders_customers"]) == 4
    assert "1 row(s) of 'orders' were DROPPED" in out, out
    assert "-30.0%" in out, out


def test_fan_out_is_called_out_even_when_approved():
    pipe = _cancelling()
    with _Answer("yes"):
        out = join_datasets("orders", "customers", on="customer_id", how="left")
    assert "duplicate key" in out, out
    assert "Do NOT sum" in out, out


def test_inputs_are_never_modified():
    pipe = _cancelling()
    before_left = pipe.datasets["orders"].copy()
    before_right = pipe.datasets["customers"].copy()
    with _Answer("yes"):
        join_datasets("orders", "customers", on="customer_id", how="inner")
    pd.testing.assert_frame_equal(pipe.datasets["orders"], before_left)
    pd.testing.assert_frame_equal(pipe.datasets["customers"], before_right)


def test_clean_join_says_so():
    pipe = _pair(pd.DataFrame({"k": [1, 2, 3], "v": [10.0, 20.0, 30.0]}),
                 pd.DataFrame({"k": [1, 2, 3], "label": ["x", "y", "z"]}))
    with _Answer("yes"):
        out = join_datasets("a", "b", on="k")
    assert "No fan-out, no orphans, no key problems" in out, out
    assert pipe.datasets["a_b"]["v"].sum() == 60.0


def test_output_name_is_honoured():
    pipe = _pair(pd.DataFrame({"k": [1], "v": [1.0]}),
                 pd.DataFrame({"k": [1], "label": ["x"]}))
    with _Answer("yes"):
        join_datasets("a", "b", on="k", output_name="combined")
    assert "combined" in pipe.datasets


def test_source_records_the_join():
    pipe = _pair(pd.DataFrame({"k": [1], "v": [1.0]}),
                 pd.DataFrame({"k": [1], "label": ["x"]}))
    with _Answer("yes"):
        join_datasets("a", "b", on="k")
    assert pipe.sources["a_b"].startswith("join:a+b")


# ─── key quality ───────────────────────────────────────────────────────────────

def test_type_mismatch_is_named():
    _pair(pd.DataFrame({"k": ["1", "2"], "v": [1.0, 2.0]}),
          pd.DataFrame({"k": [1, 2], "label": ["x", "y"]}))
    with _Answer("no"):
        out = join_datasets("a", "b", on="k")
    assert "TYPE MISMATCH" in out, out


def test_zero_padding_is_named():
    _pair(pd.DataFrame({"k": ["007", "008", "009"], "v": [1.0, 2.0, 3.0]}),
          pd.DataFrame({"k": ["7", "8", "9"], "label": ["x", "y", "z"]}))
    with _Answer("no"):
        out = join_datasets("a", "b", on="k")
    assert "ZERO PADDING" in out, out


def test_whitespace_is_named():
    _pair(pd.DataFrame({"k": [" A12", "B13", "C14"], "v": [1.0, 2.0, 3.0]}),
          pd.DataFrame({"k": ["A12", "B13", "C14"], "label": ["x", "y", "z"]}))
    with _Answer("no"):
        out = join_datasets("a", "b", on="k")
    assert "WHITESPACE" in out, out


def test_case_difference_is_named():
    _pair(pd.DataFrame({"k": ["abc", "def"], "v": [1.0, 2.0]}),
          pd.DataFrame({"k": ["ABC", "DEF"], "label": ["x", "y"]}))
    with _Answer("no"):
        out = join_datasets("a", "b", on="k")
    assert "CASE" in out, out


def test_blank_keys_are_named():
    _pair(pd.DataFrame({"k": [1.0, None, 3.0], "v": [1.0, 2.0, 3.0]}),
          pd.DataFrame({"k": [1.0, 3.0], "label": ["x", "z"]}))
    with _Answer("no"):
        out = join_datasets("a", "b", on="k")
    assert "BLANK KEYS" in out, out


def test_a_join_that_matches_nothing_is_refused_without_asking():
    pipe = _pair(pd.DataFrame({"k": ["p", "q"], "v": [1.0, 2.0]}),
                 pd.DataFrame({"k": ["y", "z"], "label": ["x", "y"]}))
    with _Answer("yes") as human:
        out = join_datasets("a", "b", on="k")
    assert "NOT performed" in out, out
    assert human.calls == 0                          # never even asks
    assert "a_b" not in pipe.datasets


# ─── key resolution ────────────────────────────────────────────────────────────

def test_auto_key_prefers_the_id_column():
    left = pd.DataFrame({"customer_id": [1], "name": ["a"], "revenue": [1.0]})
    right = pd.DataFrame({"customer_id": [1], "name": ["a"], "city": ["p"]})
    assert _auto_keys(left, right) == ["customer_id"]


def test_ambiguous_keys_are_refused_not_guessed():
    _pair(pd.DataFrame({"x": [1], "y": [2], "v": [1.0]}),
          pd.DataFrame({"x": [1], "y": [2], "label": ["p"]}))
    with _Answer("yes") as human:
        out = join_datasets("a", "b")
    assert "no join key given" in out, out
    assert human.calls == 0


def test_different_key_names_on_each_side():
    pipe = _fresh()
    pipe.datasets["orders"] = pd.DataFrame({"cust": [1, 2], "revenue": [10.0, 20.0]})
    pipe.datasets["customers"] = pd.DataFrame({"id": [1, 2], "city": ["p", "q"]})
    with _Answer("yes"):
        join_datasets("orders", "customers", left_on="cust", right_on="id")
    assert pipe.datasets["orders_customers"]["revenue"].sum() == 30.0


def test_composite_key():
    pipe = _pair(
        pd.DataFrame({"r": ["N", "N", "S"], "y": [2024, 2025, 2024], "v": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"r": ["N", "N", "S"], "y": [2024, 2025, 2024], "t": [10, 20, 30]}))
    with _Answer("yes"):
        join_datasets("a", "b", on=["r", "y"])
    assert len(pipe.datasets["a_b"]) == 3


def test_missing_key_column_is_reported():
    _pair(pd.DataFrame({"k": [1], "v": [1.0]}), pd.DataFrame({"k": [1], "label": ["x"]}))
    with _Answer("yes"):
        assert "nope" in join_datasets("a", "b", on="nope")


def test_unknown_dataset_is_reported():
    pipe = _fresh()
    pipe.datasets["a"] = pd.DataFrame({"k": [1]})
    assert "no dataset named 'ghost'" in join_datasets("a", "ghost", on="k")


def test_bad_how_is_reported():
    _pair(pd.DataFrame({"k": [1]}), pd.DataFrame({"k": [1]}))
    assert "how must be one of" in join_datasets("a", "b", on="k", how="sideways")


def test_empty_dataset_is_reported():
    _pair(pd.DataFrame({"k": []}), pd.DataFrame({"k": [1]}))
    assert "no rows" in join_datasets("a", "b", on="k")


def test_resolve_keys_rejects_mismatched_lengths():
    left = pd.DataFrame({"a": [1], "b": [2]})
    right = pd.DataFrame({"c": [1], "d": [2]})
    _, _, err = _resolve_keys(left, right, None, ["a", "b"], ["c"])
    assert "one-for-one" in err


# ─── the ledger and the report ─────────────────────────────────────────────────

def test_the_join_is_recorded():
    _cancelling()
    with _Answer("yes"):
        join_datasets("orders", "customers", on="customer_id", how="inner")
    records = joins_for("orders_customers")
    assert len(records) == 1, records
    r = records[0]
    assert r["left"] == "orders" and r["right"] == "customers"
    assert r["how"] == "inner"
    assert r["dropped_rows"] == 1
    assert r["right_duplicate_keys"] == 1
    assert r["left_keys"] == ["customer_id"]


def test_a_cancelled_join_is_not_recorded():
    _cancelling()
    with _Answer("no"):
        join_datasets("orders", "customers", on="customer_id", how="inner")
    assert not joins_for("orders_customers")


def test_reset_evidence_clears_joins():
    _cancelling()
    with _Answer("yes"):
        join_datasets("orders", "customers", on="customer_id", how="inner")
    assert joins_for("orders_customers")
    reset_evidence()
    assert not joins_for("orders_customers")


def test_report_methodology_states_the_join():
    from agent.data_report import _provenance_lines
    pipe = _cancelling()
    with _Answer("yes"):
        join_datasets("orders", "customers", on="customer_id", how="inner")
    text = "\n".join(_provenance_lines(pipe.datasets["orders_customers"],
                                       "orders_customers")).replace("**", "")
    assert "inner join" in text, text
    assert "customer_id" in text, text
    assert "1 row(s) of `orders` were dropped" in text, text
    assert "duplicate key" in text, text
    assert "700" in text, text


def test_report_limitations_state_the_loss():
    from agent.data_report import _limitations
    pipe = _cancelling()
    with _Answer("yes"):
        join_datasets("orders", "customers", on="customer_id", how="inner")
    notes = " ".join(_limitations(pipe.datasets["orders_customers"], "orders_customers"))
    assert "Rows lost to a join" in notes, notes
    assert "revenue is understated" in notes, notes
    assert "Rows were duplicated by a join" in notes, notes


def test_a_clean_join_adds_no_scary_limitations():
    from agent.data_report import _limitations
    pipe = _pair(pd.DataFrame({"k": [1, 2, 3], "v": [10.0, 20.0, 30.0]}),
                 pd.DataFrame({"k": [1, 2, 3], "label": ["x", "y", "z"]}))
    with _Answer("yes"):
        join_datasets("a", "b", on="k")
    notes = " ".join(_limitations(pipe.datasets["a_b"], "a_b"))
    assert "Rows lost to a join" not in notes, notes
    assert "Rows were duplicated" not in notes, notes


def test_unrecorded_merge_still_gets_the_fallback_warning():
    """A raw pandas merge inside run_python leaves no ledger entry to read."""
    from agent.data_report import _provenance_lines
    pipe = _fresh()
    pipe.datasets["m"] = pd.DataFrame({"k": [1], "name_x": ["a"], "name_y": ["b"]})
    pipe.sources["m"] = "sandbox:run_python"
    text = "\n".join(_provenance_lines(pipe.datasets["m"], "m"))
    assert "NOT" in text and "recorded" in text, text
    assert "join_datasets" in text, text


def test_a_recorded_join_suppresses_the_fallback_guess():
    """Both would otherwise fire on a join that also clashed on a column name."""
    from agent.data_report import _provenance_lines
    pipe = _fresh()
    pipe.datasets["orders"] = pd.DataFrame({"k": [1, 2], "name": ["a", "b"], "v": [1.0, 2.0]})
    pipe.datasets["customers"] = pd.DataFrame({"k": [1, 2], "name": ["p", "q"]})
    with _Answer("yes"):
        join_datasets("orders", "customers", on="k", suffixes=["_x", "_y"])
    text = "\n".join(_provenance_lines(pipe.datasets["orders_customers"], "orders_customers"))
    assert "cannot be reproduced" not in text, text
    assert "left join" in text, text


# ─── registration ──────────────────────────────────────────────────────────────

def test_register_into_swarn():
    from agent.runtime.tools import TOOL_REGISTRY
    register_into_swarn()
    assert "join_datasets" in TOOL_REGISTRY
    entry = TOOL_REGISTRY["join_datasets"]
    assert entry["schema"]["required"] == ["left", "right"]
    assert "left" in entry["schema"]["properties"]
