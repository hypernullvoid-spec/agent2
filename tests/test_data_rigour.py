"""
Tests for the four checks that stop a plausible number being a wrong one.

Each of these exists because a real analysis fails the same way every time:

  partial period   An extract pulled mid-month puts 8 days against a full 31
                   and the tool announces a collapse. The most common false
                   alarm in reporting, and nothing in the shape of the data
                   reveals it.
  year-on-year     December always beats November. Period-on-period change
                   mostly measures the calendar, not the business.
  grain            The same order billed twice for different amounts is not an
                   identical row, so de-duplication leaves both and every total
                   counts it twice.
  effect size      With enough rows a 0.3% gap is statistically significant.
                   Significance answers "is it real"; on its own it licenses
                   calling a meaningless difference an outperformance.

And one that reaches outside the data entirely — reconcile, which is the only
check here that can catch having been handed the wrong extract.
"""

import numpy as np
import pandas as pd

import agent.data_cleaner
from agent.data_analysis import (
    _cohens_d,
    _eta_squared,
    _partial_last_period,
    _year_on_year,
    analyze_over_time,
    compare_groups,
    effects_for,
    grain_for,
    reconcile,
    reconciliations_for,
    reset_evidence,
)
from agent.data_cleaner import check_grain
from agent.data_report import _limitations, _provenance_lines
from agent.ml.data_pipeline import get_data_pipeline


# ─── harness ───────────────────────────────────────────────────────────────────

def _fresh():
    pipe = get_data_pipeline()
    pipe.datasets.clear()
    pipe.sources.clear()
    pipe.truncation.clear()
    reset_evidence()
    return pipe


def _daily_sales(last_day="2026-03-08", start="2024-01-01", seed=3):
    """Two years of seasonal orders, ending mid-month unless told otherwise."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, last_day, freq="D")
    season = 1 + 0.6 * np.sin((days.dayofyear / 365) * 2 * np.pi)
    rows = [(d, round(float(rng.gamma(2, 300)), 2))
            for d, s in zip(days, season) for _ in range(rng.poisson(8 * s))]
    return pd.DataFrame(rows, columns=["order_date", "revenue"])


def _twice_billed():
    return pd.DataFrame({
        "order_id": [4471, 4471, 4472, 4473, 4474],
        "billed_on": pd.to_datetime(["2026-02-03", "2026-02-04", "2026-02-03",
                                     "2026-02-05", "2026-02-06"]),
        "revenue": [5000.0, 5200.0, 800.0, 1200.0, 900.0],
    })


# ─── the unfinished period ─────────────────────────────────────────────────────

def test_an_unfinished_last_period_is_detected():
    dates = pd.Series(pd.date_range("2026-01-01", "2026-03-08", freq="D"))
    series = pd.Series([1.0, 1.0, 1.0],
                       index=pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"]))
    found = _partial_last_period(dates, series, "ME")
    assert found, found
    assert 0.15 < found["share"] < 0.35, found["share"]


def test_a_finished_last_period_is_not_flagged():
    dates = pd.Series(pd.date_range("2026-01-01", "2026-02-28", freq="D"))
    series = pd.Series([1.0, 1.0], index=pd.to_datetime(["2026-01-31", "2026-02-28"]))
    assert _partial_last_period(dates, series, "ME") == {}


def test_the_trend_tool_refuses_to_read_an_unfinished_month_as_a_fall():
    pipe = _fresh()
    pipe.datasets["sales"] = _daily_sales(last_day="2026-03-08")
    out = analyze_over_time("sales", "order_date", freq="ME",
                            value_col="revenue", how="sum")
    assert "THE LAST PERIOD IS NOT FINISHED" in out, out
    assert "EXCLUDED" in out, out
    # the headline range must stop at the last SETTLED period
    assert "2026-02-28" in out, out


def test_the_partial_period_does_not_become_the_biggest_fall():
    pipe = _fresh()
    pipe.datasets["sales"] = _daily_sales(last_day="2026-03-08")
    out = analyze_over_time("sales", "order_date", freq="ME",
                            value_col="revenue", how="sum")
    fall = [ln for ln in out.splitlines() if "Biggest fall" in ln]
    assert fall, out
    assert "2026-03" not in fall[0], fall[0]


def test_a_complete_series_gets_no_partial_warning():
    pipe = _fresh()
    pipe.datasets["sales"] = _daily_sales(last_day="2026-02-28")
    out = analyze_over_time("sales", "order_date", freq="ME",
                            value_col="revenue", how="sum")
    assert "NOT FINISHED" not in out, out


def test_the_registered_frame_marks_the_incomplete_period():
    """Code that reads the result rather than the message must still know."""
    pipe = _fresh()
    pipe.datasets["sales"] = _daily_sales(last_day="2026-03-08")
    analyze_over_time("sales", "order_date", freq="ME", value_col="revenue", how="sum")
    result = pipe.datasets["sales_over_time"]
    assert "period_complete" in result.columns
    assert bool(result["period_complete"].iloc[-1]) is False
    assert bool(result["period_complete"].iloc[0]) is True


# ─── year on year ──────────────────────────────────────────────────────────────

def test_year_on_year_needs_more_than_a_year():
    short = pd.Series(range(6), index=pd.date_range("2025-01-31", periods=6, freq="ME"))
    assert _year_on_year(short, "ME") is None
    long = pd.Series(range(24), index=pd.date_range("2024-01-31", periods=24, freq="ME"))
    assert _year_on_year(long, "ME") is not None


def test_year_on_year_compares_against_twelve_periods_back():
    series = pd.Series([100.0] * 12 + [110.0] * 12,
                       index=pd.date_range("2024-01-31", periods=24, freq="ME"))
    yoy = _year_on_year(series, "ME")
    assert abs(float(yoy.iloc[-1]) - 10.0) < 1e-9, float(yoy.iloc[-1])


def test_the_trend_tool_reports_year_on_year_when_it_can():
    pipe = _fresh()
    pipe.datasets["sales"] = _daily_sales(last_day="2026-02-28")
    out = analyze_over_time("sales", "order_date", freq="ME",
                            value_col="revenue", how="sum")
    assert "SAME PERIOD LAST YEAR" in out, out


def test_too_short_a_history_says_so_rather_than_staying_quiet():
    pipe = _fresh()
    pipe.datasets["sales"] = _daily_sales(start="2025-10-01", last_day="2026-02-28")
    out = analyze_over_time("sales", "order_date", freq="ME",
                            value_col="revenue", how="sum")
    assert "No year-on-year comparison" in out, out
    assert "cannot separate a real move from an ordinary seasonal one" in out, out


def test_a_swing_off_a_near_empty_base_is_not_the_headline():
    """The '+3,578%' case — a month with 3 orders followed by a normal one."""
    pipe = _fresh()
    rows = ([("2025-01-05", 100.0)] * 40 + [("2025-02-05", 100.0)] * 40
            + [("2025-03-05", 100.0)] * 1                      # the near-empty month
            + [("2025-04-05", 100.0)] * 40 + [("2025-05-05", 100.0)] * 40)
    pipe.datasets["spiky"] = pd.DataFrame(rows, columns=["d", "v"]).assign(
        d=lambda f: pd.to_datetime(f["d"]))
    out = analyze_over_time("spiky", "d", freq="ME", value_col="v", how="sum")
    rise = [ln for ln in out.splitlines() if "Biggest rise" in ln]
    assert rise, out
    assert "3900" not in rise[0] and "+3,900" not in rise[0], rise[0]
    assert "near-empty base" in out, out


# ─── grain ─────────────────────────────────────────────────────────────────────

def test_the_twice_billed_order_is_invisible_to_the_old_check():
    """The premise of check_grain — drop_duplicates cannot see this."""
    assert int(_twice_billed().duplicated().sum()) == 0


def test_check_grain_finds_it():
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed()
    out = check_grain("orders", keys=["order_id"])
    assert "NOT one row per" in out, out
    assert "1 key(s) appear more than once" in out, out


def test_check_grain_prices_the_duplication():
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed()
    out = check_grain("orders", keys=["order_id"])
    assert "13,100.00" in out, out            # as the table stands
    assert "7,900.00" in out, out             # counting each order once
    assert "39.7%" in out, out


def test_check_grain_separates_conflicts_from_copies():
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed()
    assert "CONFLICTING RECORDS" in check_grain("orders", keys=["order_id"])

    pipe.datasets["copies"] = pd.DataFrame({
        "order_id": [1, 1, 2], "revenue": [10.0, 10.0, 20.0]})
    out = check_grain("copies", keys=["order_id"])
    assert "genuine copies" in out, out
    assert "deduplicate op removes them safely" in out, out


def test_check_grain_registers_the_offending_rows():
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed()
    check_grain("orders", keys=["order_id"])
    assert len(pipe.datasets["orders_dupe_keys"]) == 2


def test_a_clean_grain_is_confirmed_not_just_silent():
    pipe = _fresh()
    pipe.datasets["clean"] = pd.DataFrame({"order_id": [1, 2, 3], "revenue": [1.0, 2.0, 3.0]})
    out = check_grain("clean", keys=["order_id"])
    assert "IS one row per" in out, out
    assert "exactly once" in out, out


def test_check_grain_suggests_keys_when_given_none():
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed()
    out = check_grain("orders")
    assert "order_id" in out, out
    assert "NOT unique" in out, out


def test_composite_grain():
    pipe = _fresh()
    pipe.datasets["monthly"] = pd.DataFrame({
        "region": ["N", "N", "S"], "month": ["2026-01", "2026-01", "2026-01"],
        "revenue": [1.0, 2.0, 3.0]})
    assert "NOT one row per" in check_grain("monthly", keys=["region", "month"])


def test_check_grain_reports_a_missing_column():
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed()
    assert "nope" in check_grain("orders", keys=["nope"])


# ─── effect size ───────────────────────────────────────────────────────────────

def test_cohens_d_matches_a_known_value():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a + 2.0                                # two SDs apart on sd ~1.58
    assert abs(_cohens_d(b, a) - 1.264) < 0.01, _cohens_d(b, a)


def test_eta_squared_is_zero_when_groups_are_identical():
    same = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])]
    assert abs(_eta_squared(same)) < 1e-9


def test_a_tiny_difference_across_many_rows_is_called_out():
    """Significant, and meaningless — the trap effect size exists for."""
    pipe = _fresh()
    rng = np.random.default_rng(0)
    n = 200_000
    pipe.datasets["big"] = pd.DataFrame({
        "region": np.repeat(["A", "B"], n // 2),
        "revenue": np.concatenate([rng.normal(1000, 300, n // 2),
                                   rng.normal(1003, 300, n // 2)]),
    })
    out = compare_groups("big", "revenue", "region")
    assert "IS IT REAL?" in out and "IS IT BIG?" in out, out
    assert "REAL BUT NOT MEANINGFUL" in out, out
    assert "negligible" in out, out


def test_a_large_difference_is_endorsed():
    pipe = _fresh()
    rng = np.random.default_rng(0)
    pipe.datasets["real"] = pd.DataFrame({
        "region": np.repeat(["A", "B"], 500),
        "revenue": np.concatenate([rng.normal(1000, 200, 500),
                                   rng.normal(1600, 200, 500)]),
    })
    out = compare_groups("real", "revenue", "region")
    assert "large" in out, out
    assert "both real and large enough to act on" in out, out
    assert "REAL BUT NOT MEANINGFUL" not in out, out


def test_the_effect_is_recorded():
    pipe = _fresh()
    rng = np.random.default_rng(0)
    n = 40_000
    pipe.datasets["big"] = pd.DataFrame({
        "region": np.repeat(["A", "B"], n // 2),
        "revenue": np.concatenate([rng.normal(1000, 50, n // 2),
                                   rng.normal(1001, 50, n // 2)]),
    })
    compare_groups("big", "revenue", "region")
    recorded = effects_for("big")
    assert recorded, recorded
    assert recorded[0]["effect_kind"] == "d"
    assert recorded[0]["negligible"] is True


# ─── reconciliation ────────────────────────────────────────────────────────────

def test_a_matching_figure_is_confirmed():
    pipe = _fresh()
    pipe.datasets["orders"] = pd.DataFrame({"revenue": [100.0, 200.0, 700.0]})
    out = reconcile("orders", "revenue", 1000.0, label="the finance report")
    assert "MATCHES" in out, out
    assert reconciliations_for("orders")[0]["matches"] is True


def test_a_small_rounding_gap_still_matches():
    pipe = _fresh()
    pipe.datasets["orders"] = pd.DataFrame({"revenue": [1000.0]})
    assert "MATCHES" in reconcile("orders", "revenue", 1002.0)


def test_a_real_gap_is_refused_and_explained():
    pipe = _fresh()
    pipe.datasets["orders"] = pd.DataFrame({"revenue": [100.0, 200.0, 400.0]})
    out = reconcile("orders", "revenue", 1000.0, label="the finance report")
    assert "DOES NOT MATCH" in out, out
    assert "Do not present this number" in out, out
    assert "Where the difference usually comes from" in out, out


def test_the_gap_explanation_uses_what_is_already_known():
    """A recorded duplicate key is the likeliest cause and should be named."""
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed()
    check_grain("orders", keys=["order_id"])
    out = reconcile("orders", "revenue", 7900.0, label="the finance report")
    assert "DOES NOT MATCH" in out, out
    assert "duplicate" in out.lower(), out


def test_reconcile_rejects_a_non_numeric_column():
    pipe = _fresh()
    pipe.datasets["orders"] = pd.DataFrame({"region": ["N", "S"]})
    assert "not a number column" in reconcile("orders", "region", 10.0)


# ─── all of it must reach the document ─────────────────────────────────────────

def test_duplication_lands_in_the_limitations():
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed()
    check_grain("orders", keys=["order_id"])
    notes = " ".join(_limitations(pipe.datasets["orders"], "orders"))
    assert "Rows are duplicated" in notes, notes
    assert "overstated" in notes, notes
    assert "conflicting records" in notes, notes


def test_a_negligible_effect_lands_in_the_limitations():
    pipe = _fresh()
    rng = np.random.default_rng(0)
    n = 100_000
    pipe.datasets["big"] = pd.DataFrame({
        "region": np.repeat(["A", "B"], n // 2),
        "revenue": np.concatenate([rng.normal(1000, 50, n // 2),
                                   rng.normal(1002, 50, n // 2)]),
    })
    compare_groups("big", "revenue", "region")
    notes = " ".join(_limitations(pipe.datasets["big"], "big"))
    assert "is not a meaningful one" in notes, notes
    assert "outperforming" in notes, notes


def test_a_failed_reconciliation_lands_in_the_limitations():
    pipe = _fresh()
    pipe.datasets["orders"] = pd.DataFrame({"revenue": [100.0, 200.0, 400.0]})
    reconcile("orders", "revenue", 1000.0, label="the finance report")
    notes = " ".join(_limitations(pipe.datasets["orders"], "orders"))
    assert "does not reconcile" in notes, notes


def test_a_passed_reconciliation_is_stated_in_the_methodology():
    """A clean check is worth reporting too — it is why the figure is trusted."""
    pipe = _fresh()
    pipe.datasets["orders"] = pd.DataFrame({"revenue": [100.0, 200.0, 700.0]})
    reconcile("orders", "revenue", 1000.0, label="the finance report")
    text = "\n".join(_provenance_lines(pipe.datasets["orders"], "orders"))
    assert "reconciles" in text, text
    assert "the finance report" in text, text


def test_a_verified_grain_is_stated_in_the_methodology():
    pipe = _fresh()
    pipe.datasets["clean"] = pd.DataFrame({"order_id": [1, 2, 3], "revenue": [1.0, 2.0, 3.0]})
    check_grain("clean", keys=["order_id"])
    text = "\n".join(_provenance_lines(pipe.datasets["clean"], "clean"))
    assert "one row per order_id" in text, text


def test_reset_evidence_clears_all_three():
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed()
    check_grain("orders", keys=["order_id"])
    reconcile("orders", "revenue", 1.0)
    assert grain_for("orders") and reconciliations_for("orders")
    reset_evidence()
    assert not grain_for("orders")
    assert not reconciliations_for("orders")
    assert not effects_for("orders")


# ─── registration ──────────────────────────────────────────────────────────────

def test_the_new_tools_are_registered():
    from agent.runtime.tools import TOOL_REGISTRY
    agent.data_cleaner.register_into_swarn()
    for tool in ("check_grain", "reconcile"):
        assert tool in TOOL_REGISTRY, tool
    assert TOOL_REGISTRY["reconcile"]["schema"]["required"] == ["name", "column", "expected"]


# ─── found by end-to-end testing ───────────────────────────────────────────────

def test_one_typo_does_not_turn_a_measure_into_a_code():
    """A quantity column of 1-15 with a single mistyped 9999 spans 9,998 and
    used to clear the code-gap test on that alone — after which its average was
    declared meaningless and it was dropped from the correlation map."""
    from agent.data_analysis import _looks_like_coded_categorical as is_code
    rng = np.random.default_rng(0)
    clean = pd.Series(rng.integers(1, 15, 4000), name="qty")
    typo = pd.Series(np.append(rng.integers(1, 15, 4000), 9999), name="qty")
    assert is_code(clean) is False
    assert is_code(typo) is False, "one outlier must not reclassify the column"


def test_a_column_named_as_a_code_is_taken_at_its_word():
    """1/2/3 is too dense for the gap test, and shape alone cannot tell a
    status code from a 1-5 rating. The name can."""
    from agent.data_analysis import _looks_like_coded_categorical as is_code
    rng = np.random.default_rng(0)
    assert is_code(pd.Series(rng.choice([1, 2, 3], 4000), name="status_code")) is True
    assert is_code(pd.Series(rng.choice([1, 2, 3], 4000), name="order_status")) is True
    # ...but a rating on the same shape is still a measurement
    assert is_code(pd.Series(rng.integers(1, 6, 4000), name="rating")) is False
    assert is_code(pd.Series(rng.integers(1, 8, 4000), name="score")) is False


def test_scattered_codes_are_still_detected():
    from agent.data_analysis import _looks_like_coded_categorical as is_code
    rng = np.random.default_rng(0)
    assert is_code(pd.Series(rng.choice([200, 301, 404, 500], 4000), name="resp")) is True
    assert is_code(pd.Series(rng.integers(2015, 2026, 4000), name="year")) is False


def _simpson_trial():
    """'new' wins overall, loses in BOTH difficulty levels."""
    rng = np.random.default_rng(0)
    rows = []
    for method, easy_n, hard_n, easy_p, hard_p in [("new", 900, 100, .90, .30),
                                                   ("old", 100, 900, .95, .40)]:
        for _ in range(easy_n):
            rows.append((method, "easy", int(rng.random() < easy_p)))
        for _ in range(hard_n):
            rows.append((method, "hard", int(rng.random() < hard_p)))
    return pd.DataFrame(rows, columns=["method", "difficulty", "success"])


def test_the_rate_form_of_simpsons_paradox_is_caught():
    """The textbook shape — a winner overall that loses in every subgroup.
    check_subgroups covered only the correlation form and refused this."""
    from agent.data_analysis import check_subgroups
    pipe = _fresh()
    pipe.datasets["trial"] = _simpson_trial()
    out = check_subgroups("trial", "method", "success", "difficulty")
    assert "SIMPSON'S PARADOX" in out, out
    assert "LOSES to 'old' in EVERY single 'difficulty'" in out, out
    assert "would recommend the worse option" in out, out


def test_a_consistent_winner_is_endorsed_not_warned_about():
    from agent.data_analysis import check_subgroups
    pipe = _fresh()
    rng = np.random.default_rng(1)
    rows = []
    for method, p_easy, p_hard in [("new", .95, .55), ("old", .80, .40)]:
        for _ in range(500):
            rows.append((method, "easy", int(rng.random() < p_easy)))
        for _ in range(500):
            rows.append((method, "hard", int(rng.random() < p_hard)))
    pipe.datasets["clean"] = pd.DataFrame(rows, columns=["method", "difficulty", "success"])
    out = check_subgroups("clean", "method", "success", "difficulty")
    assert "SIMPSON" not in out, out
    assert "wins in every 'difficulty'" in out, out


def test_a_non_numeric_outcome_is_still_refused():
    from agent.data_analysis import check_subgroups
    pipe = _fresh()
    pipe.datasets["trial"] = _simpson_trial()
    out = check_subgroups("trial", "success", "method", "difficulty")
    assert "must be a number column" in out, out
    assert "encode it as 1/0" in out, out


def test_a_duplicate_found_in_a_source_table_reaches_a_report_on_the_join():
    """The defect travels into the joined table; the finding has to travel too."""
    import agent.data_cleaner
    pipe = _fresh()
    pipe.datasets["orders"] = _twice_billed().assign(customer_id=["c1"] * 5)
    pipe.datasets["customers"] = pd.DataFrame({"customer_id": ["c1"], "city": ["Pune"]})
    check_grain("orders", keys=["order_id"])
    saved = agent.data_cleaner.ask_human
    agent.data_cleaner.ask_human = lambda q, o=None: "yes"
    try:
        from agent.data_join import join_datasets
        join_datasets("orders", "customers", on="customer_id", how="left", output_name="oc")
    finally:
        agent.data_cleaner.ask_human = saved
    notes = " ".join(_limitations(pipe.datasets["oc"], "oc"))
    assert "Rows are duplicated" in notes, notes
    assert "came from `orders`" in notes, notes
