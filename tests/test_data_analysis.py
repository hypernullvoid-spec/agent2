"""
Tests for agent/data_analysis.py — exploratory analysis + visualisation.

Every tool must return its finding in WORDS as well as saving a chart: the
model cannot see the PNG, so a path-only return would be useless to it.
Several tests below assert on that text rather than on the image.

Run via:  python tests/run_tests.py   (or  pytest tests/test_data_analysis.py)
"""

import os
import re

import numpy as np
import pandas as pd

from agent.data_analysis import (
    PLOTS_SUBDIR,
    WORKSPACE_DIR,
    analyze_correlations,
    analyze_dataset,
    analyze_missing,
    analyze_multivalue,
    analyze_over_time,
    check_subgroups,
    compare_groups,
    pivot_dataset,
    plot_column,
    plot_relationship,
)
from agent.ml.data_pipeline import get_data_pipeline

# ─────────────────────────────────────────────── fixtures


def _sales_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 120
    region = np.array(["north", "south"] * (n // 2))
    qty = rng.integers(1, 20, n).astype(float)
    revenue = np.where(region == "north", 900.0, 600.0) + qty * 5 + rng.normal(0, 12, n)
    return pd.DataFrame({
        "order_id": [f"ORD{i:04d}" for i in range(n)],
        "region": region,
        "customer": rng.choice(["Acme", "Globex"], n),
        "signup": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 400, n), unit="D"),
        "qty": qty,
        "revenue": revenue,
    })


def _load(name: str = "an_test", df: pd.DataFrame = None) -> str:
    pipe = get_data_pipeline()
    pipe.datasets[name] = _sales_df() if df is None else df
    return name


def _cleanup(*names: str) -> None:
    pipe = get_data_pipeline()
    for n in names:
        pipe.datasets.pop(n, None)


def _chart_exists(result: str) -> bool:
    """The saved path is reported relative to workspace/ — check it landed."""
    for token in result.replace("\n", " ").split():
        if token.startswith(PLOTS_SUBDIR + os.sep) and token.endswith(".png"):
            return os.path.exists(os.path.join(WORKSPACE_DIR, token))
    return False


# ─────────────────────────────────────────────── the sweep

def test_analyze_dataset_classifies_columns():
    name = _load()
    out = analyze_dataset(name)
    assert "ANALYSIS OF" in out
    assert "'qty'" in out and "'revenue'" in out          # numbers
    assert "'region'" in out                              # category
    assert "'signup'" in out                              # date, not category
    assert "order_id" in out                              # spotted as an ID
    assert "SUGGESTED NEXT STEPS" in out
    _cleanup(name)


def test_analyze_dataset_ranks_against_a_target():
    name = _load()
    out = analyze_dataset(name, target="revenue")
    assert "WHAT MOVES WITH 'revenue'" in out
    assert "qty" in out
    _cleanup(name)


def test_analyze_dataset_flags_blanks_and_duplicates():
    df = pd.DataFrame({"a": [1.0, 1.0, None, None, None], "b": ["x", "x", "y", "z", "w"]})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    name = _load("an_dirty", df)
    out = analyze_dataset(name)
    assert "empty" in out and "duplicate" in out
    _cleanup(name)


def test_unknown_dataset_is_an_error_string_not_a_crash():
    for result in (analyze_dataset("nope"),
                   plot_column("nope", "x"),
                   analyze_correlations("nope"),
                   compare_groups("nope", "a", "b")):
        assert result.startswith("Error:") and "nope" in result


# ─────────────────────────────────────────────── one column

def test_plot_column_number_saves_chart_and_describes_shape():
    name = _load()
    out = plot_column(name, "revenue")
    assert _chart_exists(out)
    assert "Smallest" in out and "typical (middle)" in out and "Average" in out
    _cleanup(name)


def test_plot_column_category_reports_shares():
    name = _load()
    out = plot_column(name, "region")
    assert _chart_exists(out)
    assert "distinct value" in out
    assert "north" in out and "%" in out
    _cleanup(name)


def test_plot_column_flags_an_id_column():
    name = _load()
    out = plot_column(name, "order_id")
    assert "looks like an ID" in out
    _cleanup(name)


def test_plot_column_reports_extreme_values():
    df = pd.DataFrame({"revenue": [500.0 + i for i in range(40)] + [99999.0]})
    name = _load("an_out", df)
    out = plot_column(name, "revenue")
    assert "outside the usual range" in out
    _cleanup(name)


def test_plot_column_handles_an_empty_column():
    name = _load("an_empty", pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]}))
    out = plot_column(name, "a")
    assert "entirely empty" in out
    _cleanup(name)


def test_plot_column_unknown_column_is_an_error():
    name = _load()
    assert plot_column(name, "nope").startswith("Error:")
    _cleanup(name)


# ─────────────────────────────────────────────── two columns

def test_plot_relationship_numbers_reports_correlation():
    name = _load()
    out = plot_relationship(name, "qty", "revenue")
    assert _chart_exists(out)
    assert "Correlation" in out
    assert "not proof that one causes the other" in out
    _cleanup(name)


def test_plot_relationship_category_reports_group_averages():
    name = _load()
    out = plot_relationship(name, "region", "revenue")
    assert _chart_exists(out)
    assert "north" in out and "average" in out
    assert "compare_groups" in out            # points at the significance test
    _cleanup(name)


def test_plot_relationship_date_reports_direction():
    name = _load()
    out = plot_relationship(name, "signup", "revenue")
    assert _chart_exists(out)
    assert any(word in out for word in ("rising", "falling", "flat"))
    _cleanup(name)


# ─────────────────────────────────────────────── correlations

def test_analyze_correlations_ranks_pairs():
    name = _load()
    out = analyze_correlations(name)
    assert _chart_exists(out)
    assert "qty" in out and "revenue" in out
    assert any(w in out for w in ("strong", "moderate", "weak", "none"))
    _cleanup(name)


def test_plot_column_warns_that_imputed_values_are_invented():
    """A column 30% filled with the median shows a huge fake spike. Describing
    that spike as a pattern is the single easiest way to mislead a reader."""
    df = pd.DataFrame({"RunTime": [60.0] * 30 + list(range(20, 120, 5))})
    df["RunTime_was_missing"] = [1] * 30 + [0] * 20
    name = _load("an_imp", df)
    out = plot_column(name, "RunTime")
    assert "WARNING" in out and "filled in by cleaning" in out
    assert "60" in out                                  # says what they were set to
    _cleanup(name)


def test_plot_column_warns_about_capped_values():
    df = pd.DataFrame({"revenue": [500.0 + i for i in range(40)]})
    df["revenue_was_capped"] = [1] * 4 + [0] * 36
    name = _load("an_cap", df)
    out = plot_column(name, "revenue")
    assert "rewritten to the edge" in out
    _cleanup(name)


def test_box_plot_ignores_groups_too_small_to_rank():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "genre": ["drama"] * 20 + ["comedy"] * 20 + [f"rare {i}" for i in range(10)],
        "rating": np.concatenate([rng.normal(7, .5, 40), rng.normal(9.5, .1, 10)]),
    })
    name = _load("an_small", df)
    out = plot_relationship(name, "genre", "rating")
    assert "Left out 10 group(s) with fewer than 5 rows" in out
    assert "rare 0" not in out                # the 9.5-rated singletons never top the list
    _cleanup(name)


def test_box_plot_refuses_when_no_group_is_big_enough():
    df = pd.DataFrame({"g": [f"g{i}" for i in range(20)],
                       "v": np.random.default_rng(6).normal(5, 1, 20)})
    name = _load("an_tiny", df)
    out = plot_relationship(name, "g", "v")
    assert "none with at least" in out and "pivot_dataset" in out
    _cleanup(name)


def test_chart_labels_survive_embedded_newlines():
    from agent.data_analysis import _clean_label
    messy = "Director:\n   Oliver Driver\n|\n    Stars:\nRorrie D. Travis,\nJasmeet Baduwalia"
    label = _clean_label(messy)
    assert "\n" not in label and len(label) <= 42
    assert _clean_label("   ") == "(blank)"
    df = pd.DataFrame({"stars": [messy] * 6 + ["Solo Actor"] * 4})
    name = _load("an_multiline", df)
    out = plot_column(name, "stars")
    # the label is flattened onto one line, not spilled across several
    assert "director: oliver driver | stars:" in out.lower()
    assert not any(line.strip().startswith(("Rorrie", "Jasmeet")) for line in out.split("\n"))
    _cleanup(name)


def test_run_python_warns_when_reading_a_file_that_shadows_a_dataset():
    from agent.runtime.tools import _stale_registry_read_warning
    name = _load("an_shadow")
    warning = _stale_registry_read_warning("d = pd.read_csv('an_shadow.csv')")
    assert "STALE DATA WARNING" in warning and "an_shadow" in warning
    assert _stale_registry_read_warning("d = pd.read_csv('something_else.csv')") == ""
    _cleanup(name)


def test_run_python_warns_when_rereading_the_raw_source_after_cleaning():
    """'movies 2.csv' does not collide with the dataset name 'movies', but
    re-reading it throws away every fix that cleaning applied."""
    from agent.runtime.tools import _stale_registry_read_warning
    pipe = get_data_pipeline()
    pipe.datasets["an_raw"] = _sales_df()
    pipe.sources["an_raw"] = "csv:an source file.csv"      # note the connector prefix
    assert _stale_registry_read_warning("pd.read_csv('an source file.csv')") == ""
    pipe.datasets["an_raw_clean"] = _sales_df().head(10)   # now a derived version exists
    warning = _stale_registry_read_warning("pd.read_csv('an source file.csv')")
    assert "RAW file behind dataset 'an_raw'" in warning
    assert "an_raw_clean" in warning
    pipe.sources.pop("an_raw", None)
    _cleanup("an_raw", "an_raw_clean")


def test_cleaning_markers_are_kept_out_of_correlations():
    """'x_was_capped' correlates with 'x' only because capping changed those
    rows — reporting that as a finding is a tautology."""
    df = _sales_df()
    df["revenue_was_capped"] = (df["revenue"] > df["revenue"].median()).astype(int)
    df["qty_was_missing"] = (df["qty"] > df["qty"].median()).astype(int)
    name = _load("an_marked", df)
    out = analyze_correlations(name, target="revenue")
    assert "Left out 2 cleaning-marker column" in out
    assert "revenue_was_capped: " not in out          # never ranked as a finding
    _cleanup(name)


def test_analyze_dataset_separates_cleaning_markers_from_real_columns():
    df = _sales_df()
    df["revenue_was_capped"] = 0
    name = _load("an_marked2", df)
    out = analyze_dataset(name, target="revenue")
    assert "cleaning-marker column(s)" in out
    header = out.split("WORTH KNOWING")[0]
    assert "2 number column(s)" in header             # qty + revenue, not the marker
    _cleanup(name)


def test_next_steps_never_suggest_charting_a_huge_category():
    df = pd.DataFrame({
        # 100 distinct titles over 300 rows: a real category, but far too many
        # to chart. (300-of-300 would be classed as an ID instead.)
        "title": [f"movie {i % 100}" for i in range(300)],
        "rating": np.random.default_rng(2).normal(7, 1, 300),
    })
    name = _load("an_wide", df)
    out = analyze_dataset(name)
    assert "plot_relationship" not in out.split("SUGGESTED NEXT STEPS")[-1]
    assert "summarise them first" in out
    _cleanup(name)


def test_analyze_correlations_needs_two_number_columns():
    name = _load("an_thin", pd.DataFrame({"only": [1.0, 2.0, 3.0], "text": list("abc")}))
    out = analyze_correlations(name)
    assert "at least 2 are needed" in out
    _cleanup(name)


# ─────────────────────────────────────────────── pivot

def test_pivot_dataset_builds_a_grid_and_registers_it():
    name = _load()
    out = pivot_dataset(name, rows=["region"], columns=["customer"], values="qty", how="sum")
    assert "registered as 'an_test_pivot'" in out
    result = get_data_pipeline().datasets["an_test_pivot"]
    assert len(result) == 2                        # north / south
    _cleanup(name, "an_test_pivot")


def test_pivot_percent_sums_to_one_hundred():
    name = _load()
    pivot_dataset(name, rows=["region"], values="qty", how="sum", percent=True,
                  output_name="an_pct")
    result = get_data_pipeline().datasets["an_pct"]
    assert abs(result.iloc[:, 1].sum() - 100) < 0.5
    _cleanup(name, "an_pct")


def test_pivot_unknown_column_is_an_error():
    name = _load()
    assert pivot_dataset(name, rows=["nope"]).startswith("Error:")
    _cleanup(name)


# ─────────────────────────────────────────────── over time

def test_analyze_over_time_reports_change_per_period():
    name = _load()
    out = analyze_over_time(name, "signup", freq="QE", value_col="qty", how="sum")
    assert _chart_exists(out)
    assert "change_pct" in out
    result = get_data_pipeline().datasets["an_test_over_time"]
    assert len(result) > 1 and "change_pct" in result.columns
    _cleanup(name, "an_test_over_time")


def test_analyze_over_time_rejects_a_non_date_column():
    name = _load()
    out = analyze_over_time(name, "region")
    assert "does not look like dates" in out
    _cleanup(name)


# ─────────────────────────────────────────────── group comparison

def test_compare_groups_detects_a_real_difference():
    name = _load()                                  # north is built 300 higher
    out = compare_groups(name, "revenue", "region")
    assert "Unlikely to be chance" in out
    assert "north" in out and "south" in out
    _cleanup(name)


def test_compare_groups_calls_noise_noise():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"g": ["a", "b"] * 60, "v": rng.normal(100, 15, 120)})
    name = _load("an_noise", df)
    out = compare_groups(name, "v", "g")
    assert "Could easily be chance" in out
    _cleanup(name)


def test_compare_groups_always_answers_both_questions():
    """Real and big are separate questions, and only reporting the first
    licenses calling a meaningless difference an outperformance."""
    name = _load()
    out = compare_groups(name, "revenue", "region")
    assert "IS IT REAL?" in out, out
    assert "IS IT BIG?" in out, out
    assert "Cohen's d" in out, out
    _cleanup(name)


def test_compare_groups_needs_a_number_column():
    name = _load()
    assert compare_groups(name, "region", "customer").startswith("Error:")
    _cleanup(name)


# ─────────────────────────────────────────────── contracts

def test_analysis_never_modifies_the_source_data():
    name = _load()
    before = get_data_pipeline().datasets[name].copy()
    plot_column(name, "revenue")
    plot_relationship(name, "qty", "revenue")
    analyze_correlations(name, target="revenue")
    analyze_dataset(name, target="revenue")
    compare_groups(name, "revenue", "region")
    pivot_dataset(name, rows=["region"], values="qty")
    analyze_over_time(name, "signup", freq="QE")
    pd.testing.assert_frame_equal(get_data_pipeline().datasets[name], before)
    _cleanup(name, "an_test_pivot", "an_test_over_time")


def test_big_frames_say_they_were_sampled():
    from agent.data_analysis import PLOT_SAMPLE_LIMIT
    n = PLOT_SAMPLE_LIMIT + 500
    df = pd.DataFrame({"x": np.random.default_rng(1).normal(0, 1, n)})
    name = _load("an_big", df)
    out = plot_column(name, "x")
    assert "sampled" in out                     # never silently truncated
    assert "use all rows" in out                # and the numbers still cover everything
    _cleanup(name)


def test_group_dataset_rejects_a_typo_instead_of_ignoring_it():
    from agent.data_cleaner import DataAnalyzer
    df = pd.DataFrame({"r": ["n", "s", "n"], "sales": [1, 2, 3]})
    try:
        DataAnalyzer(df).group(["r"], {"revenue": "sum"})     # typo'd column
    except ValueError as e:
        assert "not in data" in str(e)
    else:
        raise AssertionError("a typo'd aggregation column must not be silently ignored")


def test_no_stale_warning_after_save_dataset_writes_the_file():
    """Warning on a file the agent just saved from the registry is crying wolf —
    and trains the model to ignore the warning that matters."""
    from agent.runtime.tools import _stale_registry_read_warning
    import tempfile, os as _os
    pipe = get_data_pipeline()
    pipe.datasets["an_sync"] = _sales_df()
    code = "pd.read_csv('an_sync.csv')"
    assert _stale_registry_read_warning(code), "shadowing an unsaved dataset must warn"
    with tempfile.TemporaryDirectory() as tmp:
        pipe.saved["an_sync"] = (_os.path.join(tmp, "an_sync.csv"), pipe.datasets["an_sync"].shape)
        assert _stale_registry_read_warning(code) == ""
        # the dataset changes after saving → the file is stale again
        pipe.datasets["an_sync"] = _sales_df().head(5)
        assert _stale_registry_read_warning(code)
    pipe.saved.pop("an_sync", None)
    _cleanup("an_sync")


# ─────────────────────────────────────────────── multi-valued columns

def _genre_df() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    combos = ["Action, Adventure", "Action, Comedy", "Drama", "Drama, Romance",
              "Animation, Comedy, Family", "Horror, Thriller", "Documentary"]
    rows = rng.choice(len(combos), 140)
    base = {"Action": 6.5, "Adventure": 7.2, "Comedy": 6.8, "Drama": 7.1,
            "Romance": 7.0, "Animation": 8.0, "Family": 7.9, "Horror": 5.5,
            "Thriller": 6.0, "Documentary": 7.4}
    genre = [combos[i] for i in rows]
    rating = [np.mean([base[g.strip()] for g in c.split(",")]) + rng.normal(0, .2)
              for c in genre]
    return pd.DataFrame({"GENRE": genre, "RATING": rating})


def test_multivalue_column_is_detected_not_treated_as_atomic():
    from agent.data_analysis import _detect_multivalue
    df = _genre_df()
    sep, combos, singles = _detect_multivalue(df["GENRE"])
    assert sep == ","
    assert (combos, singles) == (7, 10)          # 7 combinations built from 10 real values
    # a normal category column must NOT be flagged
    assert _detect_multivalue(pd.Series(["north", "south"] * 30))[0] is None


def test_analyze_multivalue_splits_and_ranks_individual_values():
    name = _load("an_mv", _genre_df())
    out = analyze_multivalue(name, "GENRE", value_col="RATING")
    assert _chart_exists(out)
    assert "10 distinct value(s)" in out
    assert "animation" in out.lower()                 # built highest, should rank top
    assert "horror" in out.lower()                    # built lowest
    assert "counted once per value" in out            # the double-counting caveat
    assert "an_mv_GENRE_split" in get_data_pipeline().datasets
    _cleanup(name, "an_mv_GENRE_split")


def test_analyze_multivalue_counts_without_a_value_column():
    name = _load("an_mv2", _genre_df())
    out = analyze_multivalue(name, "GENRE")
    assert "Most common" in out and "%" in out
    _cleanup(name, "an_mv2_GENRE_split")


def test_plot_column_redirects_a_multivalue_column():
    name = _load("an_mv3", _genre_df())
    out = plot_column(name, "GENRE")
    assert "These are LISTS, not single categories" in out
    assert "analyze_multivalue" in out
    _cleanup(name)


def test_analyze_dataset_flags_multivalue_columns_and_suggests_the_tool():
    name = _load("an_mv4", _genre_df())
    out = analyze_dataset(name)
    assert "holds LISTS" in out
    assert "analyze_multivalue" in out.split("SUGGESTED NEXT STEPS")[-1]
    _cleanup(name)


# ─────────────────────────────────────────────── heavy tails

def test_plot_column_warns_about_a_long_tail_and_offers_a_log_view():
    rng = np.random.default_rng(17)
    df = pd.DataFrame({"votes": (10 ** rng.normal(3, 1.4, 400)).round() + 1})
    name = _load("an_tail", df)
    out = plot_column(name, "votes")
    assert "naturally long tail" in out and "log_scale=true" in out
    assert "do NOT cap it" in out
    logged = plot_column(name, "votes", log_scale=True)
    assert "_log_distribution.png" in logged
    assert "naturally long tail" not in logged        # the advice was taken
    _cleanup(name)


# ─────────────────────────────────────────────── missing-value patterns

def _blocked_missing_df() -> pd.DataFrame:
    """40 rows where 'rating' and 'votes' are blank TOGETHER (unreleased titles),
    plus scattered blanks in 'runtime' that are unrelated."""
    rng = np.random.default_rng(21)
    n = 200
    df = pd.DataFrame({
        "title": [f"film {i}" for i in range(n)],
        "rating": rng.normal(7, 1, n),
        "votes": rng.integers(50, 5000, n).astype(float),
        "runtime": rng.integers(70, 140, n).astype(float),
    })
    df.loc[:39, ["rating", "votes"]] = None            # one structural block
    df.loc[rng.choice(range(40, n), 20, replace=False), "runtime"] = None
    return df


def test_analyze_missing_finds_columns_that_go_blank_together():
    name = _load("an_miss", _blocked_missing_df())
    out = analyze_missing(name)
    assert _chart_exists(out)
    assert "go blank in the SAME rows" in out
    assert "'rating'" in out and "'votes'" in out
    assert "40 row(s) missing ALL of them" in out
    assert "invents a whole record" in out
    _cleanup(name)


def test_analyze_missing_says_so_when_gaps_are_unrelated():
    rng = np.random.default_rng(22)
    df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    df.loc[rng.choice(200, 30, replace=False), "a"] = None
    df.loc[rng.choice(200, 30, replace=False), "b"] = None
    name = _load("an_miss2", df)
    out = analyze_missing(name)
    assert "no column group goes missing together" in out
    _cleanup(name)


def test_analyze_dataset_surfaces_the_missing_block():
    name = _load("an_miss3", _blocked_missing_df())
    out = analyze_dataset(name)
    assert "go blank in the SAME" in out
    assert "analyze_missing" in out
    _cleanup(name)


def test_cleaner_warns_before_imputing_a_co_missing_block():
    from agent.data_cleaner import DataCleaner
    c = DataCleaner(_blocked_missing_df(), source="t")
    c.diagnose()
    imputes = [o for o in c.ops.values() if o["action"] == "impute"]
    blocked = [o for o in imputes if "same" in o["label"].lower()]
    assert blocked, "imputing a co-missing column must carry a warning"
    assert "invents a whole record" in blocked[0]["label"]


# ─────────────────────────────────────────────── subgroup checks

def _reversing_df() -> pd.DataFrame:
    """x~y is positive in group A and negative in group B — pooling them hides
    both. This is the case a single correlation gets wrong."""
    rng = np.random.default_rng(23)
    a = pd.DataFrame({"grp": "A", "x": rng.normal(10, 2, 150)})
    a["y"] = a.x * 1.5 + rng.normal(0, 1, 150)
    b = pd.DataFrame({"grp": "B", "x": rng.normal(20, 2, 150)})
    b["y"] = 40 - b.x * 1.5 + rng.normal(0, 1, 150)
    return pd.concat([a, b], ignore_index=True)


def test_check_subgroups_flags_a_reversal():
    name = _load("an_rev", _reversing_df())
    out = check_subgroups(name, "x", "y", by="grp")
    assert _chart_exists(out)
    assert "REVERSES" in out
    assert "opposite direction" in out
    _cleanup(name)


def test_check_subgroups_confirms_a_consistent_relationship():
    rng = np.random.default_rng(24)
    df = pd.DataFrame({"grp": ["A", "B"] * 150, "x": rng.normal(10, 3, 300)})
    df["y"] = df.x * 2 + rng.normal(0, 1, 300)          # same slope in both groups
    name = _load("an_cons", df)
    out = check_subgroups(name, "x", "y", by="grp")
    assert "consistent across groups" in out
    assert "REVERSES" not in out
    _cleanup(name)


def test_check_subgroups_handles_a_list_valued_grouping_column():
    df = _genre_df()
    df["RunTime"] = np.random.default_rng(25).normal(100, 20, len(df))
    name = _load("an_mvsub", df)
    out = check_subgroups(name, "RunTime", "RATING", by="GENRE", min_rows=10)
    assert "holds lists" in out and "FIRST value" in out
    _cleanup(name)


def test_check_subgroups_refuses_when_groups_are_too_small():
    df = pd.DataFrame({"grp": [f"g{i}" for i in range(40)],
                       "x": range(40), "y": range(40)})
    name = _load("an_tinysub", df)
    out = check_subgroups(name, "x", "y", by="grp")
    assert "cannot be compared across groups" in out
    _cleanup(name)


def test_scatter_points_at_the_subgroup_check():
    df = _reversing_df()
    name = _load("an_hint", df)
    out = plot_relationship(name, "x", "y")
    assert "check_subgroups" in out
    _cleanup(name)


# ─────────────────────────────────────────────── guards against nonsense output

def test_over_time_refuses_a_numeric_year_column():
    """pandas reads numbers as nanoseconds since 1970, so a YEAR column of
    2020.0 silently becomes 1970 and every 'trend' over it is fiction."""
    df = pd.DataFrame({"YEAR": [2018.0, 2019.0, 2020.0, 2021.0] * 10,
                       "rating": np.random.default_rng(31).normal(7, 1, 40)})
    name = _load("an_yearnum", df)
    out = analyze_over_time(name, "YEAR", value_col="rating")
    assert out.startswith("Error:")
    assert "nanoseconds since 1970" in out
    assert "group_dataset" in out                  # points at the right tool
    _cleanup(name)


def test_over_time_still_accepts_real_dates():
    name = _load()
    out = analyze_over_time(name, "signup", freq="QE", value_col="qty")
    assert not out.startswith("Error:")
    _cleanup(name, "an_test_over_time")


def test_pivot_refuses_a_grid_too_wide_to_read():
    from agent.data_analysis import PIVOT_MAX_LEVELS
    n = PIVOT_MAX_LEVELS * 3
    df = pd.DataFrame({"row": ["a", "b"] * n,
                       "many": [f"v{i}" for i in range(2 * n)],
                       "value": np.random.default_rng(32).normal(0, 1, 2 * n)})
    name = _load("an_wide2", df)
    out = pivot_dataset(name, rows=["row"], columns=["many"], values="value")
    assert out.startswith("Error:") and "too many to put on the columns" in out
    assert "group_dataset" in out
    # the same pivot the other way round is fine
    assert not pivot_dataset(name, rows=["row"], values="value").startswith("Error:")
    _cleanup(name, "an_wide2_pivot")


def test_multivalue_refuses_a_column_of_names():
    """21,000 actor names is not a tag vocabulary; ranking them by an average
    over 5 rows each is confident nonsense."""
    from agent.data_analysis import MULTIVALUE_HARD_LIMIT
    rng = np.random.default_rng(33)
    n = MULTIVALUE_HARD_LIMIT + 500
    cast = [f"actor {i}, actor {i + 1}, actor {i + 2}" for i in range(n)]
    df = pd.DataFrame({"stars": cast, "rating": rng.normal(7, 1, n)})
    name = _load("an_names", df)
    out = analyze_multivalue(name, "stars", value_col="rating")
    assert out.startswith("Error:")
    assert "not a tag vocabulary" in out
    _cleanup(name)


# ─────────────────────────────────────────────── conclusion review

def test_all_does_not_sweep_in_operations_the_plan_advises_against():
    """An op labelled NOT RECOMMENDED cannot also be part of 'approve everything'."""
    from agent.data_cleaner import DataCleaner, parse_approval
    rng = np.random.default_rng(41)
    n = 300
    df = pd.DataFrame({
        "votes": (10 ** rng.normal(3, 1.4, n)).round() + 1,       # heavy tail
        "rating": rng.normal(7, 1, n),
        "keep": rng.normal(0, 1, n),
    })
    df.loc[:59, ["rating", "keep"]] = None                        # co-missing block
    c = DataCleaner(df, source="t")
    c.diagnose()
    optional = c.optional_op_ids()
    assert optional, "heavy tail / co-missing ops must be marked not-recommended"
    approved, _ = parse_approval("all", c.op_ids(), optional)
    assert not (set(approved) & set(optional))
    # naming one explicitly still approves it
    named, _ = parse_approval(f"all {optional[0]}", c.op_ids(), optional)
    assert optional[0] in named


def test_review_flags_a_trend_claim_the_data_contradicts():
    from agent.data_analysis import reset_evidence, note_correlation, review_conclusions
    reset_evidence()
    note_correlation("YEAR", "RATING", -0.02)
    issues = review_conclusions("- Ratings: Declining from ~8.0 (1960s) to ~6.9 (2020s)")
    assert issues and "CONTRADICTION" in issues[0]
    assert "RATING" in issues[0] and "-0.02" in issues[0]


def test_review_flags_a_trend_built_on_invented_values():
    from agent.data_analysis import reset_evidence, note_imputed_groups, review_conclusions
    reset_evidence()
    note_imputed_groups("VOTES", ["2022", "2023"])
    issues = review_conclusions("- Votes: declining sharply to ~789 in 2022-2023")
    assert issues and "INVENTED VALUES" in issues[0]
    assert "2022" in issues[0]


def test_review_does_not_flag_correctly_reporting_no_trend():
    """False alarms are how a checker gets ignored."""
    from agent.data_analysis import reset_evidence, note_correlation, note_imputed_groups, review_conclusions
    reset_evidence()
    note_correlation("YEAR", "RATING", -0.02)
    note_imputed_groups("RATING", ["2022"])
    for honest in ("No trend was found in ratings over time (correlation -0.02, none).",
                   "Ratings are essentially flat across years.",
                   "RATING does not change over time."):
        assert review_conclusions(honest) == [], honest


def test_review_stays_quiet_on_real_relationships_and_plain_facts():
    from agent.data_analysis import reset_evidence, note_correlation, review_conclusions
    reset_evidence()
    note_correlation("RunTime", "RATING", -0.31)          # a genuine relationship
    assert review_conclusions("- Longer titles are rated lower; RunTime declining with RATING") == []
    assert review_conclusions("- Drama is 43% of all titles.") == []


def test_agent_loop_review_hook_uses_the_ledger():
    from agent.core.agent_loop import _review_summary
    from agent.data_analysis import reset_evidence, note_correlation
    reset_evidence()
    note_correlation("YEAR", "VOTES", 0.01)
    assert _review_summary("- Votes are rising every year") 
    assert _review_summary("") == []


# ─────────────────────────────────────────────── rankings

def _orders_df() -> pd.DataFrame:
    """Two products separated by 1 unit of revenue — indistinguishable as bars,
    which is exactly the case a ranking read off a chart gets wrong."""
    return pd.DataFrame({
        "product": (["Alpha"] * 3 + ["Bravo"] * 3 + ["Charlie"] * 3 + ["Delta"] * 3
                    + ["Echo"] * 3 + ["Foxtrot"] * 3),
        "customer": [f"C{i % 5:02d}" for i in range(18)],
        "revenue": ([50.0] * 3 + [40.0] * 3 + [30.0] * 3 + [20.0] * 3
                    + [10.0] * 3 + [9.8] * 3),
        "ordered": ["2024-01-05"] * 18,
        "delivered": ["08-01-2024"] * 18,          # day-first, unlike 'ordered'
    })


def _load_orders(name="rank_test"):
    get_data_pipeline().datasets[name] = _orders_df()
    return name


def test_rank_by_returns_a_numbered_order_with_shares():
    from agent.data_analysis import rank_by, reset_evidence
    reset_evidence()
    name = _load_orders()
    out = rank_by(name, "product", "revenue", top=3)
    assert "1. Alpha" in out and "2. Bravo" in out and "3. Charlie" in out
    assert "% of total" in out and "cumulative" in out
    assert "Total across all groups" in out
    get_data_pipeline().datasets.pop(name, None)


def test_rank_by_names_the_bottom_of_the_table_too():
    """A dimension is only described once both ends are named."""
    from agent.data_analysis import rank_by, reset_evidence
    reset_evidence()
    name = _load_orders()
    out = rank_by(name, "product", "revenue", top=3)
    assert "Bottom of the table" in out and "Foxtrot" in out
    get_data_pipeline().datasets.pop(name, None)


def test_rank_by_warns_when_the_cutoff_is_a_near_tie():
    """The caution must carry the SIZE of the gap, not the threshold that found it.

    Reporting only 'within 5%' gets quoted straight back as the finding — a real
    report described two products differing by 0.009% as 'tied within 5%', which
    overstates the gap by a factor of 500.
    """
    from agent.data_analysis import rank_by, reset_evidence
    reset_evidence()
    name = _load_orders()
    out = rank_by(name, "product", "revenue", top=5)
    assert "CAUTION" in out
    assert "effectively a tie" in out
    assert re.search(r"differ by (\d+\.\d+%|under 0\.01%)", out), out
    get_data_pipeline().datasets.pop(name, None)


def test_rank_by_warns_when_a_name_covers_two_different_ids():
    """Two products sharing a name must not be summed into one league-table row.

    From a real run: 70 Product_IDs, 68 Product_Names. Ranking by name fused two
    distinct products into a 'best-seller' that does not exist, and pushed a
    genuine top-5 product out of the table entirely.
    """
    from agent.data_analysis import rank_by, reset_evidence
    reset_evidence()
    df = pd.DataFrame({
        "product_id": [1, 2, 3, 4],
        "product_name": ["Quia Gift", "Quia Gift", "Solo", "Other"],
        "revenue": [60.0, 55.0, 100.0, 10.0],
    })
    get_data_pipeline().datasets["collide"] = df
    out = rank_by("collide", "product_name", "revenue", top=3)
    assert "RANKING MERGES DISTINCT ENTITIES" in out
    assert "Quia Gift" in out and "product_id" in out
    get_data_pipeline().datasets.pop("collide", None)


def test_rank_by_stays_quiet_on_a_deliberate_many_to_one_grouping():
    """Grouping orders by city obviously merges order ids — that is the point."""
    from agent.data_analysis import rank_by, reset_evidence
    reset_evidence()
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "city": ["Pune", "Pune", "Delhi", "Delhi"],
        "revenue": [10.0, 20.0, 30.0, 40.0],
    })
    get_data_pipeline().datasets["cities"] = df
    out = rank_by("cities", "city", "revenue", top=2)
    assert "MERGES DISTINCT ENTITIES" not in out
    get_data_pipeline().datasets.pop("cities", None)


def test_day_first_dates_are_read_with_one_convention_for_the_whole_column():
    """A DD-MM-YYYY column must not be re-decided row by row.

    Under per-row parsing '24-02-2023' reads as 24 February (no month is 24) but
    '07-11-2023' reads as 11 July — half the rows land in the wrong month while
    every row parses without complaint. A real monthly revenue table was wrong in
    all 12 periods because of this.
    """
    from agent.data_analysis import _as_datetime
    dates = pd.Series(["24-02-2023", "07-11-2023", "05-03-2023"], name="Order_Date")
    note = []
    parsed = _as_datetime(dates, note=note)
    assert list(parsed.dt.month) == [2, 11, 3]
    assert list(parsed.dt.day) == [24, 7, 5]
    assert any("day-first" in n for n in note), note


def test_an_unprovable_date_layout_is_flagged_rather_than_assumed():
    """Every value ambiguous means the column cannot prove its own convention."""
    from agent.data_analysis import _as_datetime
    note = []
    _as_datetime(pd.Series(["05-03-2023", "07-11-2023"], name="d"), note=note)
    assert any("CAUTION" in n and "ambiguous" in n for n in note), note


def test_the_trend_report_states_how_it_read_the_dates():
    """The convention decides every number in the table, so it ships with it."""
    from agent.data_analysis import analyze_over_time, reset_evidence
    reset_evidence()
    df = pd.DataFrame({
        "Order_Date": ["24-02-2023", "25-02-2023", "07-11-2023", "08-11-2023"],
        "Revenue": [10.0, 20.0, 30.0, 40.0],
    })
    get_data_pipeline().datasets["trend"] = df
    out = analyze_over_time("trend", "Order_Date", freq="ME", value_col="Revenue")
    assert "day-first" in out
    # '07-11-2023' belongs in November, not July. Under per-row parsing the 30.0
    # and 40.0 land in a July bucket and November comes out empty.
    periods = get_data_pipeline().datasets["trend_over_time"].set_index("Order_Date")["value"]
    assert float(periods["2023-11-30"]) == 70.0
    assert float(periods["2023-07-31"]) == 0.0
    get_data_pipeline().datasets.pop("trend", None)


def test_rank_by_refuses_a_non_numeric_measure():
    from agent.data_analysis import rank_by, reset_evidence
    reset_evidence()
    name = _load_orders()
    assert rank_by(name, "product", "customer").startswith("Error:")
    get_data_pipeline().datasets.pop(name, None)


def test_report_is_refused_when_it_names_the_wrong_top_n():
    """The bug this guard exists for: a top-5 with the 2nd-placed row missing."""
    from agent.data_analysis import rank_by, reset_evidence, review_conclusions
    reset_evidence()
    name = _load_orders()
    rank_by(name, "product", "revenue", top=5)
    issues = review_conclusions(
        "Top 3 products by revenue are Alpha, Charlie and Delta.")
    assert issues and "WRONG RANKING" in issues[0]
    assert "Bravo" in issues[0]                       # the real #2 is named
    assert review_conclusions("Top 3 products by revenue are Alpha, Bravo and Charlie.") == []
    get_data_pipeline().datasets.pop(name, None)


def test_share_claims_are_checked_against_the_recorded_ranking():
    from agent.data_analysis import rank_by, reset_evidence, review_conclusions
    reset_evidence()
    name = _load_orders()
    rank_by(name, "product", "revenue", top=6)
    issues = review_conclusions("The top 50% of products drive 90% of revenue.")
    assert issues and "UNSUPPORTED SHARE" in issues[0]
    get_data_pipeline().datasets.pop(name, None)


def test_share_check_stays_quiet_on_percentages_that_are_not_concentration():
    """Two percentages and a column name is not a concentration claim; flagging
    it would be the guess that gets the whole checker ignored."""
    from agent.data_analysis import rank_by, reset_evidence, review_conclusions
    reset_evidence()
    name = _load_orders()
    rank_by(name, "product", "revenue", top=6)
    for benign in ("About 30% of products are seasonal and 70% are stocked year-round.",
                   "Roughly 25% of products are new; 60% of them launched in Q4."):
        assert review_conclusions(benign) == [], benign
    get_data_pipeline().datasets.pop(name, None)


# ─────────────────────────────────────────────── durations

def test_measure_duration_reports_average_median_and_range():
    from agent.data_analysis import measure_duration, reset_evidence
    reset_evidence()
    name = _load_orders()
    out = measure_duration(name, "ordered", "delivered")
    assert "average 3.00" in out and "median 3.00" in out
    get_data_pipeline().datasets.pop(name, None)


def test_measure_duration_picks_the_date_convention_that_is_possible():
    """'08-01-2024' read month-first is August 1st — five months of 'delivery'
    time and, for other rows, a delivery before its own order."""
    from agent.data_analysis import measure_duration, reset_evidence
    reset_evidence()
    name = _load_orders()
    out = measure_duration(name, "ordered", "delivered")
    assert "day-first" in out                          # the convention is stated
    assert "average 3.00" in out                       # not 209 days
    get_data_pipeline().datasets.pop(name, None)


def test_measure_duration_warns_about_impossible_gaps():
    from agent.data_analysis import measure_duration, reset_evidence
    reset_evidence()
    df = _orders_df()
    df.loc[0, "delivered"] = "01-01-2024"              # before the order date
    get_data_pipeline().datasets["dur_bad"] = df
    out = measure_duration("dur_bad", "ordered", "delivered")
    assert "WARNING" in out and "BEFORE" in out
    get_data_pipeline().datasets.pop("dur_bad", None)


def test_measure_duration_refuses_a_numeric_column():
    """pandas reads a number as nanoseconds since 1970 and puts every row there."""
    from agent.data_analysis import measure_duration, reset_evidence
    reset_evidence()
    df = _orders_df()
    df["year"] = 2024
    get_data_pipeline().datasets["dur_num"] = df
    assert measure_duration("dur_num", "year", "delivered").startswith("Error:")
    get_data_pipeline().datasets.pop("dur_num", None)


def test_a_stated_duration_that_contradicts_the_measurement_is_refused():
    from agent.data_analysis import measure_duration, reset_evidence, review_conclusions
    reset_evidence()
    name = _load_orders()
    measure_duration(name, "ordered", "delivered")
    issues = review_conclusions("Average time from ordered to delivered is 9.2 days.")
    assert issues and "WRONG DURATION" in issues[0]
    assert review_conclusions("Average time from ordered to delivered is 3 days.") == []
    get_data_pipeline().datasets.pop(name, None)


def test_new_analysis_tools_are_registered():
    from agent.runtime.tools import TOOL_REGISTRY
    for tool in ("rank_by", "measure_duration"):
        assert tool in TOOL_REGISTRY, tool


def _occasion_df(repeat: int = 10):
    """Two dimensions whose marginals point one way and whose cross-tab points
    the other — the exact shape that produced a backwards recommendation.

    Replicated with jitter so the effects clear a significance test: a fixture too
    small to distinguish from noise gets caught by the noise check instead, which
    is correct behaviour but tests the wrong thing here.
    """
    cells = [
        # Colors is the biggest category overall, driven entirely by Holi...
        ("Colors", "Holi", 300.0, 4),
        ("Colors", "Anniversary", 10.0, 2),
        # ...while Anniversary, the biggest occasion, is really a Sweets occasion.
        ("Sweets", "Anniversary", 400.0, 4),
        ("Sweets", "Holi", 20.0, 1),
        ("Plants", "Anniversary", 120.0, 2),
        ("Cake", "Anniversary", 90.0, 1),
        ("Mugs", "Holi", 60.0, 1),
    ]
    rng = np.random.default_rng(11)
    rows = []
    for category, occasion, revenue, n in cells:
        for _ in range(n * repeat):
            rows.append({"category": category, "occasion": occasion,
                         "revenue": float(revenue + rng.normal(0, revenue * 0.02))})
    return pd.DataFrame(rows)


def _rank_two_dimensions(name="pairing"):
    from agent.data_analysis import rank_by, reset_evidence
    reset_evidence()
    get_data_pipeline().datasets[name] = _occasion_df()
    rank_by(name, "category", "revenue", top=9)
    rank_by(name, "occasion", "revenue", top=9)
    return name


def test_pairing_the_top_of_two_rankings_is_checked_against_the_cross_tab():
    """The marginal fallacy: #1 category + #1 occasion is not a recommendation.

    A real report advised stocking Colors ahead of Anniversary because each led
    its own league table. Colors is the worst-selling category at Anniversary.
    """
    from agent.data_analysis import review_conclusions
    name = _rank_two_dimensions()
    issues = review_conclusions("Front-load inventory for Colors before Anniversary.")
    assert any("PAIRING NOT SUPPORTED" in i for i in issues), issues
    assert any("Colors" in i and "Anniversary" in i and "Sweets" in i for i in issues)
    get_data_pipeline().datasets.pop(name, None)


def test_a_pairing_the_cross_tab_supports_is_left_alone():
    from agent.data_analysis import review_conclusions
    name = _rank_two_dimensions()
    assert review_conclusions("Promote Sweets for Anniversary.") == []
    get_data_pipeline().datasets.pop(name, None)


def test_each_clause_is_paired_on_its_own():
    """Two recommendations in one bullet must not be cross-multiplied."""
    from agent.data_analysis import review_conclusions
    name = _rank_two_dimensions()
    issues = review_conclusions("Promote Sweets for Anniversary; push Colors for Holi.")
    assert issues == [], issues
    get_data_pipeline().datasets.pop(name, None)


def test_naming_two_dimensions_without_linking_them_is_not_a_pairing():
    """Reporting two league tables side by side claims no relationship."""
    from agent.data_analysis import review_conclusions
    name = _rank_two_dimensions()
    issues = review_conclusions("Colors is the top category and Anniversary the top occasion.")
    assert issues == [], issues
    get_data_pipeline().datasets.pop(name, None)


def test_a_comma_does_not_link_two_separate_recommendations():
    """'Colors ahead of Holi, and Sweets ahead of Anniversary' claims two pairings,
    not four — pairing across the comma blocks a correct report."""
    from agent.data_analysis import review_conclusions
    name = _rank_two_dimensions()
    issues = review_conclusions("Stock Colors ahead of Holi, and Sweets ahead of Anniversary.")
    assert issues == [], issues
    get_data_pipeline().datasets.pop(name, None)


def test_a_fronted_clause_still_pairs_across_its_comma():
    """'For Anniversary, stock Colors' is one claim split by a comma. Splitting on
    every comma would drop the only pairing in the sentence."""
    from agent.data_analysis import review_conclusions
    name = _rank_two_dimensions()
    issues = review_conclusions("For Anniversary, stock Colors.")
    assert any("PAIRING NOT SUPPORTED" in i for i in issues), issues
    get_data_pipeline().datasets.pop(name, None)


def _flat_df(n=480):
    """A dimension with no real signal — the shape that produced 'peak ordering
    hours are 19:00-21:00' from 24 groups that were flat at p = 0.45."""
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "hour": [str(i % 24) for i in range(n)],
        "revenue": rng.normal(100, 20, n),
    })


def test_rank_by_says_when_a_league_table_is_only_noise():
    from agent.data_analysis import rank_by, reset_evidence
    reset_evidence()
    get_data_pipeline().datasets["flat"] = _flat_df()
    out = rank_by("flat", "hour", "revenue", top=3)
    assert "NOISE CHECK" in out and "ranks noise" in out
    get_data_pipeline().datasets.pop("flat", None)


def test_a_winner_claimed_on_a_noise_ranking_is_refused():
    from agent.data_analysis import rank_by, reset_evidence, review_conclusions
    reset_evidence()
    get_data_pipeline().datasets["flat"] = _flat_df()
    rank_by("flat", "hour", "revenue", top=3)
    issues = review_conclusions("Peak ordering hours drive the most revenue.")
    assert any("RANKS NOISE" in i for i in issues), issues
    get_data_pipeline().datasets.pop("flat", None)


def test_correctly_reporting_a_null_result_is_not_refused():
    """Saying the groups are indistinguishable is the right call, not a claim."""
    from agent.data_analysis import rank_by, reset_evidence, review_conclusions
    reset_evidence()
    get_data_pipeline().datasets["flat"] = _flat_df()
    rank_by("flat", "hour", "revenue", top=3)
    assert review_conclusions(
        "The highest and lowest hours are indistinguishable from sampling noise.") == []
    get_data_pipeline().datasets.pop("flat", None)


def test_a_real_dimension_is_not_called_noise():
    from agent.data_analysis import rank_by, reset_evidence, review_conclusions
    name = _rank_two_dimensions()
    assert not any("RANKS NOISE" in i
                   for i in review_conclusions("Sweets is the top category by revenue."))
    get_data_pipeline().datasets.pop(name, None)


def test_run_python_refuses_to_analyse_a_stale_copy_of_a_loaded_dataset():
    """A warning above correct-looking output is not a stop sign — four were
    ignored in one real session and their figures reached the final report."""
    from agent.runtime.tools import run_python
    name = _load("an_stale")
    # Refused before the sandbox is touched, so this needs no container.
    out = run_python("d = pd.read_csv('an_stale.csv')\nprint(d.shape)")
    assert out.startswith("Error:") and "was NOT run" in out
    assert "an_stale" in out
    _cleanup(name)
