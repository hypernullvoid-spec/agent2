"""
Generality tests for the Tier-1 statistical fixes.

The point of every test here is that it must hold for ANY dataset, not for the
one the fix was written against. So each fix is exercised across several
unrelated domains (web logs, geography, clinical, retail, sensors, surveys) and
in several languages, and each is paired with a NEGATIVE case proving the fix
does not fire where it should stay quiet. A rule that only works on the file it
was debugged with is worse than no rule, because it reads as coverage.

Run via:  python tests/run_tests.py
"""

import numpy as np
import pandas as pd

from agent.data_analysis import (
    _looks_like_coded_categorical,
    _missingness_drivers,
    analyze_correlations,
    analyze_dataset,
    analyze_missing,
)
from agent.data_cleaner import DataCleaner, _looks_like_clock_time
from agent.ml.data_pipeline import get_data_pipeline


def _load(name: str, df: pd.DataFrame) -> str:
    get_data_pipeline().datasets[name] = df
    return name


def _cleanup(*names: str) -> None:
    pipe = get_data_pipeline()
    for n in names:
        pipe.datasets.pop(n, None)


def _clean(df: pd.DataFrame, source: str = "t") -> DataCleaner:
    c = DataCleaner(df.copy(), source=source)
    c.diagnose()
    c.apply_cleaning(c.op_ids())
    return c


# ─────────────────────────────── 1. date conventions ───────────────────────────
# The failure was per-VALUE parsing: '07-11-2023' read month-first while
# '14-07-2023' was read day-first, inside one column.


def test_one_date_convention_is_used_for_the_whole_column():
    # Ambiguous rows (day <= 12) must follow the convention the column proves.
    df = pd.DataFrame({"when": ["24-02-2023", "07-11-2023", "10-07-2023", "11-02-2023"] * 6})
    out = _clean(df).df["when"]
    assert list(out[:4]) == ["2023-02-24", "2023-11-07", "2023-07-10", "2023-02-11"]


def test_month_first_files_are_read_month_first():
    # Same rule, opposite convention — the fix must not hardcode day-first.
    df = pd.DataFrame({"when": ["02-24-2023", "11-07-2023", "07-10-2023", "12-31-2023"] * 6})
    out = _clean(df).df["when"]
    assert list(out[:4]) == ["2023-02-24", "2023-11-07", "2023-07-10", "2023-12-31"]


def test_iso_dates_are_left_correct():
    df = pd.DataFrame({"when": pd.date_range("2021-03-01", periods=24).astype(str)})
    out = _clean(df).df["when"]
    assert out.iloc[0] == "2021-03-01" and out.iloc[-1] == "2021-03-24"


def test_an_unprovable_column_is_still_internally_consistent():
    # Every value ambiguous (both parts <= 12), and all distinct so nothing is
    # removed as a duplicate. The layout cannot be proven, but ONE reading must
    # still be applied to every row — the original bug was mixing both.
    raw = ["01-02-2023", "03-04-2023", "05-06-2023", "02-01-2023", "04-03-2023",
           "06-05-2023", "07-08-2023", "08-07-2023", "09-10-2023", "10-09-2023",
           "11-12-2023", "12-11-2023"]
    out = _clean(pd.DataFrame({"when": raw})).df["when"]
    assert out.notna().all()                        # nothing silently dropped
    assert len(out) == len(raw)
    day_first = pd.to_datetime(pd.Series(raw), dayfirst=True).dt.strftime("%Y-%m-%d")
    month_first = pd.to_datetime(pd.Series(raw), dayfirst=False).dt.strftime("%Y-%m-%d")
    # the result must match ONE convention throughout, not a mixture of the two
    assert list(out) == list(day_first) or list(out) == list(month_first)


def test_an_assumed_convention_is_declared_not_applied_quietly():
    df = pd.DataFrame({"when": ["01-02-2023", "03-04-2023"] * 12})
    c = DataCleaner(df.copy(), source="t")
    c.diagnose()
    summary = c.apply_cleaning(c.op_ids())
    assert "ASSUMED" in summary


def test_a_proven_convention_says_it_was_proven():
    df = pd.DataFrame({"when": ["24-02-2023", "07-11-2023"] * 12})
    c = DataCleaner(df.copy(), source="t")
    c.diagnose()
    assert "proven by the column itself" in c.apply_cleaning(c.op_ids())


# ─────────────────────────────── 2. times of day ───────────────────────────────
# pd.to_datetime('23:48:13') succeeds by attaching today, so a clock column used
# to collapse to a single constant — the run date.


def test_clock_times_are_recognised_in_several_notations():
    for values in (["23:48:13", "07:16:17", "00:00:01"],
                   ["11:48 PM", "7:16 am", "12:00 PM"],
                   ["23:48", "07:16", "19:05"],
                   ["23:48:13.250", "07:16:17.001", "05:00:00.100"]):
        assert _looks_like_clock_time(pd.Series(values * 8)), values


def test_a_real_date_is_not_mistaken_for_a_clock():
    for values in (["2023-01-05", "2023-02-06"],
                   ["24-02-2023", "07-11-2023"],
                   ["2023-01-05 23:48:13", "2023-02-06 07:16:17"]):
        assert not _looks_like_clock_time(pd.Series(values * 8)), values


def test_a_time_of_day_column_survives_cleaning_intact():
    times = [f"{h:02d}:{m:02d}:{s:02d}" for h in range(8) for m in (5, 35) for s in (1, 44)]
    df = pd.DataFrame({"order_time": times})
    out = _clean(df).df["order_time"]
    assert out.nunique() == len(set(times))         # nothing collapsed
    assert set(out) == set(times)                   # nothing rewritten


def test_a_datetime_column_keeps_its_clock():
    stamps = [f"2023-01-{d:02d} {d:02d}:30:00" for d in range(1, 25)]
    out = _clean(pd.DataFrame({"seen_at": stamps})).df["seen_at"]
    assert out.iloc[0].endswith("01:30:00")         # time component preserved


def test_a_midnight_only_column_is_still_shortened_to_a_plain_date():
    stamps = [f"2023-01-{d:02d} 00:00:00" for d in range(1, 25)]
    assert _clean(pd.DataFrame({"d": stamps})).df["d"].iloc[0] == "2023-01-01"


# ──────────────────────── 3. numbers that are labels ──────────────────────────
# Judged from the value set, so the rule cannot depend on English column names.


def test_code_columns_are_detected_across_unrelated_domains():
    rng = np.random.default_rng(3)
    coded = {
        "http_status": [200, 301, 404, 500],
        "postcode": [110001, 400001, 560001, 700001],
        "store_no": [12, 47, 88, 301, 905],
        "icd_block": [250, 401, 715, 820],
        "port": [22, 80, 443, 8080],
    }
    for label, levels in coded.items():
        s = pd.Series(rng.choice(levels, 600))
        assert _looks_like_coded_categorical(s), label


def test_real_measurements_are_not_called_codes():
    rng = np.random.default_rng(4)
    measures = {
        "likert_1_5": rng.integers(1, 6, 400),
        "count_per_order": rng.integers(1, 20, 800),
        "year": rng.choice([2019, 2020, 2021, 2022, 2023], 500),
        "age": rng.integers(18, 90, 700),
        "binary_flag": rng.choice([0, 1], 500),
        "price": rng.lognormal(3, 1, 500),
        "temperature_c": rng.normal(21, 4, 600),
        "big_continuous_int": rng.integers(0, 10_000, 900),
    }
    for label, values in measures.items():
        assert not _looks_like_coded_categorical(pd.Series(values)), label


def test_code_detection_does_not_depend_on_the_column_name():
    rng = np.random.default_rng(5)
    values = rng.choice([200, 301, 404, 500], 600)
    for name in ("http_status", "応答コード", "статус", "col_7", "x"):
        assert _looks_like_coded_categorical(pd.Series(values, name=name)), name


def test_a_tiny_column_is_left_alone_rather_than_guessed():
    assert not _looks_like_coded_categorical(pd.Series([200, 404, 500]))


def test_codes_are_reported_separately_and_kept_out_of_correlations():
    rng = np.random.default_rng(6)
    name = _load("t1_codes", pd.DataFrame({
        "status": rng.choice([200, 404, 500], 600),
        "latency_ms": rng.lognormal(3, 1, 600),
        "cpu_pct": rng.normal(50, 8, 600),
    }))
    overview = analyze_dataset(name)
    assert "code/label column(s)" in overview and "status" in overview
    corr = analyze_correlations(name)
    assert "Left out" in corr and "status" in corr
    _cleanup(name)


# ──────────────────────────── 4. correlation evidence ─────────────────────────


def test_a_correlation_always_carries_its_sample_size():
    rng = np.random.default_rng(8)
    x = rng.normal(0, 1, 500)
    name = _load("t1_corr", pd.DataFrame({"x": x, "y": x * 2 + rng.normal(0, 1, 500)}))
    assert "n=500" in analyze_correlations(name)
    _cleanup(name)


def test_a_thin_correlation_is_labelled_unreliable():
    name = _load("t1_thin", pd.DataFrame({"x": [1, 2, 3, 4, 5, 6],
                                          "y": [2, 1, 4, 3, 6, 5.2]}))
    out = analyze_correlations(name)
    assert "TOO FEW ROWS" in out and "n=6" in out
    _cleanup(name)


def test_an_insignificant_correlation_says_so():
    rng = np.random.default_rng(9)
    name = _load("t1_ns", pd.DataFrame({"a": rng.normal(0, 1, 60),
                                        "b": rng.normal(0, 1, 60)}))
    assert "NOT significant" in analyze_correlations(name)
    _cleanup(name)


def test_a_strong_well_powered_correlation_is_not_undermined():
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, 800)
    name = _load("t1_ok", pd.DataFrame({"x": x, "y": x * 3 + rng.normal(0, 0.5, 800)}))
    out = analyze_correlations(name)
    assert "TOO FEW ROWS" not in out and "NOT significant" not in out
    _cleanup(name)


def test_sample_size_is_the_rows_both_columns_share():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 400)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(0, 1, 400)})
    df.loc[:149, "y"] = np.nan                  # only 250 usable pairs
    name = _load("t1_pair", df)
    assert "n=250" in analyze_correlations(name)
    _cleanup(name)


# ──────────────────────────── 5. missingness mechanism ────────────────────────


def test_missingness_driven_by_a_number_is_flagged():
    rng = np.random.default_rng(12)
    age = rng.normal(40, 12, 600)
    income = age * 900 + rng.normal(0, 4000, 600)
    income[age > 55] = np.nan
    name = _load("t1_mar", pd.DataFrame({"age": age, "income": income}))
    out = analyze_missing(name)
    assert "NOT missing at random" in out and "age" in out
    _cleanup(name)


def test_missingness_driven_by_a_category_is_flagged():
    rng = np.random.default_rng(13)
    region = rng.choice(["north", "south", "east"], 600)
    sales = np.where(region == "north", np.nan, rng.normal(100, 10, 600))
    name = _load("t1_mar2", pd.DataFrame({"region": region, "sales": sales}))
    out = analyze_missing(name)
    assert "NOT missing at random" in out and "region" in out
    _cleanup(name)


def test_truly_random_gaps_are_not_accused_of_a_pattern():
    # The false-positive guard: MCAR data must keep its clean bill of health.
    rng = np.random.default_rng(14)
    df = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(0, 1, 400),
                       "c": rng.choice(list("xyz"), 400)})
    df.loc[rng.choice(400, 60, replace=False), "a"] = np.nan
    name = _load("t1_mcar", df)
    out = analyze_missing(name)
    assert "NOT missing at random" not in out
    assert "no column group goes missing together" in out
    _cleanup(name)


def test_the_remedy_matches_the_kind_of_driver():
    rng = np.random.default_rng(15)
    # a categorical driver CAN be grouped on...
    region = rng.choice(["north", "south"], 500)
    sales = np.where(region == "north", np.nan, rng.normal(100, 10, 500))
    name = _load("t1_rem1", pd.DataFrame({"region": region, "sales": sales}))
    assert "by='region'" in analyze_missing(name)
    _cleanup(name)
    # ...a continuous one cannot — grouping on it gives one row per group.
    age = rng.normal(40, 12, 500)
    income = np.where(age > 55, np.nan, age * 900)
    name = _load("t1_rem2", pd.DataFrame({"age": age, "income": income}))
    out = analyze_missing(name)
    assert "by='age'" not in out and "bucket 'age'" in out
    _cleanup(name)


def test_the_scan_survives_degenerate_columns():
    rng = np.random.default_rng(16)
    df = pd.DataFrame({
        "const": ["same"] * 300,
        "allnull": [np.nan] * 300,
        "freetext": [f"unique sentence {i}" for i in range(300)],
        "val": rng.normal(0, 1, 300),
    })
    df.loc[rng.choice(300, 40, replace=False), "val"] = np.nan
    assert isinstance(_missingness_drivers(df, "val", df.isnull()), list)


# ──────────────────────────── 6. minority classes ─────────────────────────────


def test_a_rare_class_is_never_rounded_out_of_existence():
    for rare in (1, 5, 20):
        df = pd.DataFrame({"outcome": ["no"] * (2000 - rare) + ["yes"] * rare,
                           "v": np.arange(2000)})
        name = _load("t1_imb", df)
        out = analyze_dataset(name)
        assert "100%" not in out, rare              # never claims the minority away
        assert f"yes={rare:,}" in out, rare         # counts it explicitly
        _cleanup(name)


def test_a_genuine_constant_is_still_called_a_constant():
    name = _load("t1_const", pd.DataFrame({"only": ["same"] * 300,
                                           "v": np.arange(300.0)}))
    assert "only one value" in analyze_dataset(name)
    _cleanup(name)


def test_imbalance_wording_works_for_non_english_labels():
    df = pd.DataFrame({"結果": ["正常"] * 995 + ["異常"] * 5, "v": np.arange(1000)})
    name = _load("t1_i18n", df)
    out = analyze_dataset(name)
    assert "異常=5" in out and "100%" not in out
    _cleanup(name)
