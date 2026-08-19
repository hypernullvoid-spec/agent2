"""
Cross-dataset generality — the analysis layer must be right on data it has
never seen, not merely on the dataset its guards were written against.

Every guard here was originally tuned on one e-commerce table, and each one had
a defect that only appeared on a structurally different dataset:

  * 'origin'/'destination' share their whole vocabulary by design, and were
    reported as duplicate columns.
  * A legal corpus of genuine Latin was declared "not real" data.
  * A 20-column frame silently reported no derived columns because the search
    bailed above a column limit.
  * A large-magnitude finance table missed its own derivation because the
    tolerance was absolute rather than relative.

The datasets below deliberately share nothing with each other: different
domains, magnitudes, languages, shapes and time spans. A guard that needs to
know what the columns MEAN will fail at least one of them.

Two rules these encode:
  1. A wrong assertion is worse than silence. Where the data is ambiguous the
     right output is nothing, not a confident guess.
  2. Silence must not be indistinguishable from "nothing found". A search that
     gives up has to say so.
"""

import numpy as np
import pandas as pd

from agent.ml.data_pipeline import DataPipeline, get_data_pipeline
from agent.data_report import _derived_formulas, _limitations, _placeholder_columns

RNG = np.random.default_rng(17)


# ─────────────────────────────────────────────── fixtures: unrelated domains

def _flights(n=200):
    """Two columns that share their ENTIRE vocabulary, legitimately."""
    codes = ["DEL", "BOM", "BLR", "MAA", "CCU"]
    return pd.DataFrame({
        "origin": RNG.choice(codes, n),
        "destination": RNG.choice(codes, n),
        "delay_min": RNG.normal(12, 30, n),
    })


def _server_logs(n=300):
    """'error' is a real value here, not filler text."""
    return pd.DataFrame({
        "level": RNG.choice(["info", "warn", "error", "debug"], n),
        "service": RNG.choice(["auth", "cart", "search"], n),
        "latency_ms": RNG.gamma(2, 50, n),
    })


def _legal_corpus(n=60):
    """Genuine Latin. Real data that a naive filler check calls fake."""
    maxims = [
        "Res ipsa loquitur sed lex non cogit ad impossibilia",
        "Quercus alba et Rosa canina in situ conservata",
        "Nemo iudex in causa sua potest esse",
        "Sit venia verbo, nam error humanus est",
    ]
    return pd.DataFrame({
        "maxim": RNG.choice(maxims, n),
        "court": RNG.choice(["HC", "SC"], n),
        "year": RNG.integers(1990, 2020, n),
    })


def _lorem_table(n=60):
    """Actual Faker filler — this one SHOULD be caught."""
    filler = ["Quam numquam iste sunt nemo",
              "Earum quos enim minima dicta",
              "Voluptas harum deserunt maiores dolorum"]
    return pd.DataFrame({
        "description": RNG.choice(filler, n),
        "grp": RNG.choice(["a", "b"], n),
        "v": RNG.normal(size=n),
    })


def _finance(n=100):
    """Derived column in the hundreds of millions — float error far above 1e-9."""
    df = pd.DataFrame({
        "units": RNG.integers(1000, 9000, n).astype(float),
        "unit_price": RNG.uniform(10000, 99000, n).round(2),
    })
    df["notional"] = df["units"] * df["unit_price"]
    return df


def _wide(n=60, k=20):
    df = pd.DataFrame({f"m{i}": RNG.normal(size=n) for i in range(k)})
    df["total"] = df["m0"] * df["m1"]
    return df


def _multi_year(n=300):
    return pd.DataFrame({
        "observed_on": pd.date_range("2021-01-01", "2023-12-31", periods=n),
        "reading": RNG.normal(size=n),
    })


# ─────────────────────────────────────────────── wrong assertions

def test_columns_sharing_a_whole_vocabulary_are_not_called_duplicates():
    """origin/destination are two roles of one entity type, not a defect."""
    notes = DataPipeline._overlapping_categories(_flights())
    assert notes == [], notes


def test_a_stray_value_in_the_wrong_column_is_still_caught():
    """The check must stay useful after being made conservative: a value that is
    a small minority of BOTH vocabularies is the shape of a real leak."""
    df = pd.DataFrame({
        "category": ["Sweets", "Cake", "Mugs", "Diwali"] * 10,
        "occasion": ["Diwali", "Holi", "Birthday", "Holi"] * 10,
    })
    notes = " ".join(DataPipeline._overlapping_categories(df))
    assert "Diwali" in notes and "leaked" in notes


def test_genuine_latin_is_not_declared_fake():
    """Telling someone their real data is generated is the most damaging thing
    this module can say, so presence of Latin words is not enough."""
    assert _placeholder_columns(_legal_corpus()) == []
    assert not any("placeholder" in n for n in _limitations(_legal_corpus(), "corpus"))


def test_real_filler_text_is_still_caught():
    found = [c for c, _ in _placeholder_columns(_lorem_table())]
    assert found == ["description"], found


def test_an_ordinary_english_word_is_not_filler():
    """A log level of 'error' is data. 'error' is a Latin word too."""
    assert _placeholder_columns(_server_logs()) == []


# ─────────────────────────────────────────────── silent misses

def test_derived_columns_are_found_at_large_magnitudes():
    """notional = units x unit_price runs to ~1e8, where an absolute 1e-9
    tolerance finds nothing — on exactly the tables where the formula matters."""
    found = " ".join(_derived_formulas(_finance()))
    assert "notional" in found and "units" in found and "unit_price" in found


def test_derived_columns_are_found_in_a_wide_frame():
    """Bailing above a column count made 'no derivations' indistinguishable from
    'did not look'."""
    found = " ".join(_derived_formulas(_wide()))
    assert "total" in found and "m0" in found and "m1" in found


def test_a_frame_with_no_derived_column_reports_none():
    df = pd.DataFrame({"a": RNG.normal(size=50), "b": RNG.normal(size=50),
                       "c": RNG.normal(size=50)})
    assert _derived_formulas(df) == []


# ─────────────────────────────────────────────── shape-dependent judgements

def test_multi_year_data_is_not_told_it_lacks_a_second_year():
    notes = " ".join(_limitations(_multi_year(), "readings"))
    assert "month window only" not in notes


def test_single_character_categories_are_still_examined():
    """'M'/'F' and grade bands are ordinary categoricals; excluding them meant
    whole columns were never checked."""
    df = pd.DataFrame({"sex": ["M", "F"] * 20, "grade": ["A", "B"] * 20})
    # Nothing shared here, so nothing to report — but the columns must be seen.
    assert DataPipeline._overlapping_categories(df) == []
    # A stray 'M' that is a small minority of BOTH vocabularies still reports.
    shared = pd.DataFrame({"sex": ["M", "F", "X", "O"] * 10,
                           "size_band": ["M", "L", "S", "XL"] * 10})
    assert DataPipeline._overlapping_categories(shared)


def test_every_fixture_survives_a_full_validation_pass():
    """No dataset here may crash the validator or return an error string."""
    pipe = get_data_pipeline()
    fixtures = {
        "gen_flights": _flights(), "gen_logs": _server_logs(),
        "gen_corpus": _legal_corpus(), "gen_finance": _finance(),
        "gen_wide": _wide(), "gen_years": _multi_year(),
    }
    for name, df in fixtures.items():
        pipe.datasets[name] = df
        out = pipe.validate_dataset(name)
        assert not out.startswith("Error"), (name, out[:200])
        assert "category consistency" in out
        assert isinstance(_limitations(df, name), list)
        pipe.datasets.pop(name, None)
