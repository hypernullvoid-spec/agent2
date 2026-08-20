"""
Tests for the size guards on the loading path — agent/ml/data_pipeline.py and
the sandbox bridge's Parquet round trip.

Two things are being defended here, and only the second one is obvious.

The first is that a large file no longer takes the process down. That is easy
to state and easy to test with a lowered cap.

The second is that a capped read is never SILENT. Two million rows look exactly
like two million rows whether or not another three million were left on disk,
so nothing downstream can rediscover the shortfall — if the loader does not say
it, the report will confidently describe a subset as though it were the whole
dataset. Most of what follows is about that, not about memory.

The caps are read from the environment at import time, so these tests set the
module constants directly rather than re-importing.
"""

import os
import uuid

import numpy as np
import pandas as pd

from agent.data_analysis import reset_evidence, truncation_for
from agent.data_report import _limitations, _provenance_lines
from agent.ml import data_pipeline as dp
from agent.ml.data_pipeline import WORKSPACE_DIR, get_data_pipeline


# ─── harness ───────────────────────────────────────────────────────────────────

class _Caps:
    """Shrink the thresholds so ordinary fixtures exercise the big-file paths."""

    def __init__(self, max_mb=0.0, row_cap=50, chunk=10):
        self.want = (max_mb, row_cap, chunk)

    def __enter__(self):
        self.saved = (dp.MAX_LOAD_MB, dp.ROW_CAP, dp.STREAM_CHUNK_ROWS)
        dp.MAX_LOAD_MB, dp.ROW_CAP, dp.STREAM_CHUNK_ROWS = self.want
        return self

    def __exit__(self, *exc):
        dp.MAX_LOAD_MB, dp.ROW_CAP, dp.STREAM_CHUNK_ROWS = self.saved
        return False


def _fresh():
    pipe = get_data_pipeline()
    pipe.datasets.clear()
    pipe.sources.clear()
    pipe.truncation.clear()
    reset_evidence()
    return pipe


def _frame(n=200):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "order_id": np.arange(n),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "revenue": rng.gamma(2, 100, n).round(2),
    })


# Fixtures live in the workspace because safe_path() refuses anything outside
# it — the loaders are only reachable through that guard.
def _scratch(suffix: str) -> str:
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    return os.path.join(WORKSPACE_DIR, f"scale_{uuid.uuid4().hex[:8]}{suffix}")


def _csv(n=200):
    path = _scratch(".csv")
    _frame(n).to_csv(path, index=False)
    return path


def _xlsx(n=200):
    path = _scratch(".xlsx")
    _frame(n).to_excel(path, index=False, sheet_name="Sheet1")
    return path


# ─── small files keep the fast path ────────────────────────────────────────────

def test_a_small_csv_loads_whole_and_reports_no_truncation():
    pipe = _fresh()
    path = _csv(200)
    try:
        out = pipe.load_csv(path, "small")
        assert "NOT THE WHOLE FILE" not in out, out
        assert len(pipe.datasets["small"]) == 200
        assert "small" not in pipe.truncation
        assert truncation_for("small") == {}
    finally:
        os.unlink(path)


def test_a_small_excel_loads_whole():
    pipe = _fresh()
    path = _xlsx(120)
    try:
        pipe.load_excel(path, "small_x")
        assert len(pipe.datasets["small_x"]) == 120
        assert "small_x" not in pipe.truncation
    finally:
        os.unlink(path)


def test_an_explicit_nrows_is_not_second_guessed():
    """The caller asked for 10 rows; that is not the loader truncating."""
    pipe = _fresh()
    path = _csv(200)
    try:
        with _Caps():
            pipe.load_csv(path, "explicit", nrows=10)
        assert len(pipe.datasets["explicit"]) == 10
        assert "explicit" not in pipe.truncation
    finally:
        os.unlink(path)


# ─── large files are capped, and say so ────────────────────────────────────────

def test_a_capped_csv_load_announces_itself():
    pipe = _fresh()
    path = _csv(200)
    try:
        with _Caps(row_cap=50):
            out = pipe.load_csv(path, "big")
        assert len(pipe.datasets["big"]) == 50
        assert "NOT THE WHOLE FILE" in out, out
        assert "50 of 200" in out, out
        assert "25.0%" in out, out
        assert "LAST ones in the file" in out, out
    finally:
        os.unlink(path)


def test_a_capped_csv_load_records_the_shortfall():
    pipe = _fresh()
    path = _csv(200)
    try:
        with _Caps(row_cap=50):
            pipe.load_csv(path, "big")
        assert pipe.truncation["big"]["rows_read"] == 50
        assert pipe.truncation["big"]["rows_total"] == 200
        assert truncation_for("big")["rows_total"] == 200
    finally:
        os.unlink(path)


def test_a_capped_excel_load_streams_and_says_so():
    pipe = _fresh()
    path = _xlsx(200)
    try:
        with _Caps(row_cap=40):
            out = pipe.load_excel(path, "bigx")
        df = pipe.datasets["bigx"]
        assert len(df) == 40, len(df)
        assert list(df.columns) == ["order_id", "region", "date", "revenue"], list(df.columns)
        assert "NOT THE WHOLE FILE" in out, out
        assert truncation_for("bigx")["rows_total"] == 200
    finally:
        os.unlink(path)


def test_streamed_excel_values_match_the_file():
    """Streaming must not quietly shift or drop a column."""
    pipe = _fresh()
    path = _xlsx(200)
    try:
        with _Caps(row_cap=40):
            pipe.load_excel(path, "bigx")
        expected = _frame(200).head(40)
        streamed = pipe.datasets["bigx"]
        assert list(streamed["order_id"]) == list(expected["order_id"])
        assert list(streamed["region"]) == list(expected["region"])
        assert round(float(streamed["revenue"].sum()), 2) == round(
            float(expected["revenue"].sum()), 2)
    finally:
        os.unlink(path)


def test_streamed_excel_honours_usecols():
    pipe = _fresh()
    path = _xlsx(200)
    try:
        with _Caps(row_cap=40):
            pipe.load_excel(path, "cols", usecols=["region", "revenue"])
        assert list(pipe.datasets["cols"].columns) == ["region", "revenue"]
    finally:
        os.unlink(path)


def test_reloading_whole_clears_a_previous_truncation():
    """Stale truncation would make a full reload look partial forever."""
    pipe = _fresh()
    path = _csv(200)
    try:
        with _Caps(row_cap=50):
            pipe.load_csv(path, "twice")
        assert "twice" in pipe.truncation
        pipe.load_csv(path, "twice")
        assert "twice" not in pipe.truncation
    finally:
        os.unlink(path)


# ─── shrinking must not change an answer ───────────────────────────────────────

def test_shrink_saves_memory_without_changing_a_single_number():
    """Lossless is the whole contract — see _shrink's docstring on float32."""
    df = pd.DataFrame({
        "region": ["North", "South"] * 500,
        "revenue": np.linspace(1.0, 1000.0, 1000),
    })
    before_bytes = df.memory_usage(deep=True).sum()
    out = dp._shrink(df.copy())
    assert str(out["region"].dtype) == "category"
    assert out.memory_usage(deep=True).sum() < before_bytes
    assert list(out["region"].astype(str)) == list(df["region"])
    # exact, not approximate: a saving must never move a total
    assert float(out["revenue"].sum()) == float(df["revenue"].sum())
    assert str(out["revenue"].dtype) == "float64"


def test_shrink_never_downcasts_a_float():
    """A 6-rupee error on a 200-million total is a number that stops
    reconciling, which costs far more than the memory it saves."""
    df = pd.DataFrame({"revenue": np.random.default_rng(0).gamma(2, 500, 5000)})
    out = dp._shrink(df.copy())
    assert str(out["revenue"].dtype) == "float64"
    assert float(out["revenue"].sum()) == float(df["revenue"].sum())


def test_shrink_leaves_high_cardinality_text_alone():
    """All-distinct text as a category costs more than it saves."""
    df = pd.DataFrame({"email": [f"user{i}@x.com" for i in range(200)]})
    out = dp._shrink(df.copy())
    assert str(out["email"].dtype) != "category", out["email"].dtype
    assert list(out["email"]) == list(df["email"])


def test_shrink_does_not_touch_integers():
    """Downcasting an id column could change values that overflow."""
    df = pd.DataFrame({"order_id": np.arange(100, dtype="int64")})
    assert str(dp._shrink(df.copy())["order_id"].dtype) == "int64"


# ─── the report must state it ──────────────────────────────────────────────────

def test_provenance_states_the_partial_read():
    pipe = _fresh()
    path = _csv(200)
    try:
        with _Caps(row_cap=50):
            pipe.load_csv(path, "big")
        text = "\n".join(_provenance_lines(pipe.datasets["big"], "big")).replace("**", "")
        assert "Only part of the source file was read" in text, text
        assert "50 of 200" in text, text
        assert "150 rows not read" in text, text
    finally:
        os.unlink(path)


def test_limitations_state_the_partial_read_first():
    pipe = _fresh()
    path = _csv(200)
    try:
        with _Caps(row_cap=50):
            pipe.load_csv(path, "big")
        notes = _limitations(pipe.datasets["big"], "big")
        joined = " ".join(notes)
        assert "This is not the whole file" in joined, joined
        assert "50 of the 200 rows" in joined, joined
        # it qualifies every other caveat, so it leads
        assert "not the whole file" in notes[0].lower(), notes[0]
    finally:
        os.unlink(path)


def test_a_whole_file_adds_no_truncation_limitation():
    pipe = _fresh()
    path = _csv(200)
    try:
        pipe.load_csv(path, "whole")
        joined = " ".join(_limitations(pipe.datasets["whole"], "whole"))
        assert "not the whole file" not in joined.lower(), joined
    finally:
        os.unlink(path)


def test_reset_evidence_clears_truncation():
    pipe = _fresh()
    path = _csv(200)
    try:
        with _Caps(row_cap=50):
            pipe.load_csv(path, "big")
        assert truncation_for("big")
        reset_evidence()
        assert truncation_for("big") == {}
    finally:
        os.unlink(path)


# ─── the Parquet round trip ────────────────────────────────────────────────────

def test_dtypes_survive_the_bridge_round_trip():
    """The case the CSV sidecar existed to paper over."""
    from agent import data_bridge
    data_bridge.reset()
    df = pd.DataFrame({
        "when": pd.date_range("2024-01-01", periods=5, freq="D"),
        "region": pd.Categorical(["N", "S", "N", "E", "S"]),
        "revenue": [1.5, 2.5, 3.5, 4.5, 5.5],
        "flag": [True, False, True, False, True],
    })
    base = data_bridge._materialise("round_trip", df)
    assert base is not None
    namespace = {"pd": pd, "json": __import__("json")}
    exec(data_bridge._READER_SRC, namespace)
    back = namespace["_swarn_read"](data_bridge._abs(base))
    assert str(back["when"].dtype).startswith("datetime64"), back.dtypes.to_dict()
    assert str(back["region"].dtype) == "category", back.dtypes.to_dict()
    assert str(back["flag"].dtype) == "bool", back.dtypes.to_dict()
    pd.testing.assert_frame_equal(back, df)
    data_bridge.reset()


def test_the_bridge_prefers_parquet():
    from agent import data_bridge
    data_bridge.reset()
    base = data_bridge._materialise("fmt", pd.DataFrame({"a": [1, 2, 3]}))
    assert os.path.exists(data_bridge._abs(base + ".parquet"))
    assert not os.path.exists(data_bridge._abs(base + ".csv"))
    data_bridge.reset()


def test_the_bridge_falls_back_to_csv_on_an_impossible_frame():
    """A frame Arrow cannot type must not take the whole call down."""
    from agent import data_bridge
    data_bridge.reset()
    df = pd.DataFrame({"mixed": [1, "two", {"three": 3}, None]})
    base = data_bridge._materialise("awkward", df)
    assert base is not None
    assert os.path.exists(data_bridge._abs(base + ".csv"))
    data_bridge.reset()
