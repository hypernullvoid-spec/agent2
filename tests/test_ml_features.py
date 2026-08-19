"""
Tests for the gap-fix features: year-string parsing, target-aware cleaning,
cross-validated training metrics, feature importance, and in-session predict.
"""

import pandas as pd

from agent.data_cleaner import DataCleaner
from agent.ml.data_pipeline import get_data_pipeline
from agent.ml.model_training import get_model_trainer
from agent.runtime.tools import run_tool


# ── cleaning: year parsing ──────────────────────────────────────────────

def test_year_range_strings_are_parsed():
    df = pd.DataFrame({
        "YEAR": ["-2021", "(2021– )", "(2010–2022)", "2015"],
        "x": [1, 2, 3, 4],
    })
    c = DataCleaner(df, source="t")
    plan = c.diagnose()
    assert "parse year/year-range strings" in plan
    c.apply_cleaning(c.op_ids())
    assert c.df["YEAR"].tolist() == [2021, 2021, 2010, 2015]
    assert c.df["YEAR"].dtype.kind == "i"


def test_full_dates_not_treated_as_years():
    df = pd.DataFrame({
        "Date": ["2023-01-05", "2023-06-01", "2022-11-30"],
        "x": [1, 2, 3],
    })
    c = DataCleaner(df, source="t")
    plan = c.diagnose()
    assert "parse year" not in plan
    assert "date formats" in plan


# ── cleaning: target-aware null handling ─────────────────────────────────

def test_target_col_drops_rows_instead_of_imputing():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "rating": [3.5, None, 4.0, None, 5.0],
    })
    c = DataCleaner(df, source="t", target_col="rating")
    plan = c.diagnose()
    assert "rows missing target 'rating'" in plan
    assert "impute 'rating'" not in plan
    c.apply_cleaning(c.op_ids())
    assert c.df["rating"].isna().sum() == 0
    assert len(c.df) == 3


def test_target_col_default_keeps_impute():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "rating": [3.5, 4.0, 4.2, 4.5, 5.0, None],
    })
    c = DataCleaner(df, source="t")
    plan = c.diagnose()
    assert "impute 'rating'" in plan


# ── ML: task-type detection ──────────────────────────────────────────────

def _trainer():
    return get_model_trainer()


def test_small_integer_ratings_are_classification():
    y = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    assert _trainer()._detect_task_type(y) == "multiclass_classification"


def test_step_continuous_integers_are_regression():
    y = pd.Series([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45])
    assert _trainer()._detect_task_type(y) == "regression"


def test_fractional_values_are_regression():
    y = pd.Series([1.5, 2.1, 3.7, 4.2, 5.9, 6.3, 7.1, 8.8, 9.4, 10.2])
    assert _trainer()._detect_task_type(y) == "regression"


def test_binary_is_binary():
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    assert _trainer()._detect_task_type(y) == "binary_classification"


# ── ML: CV in leaderboard, feature importance, predict ───────────────────

def _fit_linear_fixture():
    pipe = get_data_pipeline()
    df = pd.DataFrame({
        "a": [float(i) for i in range(30)],
        "b": [float(i) * 2 for i in range(30)],
        "t": [float(i) * 3 + 0.5 for i in range(30)],
    })
    pipe.datasets["ml_fix_src"] = df
    trainer = get_model_trainer()
    result = trainer.train_models("ml_fix_src", "t", candidates=["linear"], cv_folds=3)
    return pipe, trainer, result


def test_train_reports_cv_score():
    pipe, trainer, result = _fit_linear_fixture()
    try:
        assert "cv_rmse=" in result
        art = trainer.get_trained_model("ml_fix_src__linear")
        assert art is not None and "cv_score" not in art  # cv is leaderboard-only
        assert art["task_type"] == "regression"
    finally:
        pipe.datasets.pop("ml_fix_src", None)
        trainer.delete_model("ml_fix_src__linear")


def test_feature_importance_ranks_features():
    pipe, trainer, result = _fit_linear_fixture()
    try:
        out = run_tool("feature_importance", {"artifact_id": "ml_fix_src__linear"})
        assert "Feature importance" in out
        assert "numeric" in out or "a" in out
        assert "importance.png" in out
    finally:
        pipe.datasets.pop("ml_fix_src", None)
        trainer.delete_model("ml_fix_src__linear")


def test_predict_scores_new_rows():
    pipe, trainer, result = _fit_linear_fixture()
    try:
        art = trainer.get_trained_model("ml_fix_src__linear")
        cols = art["feature_columns"]
        row = {cols[0]: 10.0, cols[1]: 20.0}
        out = run_tool("predict", {"artifact_id": "ml_fix_src__linear", "rows": [row]})
        assert "Predictions" in out and "row 0" in out
        missing = run_tool("predict", {"artifact_id": "ml_fix_src__linear", "rows": [{"nope": 1}]})
        assert "missing feature column" in missing
    finally:
        pipe.datasets.pop("ml_fix_src", None)
        trainer.delete_model("ml_fix_src__linear")
