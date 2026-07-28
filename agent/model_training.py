"""
Phase 8: Model Training & Hyperparameter Optimization

Consumes a feature-engineered dataset from Phase 7 (or any dataset in
DataPipeline's registry) and trains candidate models from three families
— scikit-learn, XGBoost/LightGBM, and PyTorch (a small MLP) — selecting
sensible candidates based on the task type, then optionally runs Optuna
to tune the best-performing family.

Why this lives outside the sandbox
────────────────────────────────────
Earlier phases (2) run arbitrary agent-written code inside Docker for
isolation. Training here runs in-process instead, because:
  - it needs direct access to DataPipeline's in-memory DataFrames
    (round-tripping through the sandbox would mean serializing data
    across the container boundary for every fit/predict call)
  - the model families are fixed, named functions, not arbitrary code —
    same trust boundary as tools.py's other deterministic tools
If the agent wants the *training script itself* containerized (e.g. to
mirror a production training job), it can still write one with
write_file and run it via run_python — this module is the fast,
in-process path for iterating on model choice.

Output contract
─────────────────
train_models() fits each requested candidate, evaluates with an
appropriate metric for the task type, and returns a leaderboard as text.
The winning model object is kept in memory (self._trained_models) so
Phase 9 (evaluation/visualization) and Phase 10 (deployment) can retrieve
it via get_trained_model() without retraining.
"""

import time
from typing import Optional

import numpy as np
import pandas as pd

from agent.data_pipeline import get_data_pipeline

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


class ModelTrainer:
    """
    Holds trained model artifacts for the current session so later
    phases (evaluation, deployment) can retrieve the winner without
    re-fitting. One instance per process — same singleton pattern as
    DataPipeline / FeatureEngine / Sandbox.
    """

    def __init__(self):
        self._trained_models: dict[str, dict] = {}   # name -> {"model":, "metrics":, "task_type":, ...}
        self._last_leaderboard: list[dict] = []

    # ───────────────────────────────────────────────── task detection

    def _detect_task_type(self, y: pd.Series) -> str:
        n_unique = y.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(y) and n_unique > 20:
            return "regression"
        if n_unique == 2:
            return "binary_classification"
        if 2 < n_unique <= 20:
            return "multiclass_classification"
        return "regression"   # safest fallback for high-cardinality numeric-like targets

    def _candidate_models(self, task_type: str) -> dict[str, str]:
        """Return {candidate_key: human_label} appropriate for the task type."""
        if task_type == "regression":
            return {
                "linear":        "Linear/Ridge Regression (sklearn)",
                "random_forest": "Random Forest Regressor (sklearn)",
                "xgboost":       "XGBoost Regressor",
                "lightgbm":      "LightGBM Regressor",
                "mlp":           "PyTorch MLP Regressor",
            }
        return {
            "logistic":      "Logistic Regression (sklearn)",
            "random_forest": "Random Forest Classifier (sklearn)",
            "xgboost":       "XGBoost Classifier",
            "lightgbm":      "LightGBM Classifier",
            "mlp":           "PyTorch MLP Classifier",
        }

    # ───────────────────────────────────────────────── public: training

    def train_models(
        self,
        name: str,
        target_col: str,
        candidates: Optional[list[str]] = None,
        test_size: float = DEFAULT_TEST_SIZE,
        run_id: Optional[str] = None,
    ) -> str:
        """
        Train and evaluate candidate models on a dataset already in
        DataPipeline's registry (typically Phase 7's *_features output).

        candidates: subset of {"linear"/"logistic", "random_forest",
        "xgboost", "lightgbm", "mlp"}. If omitted, all available
        candidates for the detected task type are tried.
        """
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            return "Error: training requires 'pip install scikit-learn'."

        df = get_data_pipeline().datasets.get(name)
        if df is None:
            return f"Error: no dataset named '{name}' is loaded."
        if target_col not in df.columns:
            return f"Error: target_col '{target_col}' not found in '{name}'."

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.select_dtypes(exclude="number").shape[1] > 0:
            return (
                "Error: non-numeric columns remain in the feature set "
                f"({list(X.select_dtypes(exclude='number').columns)}). "
                "Run Phase 7's engineer_features() on this dataset first."
            )

        task_type = self._detect_task_type(y)
        available = self._candidate_models(task_type)
        chosen = candidates or list(available.keys())
        chosen = [c for c in chosen if c in available]
        if not chosen:
            return (
                f"Error: no valid candidates for task_type='{task_type}'. "
                f"Available: {list(available.keys())}"
            )

        stratify = y if task_type != "regression" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=DEFAULT_RANDOM_STATE, stratify=stratify
            )
        except Exception as e:
            return f"Error splitting data: {type(e).__name__}: {e}"

        leaderboard = []
        for key in chosen:
            try:
                t0 = time.time()
                model, metrics = self._fit_and_eval(key, task_type, X_train, y_train, X_test, y_test)
                elapsed = round(time.time() - t0, 2)
                leaderboard.append({
                    "candidate": key,
                    "label": available[key],
                    "metrics": metrics,
                    "train_seconds": elapsed,
                    "model": model,
                })
            except ImportError as e:
                leaderboard.append({"candidate": key, "label": available[key], "error": f"not installed: {e}"})
            except Exception as e:
                leaderboard.append({"candidate": key, "label": available[key], "error": f"{type(e).__name__}: {e}"})

        self._last_leaderboard = leaderboard
        primary_metric = "rmse" if task_type == "regression" else "accuracy"
        lower_is_better = task_type == "regression"

        scored = [e for e in leaderboard if "metrics" in e]
        if scored:
            scored.sort(key=lambda e: e["metrics"][primary_metric], reverse=not lower_is_better)
            best = scored[0]
            artifact_id = run_id or f"{name}__{best['candidate']}"
            self._trained_models[artifact_id] = {
                "model": best["model"],
                "task_type": task_type,
                "metrics": best["metrics"],
                "feature_columns": list(X.columns),
                "target_col": target_col,
                "candidate": best["candidate"],
                "X_test": X_test,
                "y_test": y_test,
            }

        return self._format_leaderboard(name, task_type, primary_metric, lower_is_better, leaderboard, scored)

    def _format_leaderboard(self, name, task_type, primary_metric, lower_is_better, leaderboard, scored) -> str:
        lines = [
            f"Training run on '{name}'  →  task_type detected: {task_type}",
            f"Primary metric: {primary_metric} ({'lower' if lower_is_better else 'higher'} is better)",
            "",
            "Leaderboard:",
        ]
        for e in sorted(leaderboard, key=lambda x: 0 if "error" in x else 1):
            if "error" in e:
                lines.append(f"  {e['label']:<32} FAILED — {e['error']}")
            else:
                m = ", ".join(f"{k}={v:.4f}" for k, v in e["metrics"].items())
                lines.append(f"  {e['label']:<32} {m}   ({e['train_seconds']}s)")

        if scored:
            best = scored[0]
            artifact_id = best["candidate"]
            lines.append(
                f"\nBest: {best['label']}  →  registered as a trained model artifact "
                f"(key contains '{name}__{artifact_id}' or the run_id you supplied).\n"
                f"Next: call evaluate_model(...) for full diagnostics/plots (Phase 9), "
                f"or tune_hyperparameters(...) below to search around this candidate."
            )
        else:
            lines.append(
                "\nAll candidates failed — see error messages above. Common "
                "causes: missing packages (install_package), or nulls in the "
                "target column (engineer_features preserves the target "
                "unchanged — drop or impute target nulls in the source "
                "dataset before engineering features)."
            )
        return "\n".join(lines)

    def _fit_and_eval(self, key, task_type, X_train, y_train, X_test, y_test):
        is_regression = task_type == "regression"

        if key in ("linear", "logistic"):
            from sklearn.linear_model import LinearRegression, LogisticRegression
            model = LinearRegression() if is_regression else LogisticRegression(max_iter=1000)
        elif key == "random_forest":
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            model = (RandomForestRegressor(n_estimators=200, random_state=DEFAULT_RANDOM_STATE)
                      if is_regression else
                      RandomForestClassifier(n_estimators=200, random_state=DEFAULT_RANDOM_STATE))
        elif key == "xgboost":
            import xgboost as xgb
            model = (xgb.XGBRegressor(random_state=DEFAULT_RANDOM_STATE)
                      if is_regression else
                      xgb.XGBClassifier(random_state=DEFAULT_RANDOM_STATE, eval_metric="logloss"))
        elif key == "lightgbm":
            import lightgbm as lgb
            model = (lgb.LGBMRegressor(random_state=DEFAULT_RANDOM_STATE, verbosity=-1)
                      if is_regression else
                      lgb.LGBMClassifier(random_state=DEFAULT_RANDOM_STATE, verbosity=-1))
        elif key == "mlp":
            return self._fit_torch_mlp(task_type, X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown candidate key '{key}'")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = self._compute_metrics(task_type, y_test, preds)
        return model, metrics

    def _fit_torch_mlp(self, task_type, X_train, y_train, X_test, y_test):
        import torch
        import torch.nn as nn

        is_regression = task_type == "regression"
        n_features = X_train.shape[1]
        n_classes = int(y_train.nunique()) if not is_regression else 1

        X_train_t = torch.tensor(np.asarray(X_train, dtype=np.float32))
        X_test_t = torch.tensor(np.asarray(X_test, dtype=np.float32))

        if is_regression:
            y_train_t = torch.tensor(np.asarray(y_train, dtype=np.float32)).reshape(-1, 1)
            out_dim = 1
        else:
            classes = sorted(y_train.unique())
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_train_t = torch.tensor(y_train.map(class_to_idx).values, dtype=torch.long)
            out_dim = max(2, n_classes)

        model = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, out_dim),
        )
        loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for _epoch in range(100):
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = loss_fn(out, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            raw_preds = model(X_test_t)
            if is_regression:
                preds = raw_preds.squeeze(-1).numpy()
            else:
                idx_preds = raw_preds.argmax(dim=1).numpy()
                idx_to_class = {i: c for c, i in class_to_idx.items()}
                preds = np.array([idx_to_class[i] for i in idx_preds])

        metrics = self._compute_metrics(task_type, y_test, preds)
        return model, metrics

    def _compute_metrics(self, task_type, y_true, y_pred) -> dict:
        if task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            return {
                "rmse": float(rmse),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }
        from sklearn.metrics import accuracy_score, f1_score
        average = "binary" if task_type == "binary_classification" else "macro"
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        }

    # ───────────────────────────────────────────────── public: HPO

    def tune_hyperparameters(
        self,
        name: str,
        target_col: str,
        candidate: str = "xgboost",
        n_trials: int = 25,
        test_size: float = DEFAULT_TEST_SIZE,
    ) -> str:
        """
        Run an Optuna study over a fixed, sensible search space for the
        given candidate family, on the same train/test split logic as
        train_models(). Registers the best-found model under
        "<name>__<candidate>_tuned".
        """
        try:
            import optuna
            from sklearn.model_selection import train_test_split
        except ImportError:
            return "Error: HPO requires 'pip install optuna scikit-learn'."

        df = get_data_pipeline().datasets.get(name)
        if df is None:
            return f"Error: no dataset named '{name}' is loaded."
        if target_col not in df.columns:
            return f"Error: target_col '{target_col}' not found in '{name}'."

        X = df.drop(columns=[target_col])
        y = df[target_col]
        task_type = self._detect_task_type(y)
        is_regression = task_type == "regression"

        stratify = y if not is_regression else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=DEFAULT_RANDOM_STATE, stratify=stratify
        )

        search_space_fn, build_model_fn = self._hpo_space_for(candidate, is_regression)
        if search_space_fn is None:
            return (
                f"Error: HPO search space not defined for candidate '{candidate}'. "
                "Supported: random_forest, xgboost, lightgbm."
            )

        primary_metric = "rmse" if is_regression else "accuracy"
        lower_is_better = is_regression

        def objective(trial):
            params = search_space_fn(trial)
            model = build_model_fn(params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = self._compute_metrics(task_type, y_test, preds)
            trial.set_user_attr("metrics", metrics)
            return metrics[primary_metric]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        direction = "minimize" if lower_is_better else "maximize"
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_trial.params
        best_metrics = study.best_trial.user_attrs.get("metrics", {})
        best_model = build_model_fn(study.best_trial.params)
        best_model.fit(X_train, y_train)

        artifact_id = f"{name}__{candidate}_tuned"
        self._trained_models[artifact_id] = {
            "model": best_model,
            "task_type": task_type,
            "metrics": best_metrics,
            "feature_columns": list(X.columns),
            "target_col": target_col,
            "candidate": f"{candidate}_tuned",
            "best_params": best_params,
            "X_test": X_test,
            "y_test": y_test,
        }

        lines = [
            f"Optuna HPO complete — {n_trials} trials, candidate='{candidate}', task_type={task_type}",
            f"Best {primary_metric}: {best_metrics.get(primary_metric):.4f}",
            f"Best params: {best_params}",
            f"Registered as trained model artifact '{artifact_id}'.",
        ]
        return "\n".join(lines)

    def _hpo_space_for(self, candidate: str, is_regression: bool):
        if candidate == "random_forest":
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

            def space(trial):
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                }

            def build(params):
                cls = RandomForestRegressor if is_regression else RandomForestClassifier
                return cls(random_state=DEFAULT_RANDOM_STATE, **params)

            return space, build

        if candidate == "xgboost":
            import xgboost as xgb

            def space(trial):
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                }

            def build(params):
                cls = xgb.XGBRegressor if is_regression else xgb.XGBClassifier
                kwargs = {} if is_regression else {"eval_metric": "logloss"}
                return cls(random_state=DEFAULT_RANDOM_STATE, **params, **kwargs)

            return space, build

        if candidate == "lightgbm":
            import lightgbm as lgb

            def space(trial):
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 2, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 8, 128),
                }

            def build(params):
                cls = lgb.LGBMRegressor if is_regression else lgb.LGBMClassifier
                return cls(random_state=DEFAULT_RANDOM_STATE, verbosity=-1, **params)

            return space, build

        return None, None

    # ───────────────────────────────────────────────── retrieval (used by Phase 9/10)

    def get_trained_model(self, artifact_id: str) -> Optional[dict]:
        return self._trained_models.get(artifact_id)

    def list_trained_models(self) -> str:
        if not self._trained_models:
            return "No trained model artifacts yet. Call train_models() or tune_hyperparameters() first."
        lines = ["Trained model artifacts:"]
        for key, art in self._trained_models.items():
            m = ", ".join(f"{k}={v:.4f}" for k, v in art["metrics"].items())
            lines.append(f"  {key}  [{art['candidate']}]  {m}")
        return "\n".join(lines)


# ─── singleton, matching the rest of the codebase ──────────────────────────────

_trainer: Optional[ModelTrainer] = None


def get_model_trainer() -> ModelTrainer:
    global _trainer
    if _trainer is None:
        _trainer = ModelTrainer()
    return _trainer
