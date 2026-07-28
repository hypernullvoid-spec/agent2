"""
Phase 9: Evaluation & Visualization

Consumes a trained model artifact from Phase 8's ModelTrainer (by
artifact_id, e.g. "raw_features__xgboost" or "raw_features__xgboost_tuned")
and produces:
  - a detailed metrics report (regression: RMSE/MAE/R²/residual stats;
    classification: accuracy/precision/recall/F1/per-class breakdown)
  - a confusion matrix plot (classification)
  - an ROC curve + AUC (binary classification)
  - a residuals-vs-predicted plot (regression)
  - a multi-model comparison bar chart, reading every artifact currently
    held by ModelTrainer

Why this needs Phase 8's stored test split
─────────────────────────────────────────────
Confusion matrices and ROC curves need actual (y_true, y_pred) pairs,
not just the scalar metrics train_models() already returns. Phase 8 now
stores X_test/y_test on each artifact for exactly this reason — Phase 9
re-runs model.predict() on that *same* held-out split rather than
re-splitting the data, which would silently evaluate against a
different sample than the one the leaderboard metrics were computed on.

Output contract
─────────────────
Every plotting method here returns a result string describing what was
computed AND saves a PNG into the workspace (so the file persists and
is visible to the user through the existing file tools), matching the
write_file / save_dataset convention already used in this codebase.
Methods never raise — same "errors as strings" contract as the rest of
tools.py — and never block on a missing matplotlib backend (Agg is
forced before any pyplot import, since this runs headless).

Phase boundary
─────────────────
This module answers "is the model good, and how does it fail." It does
not decide deployment — Phase 10 (deployment.py) takes whichever
artifact_id the agent (or the user) judges acceptable from here and
packages it.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from agent.model_training import get_model_trainer

WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace")
)

PLOTS_SUBDIR = "plots"   # all Phase 9 figures land in workspace/plots/


class ModelEvaluator:
    """
    Stateless evaluation/plotting logic over Phase 8's trained artifacts.
    Holds no model state of its own — always reads through
    get_model_trainer() so it never goes stale relative to what's
    actually been trained.
    """

    # ───────────────────────────────────────────────── shared helpers

    def _get_artifact_or_error(self, artifact_id: str):
        trainer = get_model_trainer()
        artifact = trainer.get_trained_model(artifact_id)
        if artifact is None:
            available = trainer.list_trained_models()
            return None, (
                f"Error: no trained model artifact named '{artifact_id}'.\n{available}"
            )
        if "X_test" not in artifact or "y_test" not in artifact:
            return None, (
                f"Error: artifact '{artifact_id}' has no stored test split "
                "(it may have been trained by an older version of train_models). "
                "Re-run train_models or tune_hyperparameters to regenerate it."
            )
        return artifact, None

    def _plots_dir(self) -> str:
        d = os.path.join(WORKSPACE_DIR, PLOTS_SUBDIR)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_fig(self, fig, filename: str) -> str:
        path = os.path.join(self._plots_dir(), filename)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return os.path.join(PLOTS_SUBDIR, filename)   # relative path, for display to the agent

    # ───────────────────────────────────────────────── metrics report

    def evaluate_model(self, artifact_id: str) -> str:
        """
        Full diagnostic report for one trained artifact: re-computes
        metrics on its stored test split, plus extra detail beyond the
        leaderboard's headline numbers (per-class precision/recall for
        classification, residual distribution stats for regression).
        """
        artifact, err = self._get_artifact_or_error(artifact_id)
        if err:
            return err

        model = artifact["model"]
        X_test, y_test = artifact["X_test"], artifact["y_test"]
        task_type = artifact["task_type"]

        try:
            preds = self._predict(model, X_test)
        except Exception as e:
            return f"Error running model.predict(): {type(e).__name__}: {e}"

        lines = [
            f"Evaluation report — artifact '{artifact_id}'",
            f"  candidate   : {artifact.get('candidate', '?')}",
            f"  task_type   : {task_type}",
            f"  test rows   : {len(y_test)}",
            f"  target_col  : {artifact.get('target_col', '?')}",
        ]
        if artifact.get("best_params"):
            lines.append(f"  best_params : {artifact['best_params']}")

        if task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            residuals = np.asarray(y_test) - np.asarray(preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            lines += [
                "",
                "Regression metrics:",
                f"  RMSE : {rmse:.4f}",
                f"  MAE  : {mean_absolute_error(y_test, preds):.4f}",
                f"  R²   : {r2_score(y_test, preds):.4f}",
                "",
                "Residuals (y_true - y_pred):",
                f"  mean : {residuals.mean():.4f}   (closer to 0 is better — nonzero means systematic bias)",
                f"  std  : {residuals.std():.4f}",
                f"  min  : {residuals.min():.4f}   max: {residuals.max():.4f}",
            ]
        else:
            from sklearn.metrics import classification_report, accuracy_score
            lines += [
                "",
                f"Accuracy: {accuracy_score(y_test, preds):.4f}",
                "",
                "Per-class report:",
                classification_report(y_test, preds, zero_division=0),
            ]

        lines.append(
            f"\nNext: plot_confusion_matrix / plot_roc_curve (classification) or "
            f"plot_residuals (regression) on '{artifact_id}' for a visual breakdown, "
            f"or compare_models() to see every trained artifact side by side."
        )
        return "\n".join(lines)

    def _predict(self, model, X_test):
        """Handle both sklearn-style and the Phase 8 PyTorch-MLP code path.

        Checks the model's own module path before importing torch at all —
        torch's import has real side effects (loading native shared libs)
        that can fail in environments where torch is broken/misconfigured,
        and there's no reason to pay that cost for a non-torch model.
        """
        if type(model).__module__.startswith("torch"):
            try:
                import torch
                model.eval()
                with torch.no_grad():
                    out = model(torch.tensor(np.asarray(X_test, dtype=np.float32)))
                if out.shape[-1] == 1:
                    return out.squeeze(-1).numpy()
                return out.argmax(dim=1).numpy()
            except Exception as e:
                raise RuntimeError(f"PyTorch inference failed: {type(e).__name__}: {e}") from e
        return model.predict(X_test)

    # ───────────────────────────────────────────────── classification plots

    def plot_confusion_matrix(self, artifact_id: str) -> str:
        artifact, err = self._get_artifact_or_error(artifact_id)
        if err:
            return err
        if artifact["task_type"] == "regression":
            return f"Error: '{artifact_id}' is a regression model — use plot_residuals instead."

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        model = artifact["model"]
        X_test, y_test = artifact["X_test"], artifact["y_test"]
        try:
            preds = self._predict(model, X_test)
        except Exception as e:
            return f"Error running model.predict(): {type(e).__name__}: {e}"

        labels = sorted(pd.Series(y_test).unique())
        cm = confusion_matrix(y_test, preds, labels=labels)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {artifact_id}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        rel_path = self._save_fig(fig, f"{self._safe_name(artifact_id)}_confusion_matrix.png")
        return (
            f"Confusion matrix for '{artifact_id}' saved to {rel_path}\n"
            f"Labels: {labels}\n"
            f"Diagonal = correct predictions; off-diagonal = where the model confuses classes."
        )

    def plot_roc_curve(self, artifact_id: str) -> str:
        artifact, err = self._get_artifact_or_error(artifact_id)
        if err:
            return err
        if artifact["task_type"] != "binary_classification":
            return (
                f"Error: ROC curves require binary classification; "
                f"'{artifact_id}' is task_type='{artifact['task_type']}'."
            )

        model = artifact["model"]
        X_test, y_test = artifact["X_test"], artifact["y_test"]

        if not hasattr(model, "predict_proba"):
            return (
                f"Error: model for '{artifact_id}' has no predict_proba "
                "(e.g. the PyTorch MLP path doesn't expose probabilities here) — "
                "ROC curve isn't available for this candidate."
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        try:
            proba = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            return f"Error computing predict_proba: {type(e).__name__}: {e}"

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {artifact_id}")
        ax.legend(loc="lower right")

        rel_path = self._save_fig(fig, f"{self._safe_name(artifact_id)}_roc_curve.png")
        return (
            f"ROC curve for '{artifact_id}' saved to {rel_path}\n"
            f"AUC = {roc_auc:.4f}  (1.0 = perfect, 0.5 = no better than random)"
        )

    # ───────────────────────────────────────────────── regression plots

    def plot_residuals(self, artifact_id: str) -> str:
        artifact, err = self._get_artifact_or_error(artifact_id)
        if err:
            return err
        if artifact["task_type"] != "regression":
            return (
                f"Error: plot_residuals is for regression models; "
                f"'{artifact_id}' is task_type='{artifact['task_type']}' — "
                f"use plot_confusion_matrix or plot_roc_curve instead."
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model = artifact["model"]
        X_test, y_test = artifact["X_test"], artifact["y_test"]
        try:
            preds = self._predict(model, X_test)
        except Exception as e:
            return f"Error running model.predict(): {type(e).__name__}: {e}"

        residuals = np.asarray(y_test) - np.asarray(preds)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(preds, residuals, alpha=0.5, s=18)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual (actual - predicted)")
        axes[0].set_title("Residuals vs. Predicted")

        axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Residual Distribution")

        fig.suptitle(f"Residual diagnostics — {artifact_id}")
        rel_path = self._save_fig(fig, f"{self._safe_name(artifact_id)}_residuals.png")
        return (
            f"Residual plots for '{artifact_id}' saved to {rel_path}\n"
            f"A random scatter around 0 (left) and a roughly bell-shaped histogram "
            f"(right) indicate a well-fit model. A funnel/curve shape in the left "
            f"plot suggests the model is missing structure in the data."
        )

    # ───────────────────────────────────────────────── cross-model comparison

    def compare_models(self) -> str:
        """
        Bar chart comparing every trained artifact currently held by
        ModelTrainer on its primary metric (RMSE for regression,
        accuracy for classification — artifacts of different task types
        are grouped separately).
        """
        trainer = get_model_trainer()
        artifacts = trainer._trained_models   # read-only access, same module family
        if not artifacts:
            return "No trained model artifacts yet. Call train_models() first."

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        by_task: dict[str, list[tuple[str, float]]] = {}
        for art_id, art in artifacts.items():
            task = art["task_type"]
            metric_key = "rmse" if task == "regression" else "accuracy"
            value = art["metrics"].get(metric_key)
            if value is not None:
                by_task.setdefault(task, []).append((art_id, value))

        if not by_task:
            return "No artifacts with computed metrics found."

        n_groups = len(by_task)
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 4.5), squeeze=False)
        axes = axes[0]

        lines = ["Model comparison:"]
        for ax, (task, rows) in zip(axes, by_task.items()):
            metric_key = "rmse" if task == "regression" else "accuracy"
            lower_is_better = task == "regression"
            rows.sort(key=lambda r: r[1], reverse=not lower_is_better)
            names = [r[0] for r in rows]
            values = [r[1] for r in rows]

            ax.barh(names, values, color="steelblue")
            ax.set_xlabel(metric_key)
            ax.set_title(f"{task} ({metric_key}, {'lower' if lower_is_better else 'higher'} is better)")
            ax.invert_yaxis()   # best at top

            lines.append(f"\n  {task} (by {metric_key}):")
            for name, val in rows:
                lines.append(f"    {name}: {val:.4f}")

        fig.suptitle("Trained model comparison")
        fig.tight_layout()
        rel_path = self._save_fig(fig, "model_comparison.png")
        lines.insert(0, f"Comparison chart saved to {rel_path}")
        return "\n".join(lines)

    # ───────────────────────────────────────────────── internals

    @staticmethod
    def _safe_name(artifact_id: str) -> str:
        return "".join(c if c.isalnum() or c in "_-" else "_" for c in artifact_id)


# ─── singleton, matching the rest of the codebase ──────────────────────────────

_evaluator: Optional[ModelEvaluator] = None


def get_model_evaluator() -> ModelEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = ModelEvaluator()
    return _evaluator
