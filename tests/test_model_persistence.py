"""
Persistence of trained model artifacts: train_models/tune_hyperparameters
auto-save to workspace/artifacts/ and ModelTrainer reloads them at startup,
so models survive process restarts.
"""

import os

import pandas as pd

from agent.ml.data_pipeline import get_data_pipeline
from agent.ml.model_training import get_model_trainer


def test_train_autosaves_and_reloads_artifact():
    trainer = get_model_trainer()
    pipe = get_data_pipeline()
    df = pd.DataFrame({
        "a": [float(i) for i in range(40)],
        "b": [float(i) * 2 for i in range(40)],
        "t": [float(i) * 3 + 0.5 for i in range(40)],
    })
    pipe.datasets["persist_src"] = df
    try:
        result = trainer.train_models("persist_src", "t", candidates=["linear"])
        assert "Best:" in result
        artifact_id = "persist_src__linear"
        assert trainer.get_trained_model(artifact_id) is not None

        save_msg = trainer.save_model(artifact_id)
        assert "Saved" in save_msg and "artifacts" in save_msg

        trainer._trained_models.pop(artifact_id)
        assert trainer.get_trained_model(artifact_id) is None

        trainer._load_artifacts_from_disk()
        art = trainer.get_trained_model(artifact_id)
        assert art is not None
        assert art["target_col"] == "t"
        assert art["task_type"] == "regression"
        assert "model" in art and "X_test" in art and "y_test" in art

        trainer.delete_model(artifact_id)
        assert trainer.get_trained_model(artifact_id) is None
    finally:
        pipe.datasets.pop("persist_src", None)


def test_delete_model_unknown_is_error():
    trainer = get_model_trainer()
    msg = trainer.delete_model("no_such_artifact_zzz")
    assert "Error" in msg
