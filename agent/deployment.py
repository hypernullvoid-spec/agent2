"""
Phase 10: Deployment Automation

Takes a trained model artifact from Phase 8 (optionally vetted via
Phase 9's evaluation tools first) and packages it as a deployable unit:
  - a serialized model file: pickle (joblib) always; ONNX when the
    candidate family supports it and the converter is installed
  - a FastAPI wrapper exposing /predict, with OpenAPI docs auto-generated
    by FastAPI itself from a Pydantic input schema built from the
    artifact's feature_columns
  - a Dockerfile to containerize the FastAPI service

This mirrors HeyNeo's "one-click deployment" output: the agent calls one
tool, gets a self-contained folder under workspace/deployments/<id>/
that a user can `docker build && docker run` without touching the rest
of this codebase.

Design choices
────────────────
- Pickle via joblib is the default and always-available path. ONNX
  export is attempted only for sklearn-native estimators (linear/
  logistic/random forest) via skl2onnx, since XGBoost/LightGBM/PyTorch
  each need their own converter library this project doesn't assume is
  installed. If skl2onnx isn't available or the candidate isn't sklearn-
  native, export_format silently falls back to pickle and says so in
  the returned report — never raises, same contract as every other tool
  here.
- The generated FastAPI app is a plain template string, not executed by
  this process — it's written to disk for the user (or a later
  containerized run) to serve. This keeps Phase 10 itself dependency-
  light: fastapi/uvicorn don't need to be installed in *this*
  environment, only in the one that eventually runs the generated app
  (the generated Dockerfile installs them there).
- feature_columns on the artifact (set by Phase 8) becomes the
  prediction endpoint's required JSON fields — so the API's input
  contract is generated from the exact columns the model was trained
  on, not retyped by hand.
"""

import json
import os
import textwrap
from typing import Optional

from agent.model_training import get_model_trainer

WORKSPACE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace")
)
DEPLOYMENTS_SUBDIR = "deployments"

# sklearn-native estimator class name prefixes that skl2onnx can convert.
# XGBoost/LightGBM/PyTorch need their own converters and aren't attempted here.
_ONNX_ELIGIBLE_MODULES = ("sklearn.linear_model", "sklearn.ensemble")


class DeploymentPackager:
    """
    Stateless packaging logic over Phase 8's trained artifacts. Like
    ModelEvaluator, holds no state of its own — always reads through
    get_model_trainer() so it can never go stale.
    """

    def _deployments_dir(self, artifact_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in artifact_id)
        d = os.path.join(WORKSPACE_DIR, DEPLOYMENTS_SUBDIR, safe)
        os.makedirs(d, exist_ok=True)
        return d

    # ───────────────────────────────────────────────── public entry point

    def package_model(
        self,
        artifact_id: str,
        export_format: str = "pickle",
        api_title: Optional[str] = None,
    ) -> str:
        """
        Package a trained artifact for deployment: serialize the model,
        generate a FastAPI wrapper + OpenAPI-documented /predict endpoint,
        and write a Dockerfile. Everything lands under
        workspace/deployments/<artifact_id>/.

        export_format: "pickle" (default, always works) or "onnx"
        (attempted only for sklearn-native estimators; falls back to
        pickle with an explanation if unavailable).
        """
        trainer = get_model_trainer()
        artifact = trainer.get_trained_model(artifact_id)
        if artifact is None:
            return (
                f"Error: no trained model artifact named '{artifact_id}'.\n"
                f"{trainer.list_trained_models()}"
            )

        out_dir = self._deployments_dir(artifact_id)
        model = artifact["model"]
        feature_columns = artifact.get("feature_columns", [])
        target_col = artifact.get("target_col", "prediction")
        task_type = artifact.get("task_type", "unknown")

        if not feature_columns:
            return (
                f"Error: artifact '{artifact_id}' has no recorded feature_columns "
                "— cannot generate a typed API schema from it."
            )

        # ── 1. serialize the model ──────────────────────────────────
        serialize_report, model_filename, actual_format = self._serialize_model(
            model, out_dir, export_format
        )

        # ── 2. generate the FastAPI app ──────────────────────────────
        api_title = api_title or f"{artifact_id} prediction service"
        app_code = self._render_fastapi_app(
            artifact_id=artifact_id,
            model_filename=model_filename,
            model_format=actual_format,
            feature_columns=feature_columns,
            target_col=target_col,
            task_type=task_type,
            api_title=api_title,
        )
        app_path = os.path.join(out_dir, "app.py")
        with open(app_path, "w", encoding="utf-8") as f:
            f.write(app_code)

        # ── 3. requirements.txt for the deployed service ─────────────
        req_path = os.path.join(out_dir, "requirements.txt")
        with open(req_path, "w", encoding="utf-8") as f:
            f.write(self._render_requirements(actual_format))

        # ── 4. Dockerfile ──────────────────────────────────────────────
        dockerfile_path = os.path.join(out_dir, "Dockerfile")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(self._render_dockerfile())

        # ── 5. a tiny metadata file describing the artifact ───────────
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "artifact_id": artifact_id,
                "candidate": artifact.get("candidate"),
                "task_type": task_type,
                "metrics": artifact.get("metrics"),
                "feature_columns": feature_columns,
                "target_col": target_col,
                "model_format": actual_format,
                "model_filename": model_filename,
            }, f, indent=2, default=str)

        rel_dir = os.path.relpath(out_dir, WORKSPACE_DIR)
        return (
            f"Deployment package created at workspace/{rel_dir}/\n"
            f"{serialize_report}\n\n"
            f"Files written:\n"
            f"  {model_filename}     — serialized model\n"
            f"  app.py                — FastAPI service (GET /health, POST /predict, "
            f"auto OpenAPI docs at /docs)\n"
            f"  requirements.txt      — deploy-time dependencies\n"
            f"  Dockerfile            — container build\n"
            f"  metadata.json         — artifact metadata for reference\n\n"
            f"To run locally:\n"
            f"  cd workspace/{rel_dir} && pip install -r requirements.txt && "
            f"uvicorn app:app --reload\n"
            f"To containerize:\n"
            f"  cd workspace/{rel_dir} && docker build -t {artifact_id.lower().replace('_','-')} . "
            f"&& docker run -p 8000:8000 {artifact_id.lower().replace('_','-')}"
        )

    # ───────────────────────────────────────────────── serialization

    def _serialize_model(self, model, out_dir: str, export_format: str) -> tuple[str, str, str]:
        """Returns (report_text, filename, actual_format_used)."""
        if export_format == "onnx":
            onnx_report, onnx_filename = self._try_export_onnx(model, out_dir)
            if onnx_filename:
                return onnx_report, onnx_filename, "onnx"
            # fall through to pickle, with an explanation already in onnx_report
            pickle_report, pickle_filename = self._export_pickle(model, out_dir)
            return f"{onnx_report}\nFalling back to pickle: {pickle_report}", pickle_filename, "pickle"

        report, filename = self._export_pickle(model, out_dir)
        return report, filename, "pickle"

    def _export_pickle(self, model, out_dir: str) -> tuple[str, str]:
        import joblib
        filename = "model.joblib"
        path = os.path.join(out_dir, filename)
        try:
            joblib.dump(model, path)
        except Exception as e:
            return f"Error pickling model: {type(e).__name__}: {e}", filename
        size_kb = os.path.getsize(path) / 1024
        return f"Serialized model to {filename} ({size_kb:.1f} KB, joblib/pickle format)", filename

    def _try_export_onnx(self, model, out_dir: str) -> tuple[str, Optional[str]]:
        module_name = type(model).__module__
        if not any(module_name.startswith(m) for m in _ONNX_ELIGIBLE_MODULES):
            return (
                f"ONNX export skipped: candidate type '{type(model).__name__}' "
                f"(module {module_name}) isn't a sklearn-native estimator this "
                "packager converts — only linear/logistic/random-forest models "
                "support ONNX export here. XGBoost/LightGBM/PyTorch need their "
                "own converter library and aren't attempted automatically.",
                None,
            )
        try:
            import skl2onnx
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            return (
                "ONNX export skipped: 'pip install skl2onnx onnxruntime' to enable it.",
                None,
            )

        n_features = getattr(model, "n_features_in_", None)
        if n_features is None:
            return ("ONNX export skipped: model has no n_features_in_ (not fitted?).", None)

        try:
            onnx_model = skl2onnx.to_onnx(model, initial_types=[("input", FloatTensorType([None, n_features]))])
        except Exception as e:
            return (f"ONNX export failed: {type(e).__name__}: {e}", None)

        filename = "model.onnx"
        path = os.path.join(out_dir, filename)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        size_kb = os.path.getsize(path) / 1024
        return (f"Exported model to {filename} ({size_kb:.1f} KB, ONNX format)", filename)

    # ───────────────────────────────────────────────── code generation

    def _render_fastapi_app(
        self, artifact_id, model_filename, model_format, feature_columns,
        target_col, task_type, api_title,
    ) -> str:
        # Map each original column name to a valid Python identifier, once,
        # and reuse that mapping everywhere (Pydantic field, FEATURE_ORDER,
        # and the getattr() call in /predict all have to agree on the same
        # name or the request body silently won't line up with the model's
        # expected column order).
        safe_names = [self._py_identifier(c) for c in feature_columns]
        target_field = self._py_identifier(target_col) or "prediction"

        field_lines = "\n".join(f"    {name}: float" for name in safe_names)
        feature_order_literal = json.dumps(safe_names)

        loader_lines = (
            self._onnx_loader_block(model_filename)
            if model_format == "onnx"
            else self._pickle_loader_block(model_filename)
        )

        module_docstring = (
            f'"""\n'
            f"Auto-generated FastAPI service for trained artifact '{artifact_id}'.\n"
            f"Generated by Phase 10 (deployment.py) — edit freely, this is yours now.\n\n"
            f"Run locally:\n"
            f"    pip install -r requirements.txt\n"
            f"    uvicorn app:app --reload\n"
            f"Then open http://127.0.0.1:8000/docs for interactive OpenAPI docs.\n"
            f'"""'
        )

        # Built as a flat list of top-level lines (no shared indentation to
        # accidentally get dedented incorrectly) — each multi-line block
        # (docstring, loader, field_lines) is already correctly formatted
        # internally and just gets concatenated in, not re-indented.
        parts = [
            module_docstring,
            "",
            "from fastapi import FastAPI, HTTPException",
            "from pydantic import BaseModel",
            "import numpy as np",
            "",
            loader_lines.rstrip("\n"),
            "",
            f"FEATURE_ORDER = {feature_order_literal}",
            "",
            "app = FastAPI(",
            f'    title="{api_title}",',
            f'    description="Auto-generated prediction service for artifact \'{artifact_id}\' (task_type={task_type}).",',
            '    version="1.0.0",',
            ")",
            "",
            "",
            "class PredictRequest(BaseModel):",
            field_lines,
            "",
            "",
            "class PredictResponse(BaseModel):",
            f"    {target_field}: float",
            "",
            "",
            '@app.get("/health")',
            "def health():",
            f'    return {{"status": "ok", "artifact_id": "{artifact_id}"}}',
            "",
            "",
            '@app.post("/predict", response_model=PredictResponse)',
            "def predict(request: PredictRequest):",
            "    try:",
            "        row = [getattr(request, f) for f in FEATURE_ORDER]",
            "        x = np.array([row], dtype=np.float32)",
            "        prediction = run_inference(x)",
            "    except Exception as e:",
            '        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")',
            f"    return PredictResponse(**{{\"{target_field}\": float(prediction)}})",
            "",
        ]
        return "\n".join(parts)

    def _py_identifier(self, col: str) -> str:
        """Sanitize a feature/column name into a valid Python identifier for Pydantic fields."""
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in col)
        if not safe or safe[0].isdigit():
            safe = f"f_{safe}"
        return safe

    def _pickle_loader_block(self, model_filename: str) -> str:
        return textwrap.dedent(f'''\
            import joblib

            _model = joblib.load("{model_filename}")


            def run_inference(x: np.ndarray):
                pred = _model.predict(x)
                return pred[0]
        ''')

    def _onnx_loader_block(self, model_filename: str) -> str:
        return textwrap.dedent(f'''\
            import onnxruntime as ort

            _session = ort.InferenceSession("{model_filename}")
            _input_name = _session.get_inputs()[0].name


            def run_inference(x: np.ndarray):
                outputs = _session.run(None, {{_input_name: x}})
                return outputs[0].reshape(-1)[0]
        ''')

    def _render_requirements(self, model_format: str) -> str:
        lines = [
            "fastapi>=0.110.0",
            "uvicorn[standard]>=0.29.0",
            "numpy>=1.26.0",
            "pydantic>=2.6.0",
        ]
        if model_format == "onnx":
            lines.append("onnxruntime>=1.17.0")
        else:
            lines.append("joblib>=1.3.0")
            lines.append("scikit-learn>=1.4.0   # needed to unpickle sklearn-family estimators")
        return "\n".join(lines) + "\n"

    def _render_dockerfile(self) -> str:
        return textwrap.dedent('''\
            FROM python:3.11-slim

            WORKDIR /app

            COPY requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt

            COPY . .

            EXPOSE 8000

            CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
        ''')


# ─── singleton, matching the rest of the codebase ──────────────────────────────

_packager: Optional[DeploymentPackager] = None


def get_deployment_packager() -> DeploymentPackager:
    global _packager
    if _packager is None:
        _packager = DeploymentPackager()
    return _packager
