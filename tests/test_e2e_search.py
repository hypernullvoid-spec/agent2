"""
End-to-end search run — no network, no API keys.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from agent.llm.mock_client import tool_response
from agent.llm.router import create_client
from agent.search import SearchConfig, run_search

BROKEN = "plan: quick baseline\n```python\nimport nonexistent_module_xyz\n```"

WORKING = """plan: mean-threshold classifier with a proper holdout split
```python
import numpy as np, pandas as pd
df = pd.read_csv('./input/train.csv')
X, y = df[['x1','x2']].values, df['y'].values
rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); cut = int(len(df)*0.8)
tr, va = idx[:cut], idx[cut:]
score = X[:,0] + X[:,1]
thresh = np.median(score[tr])
pred = (score[va] > thresh).astype(int)
acc = (pred == y[va]).mean()
print(f"Final Validation Metric: {acc:.4f}")
```"""


def _make_dataset(dirpath: str):
    rng = np.random.default_rng(42)
    x1, x2 = rng.normal(size=400), rng.normal(size=400)
    y = ((x1 + x2) > 0).astype(int)
    pd.DataFrame({"x1": x1, "x2": x2, "y": y}).to_csv(
        os.path.join(dirpath, "train.csv"), index=False)


def _feedback(system, messages, tools):
    prompt = str(messages)
    if "Final Validation Metric" in prompt:
        import re
        m = re.search(r"Final Validation Metric:\s*([\d.]+)", prompt)
        metric = float(m.group(1)) if m else None
        return tool_response("submit_review", {
            "is_bug": metric is None, "summary": "ran and reported a metric",
            "metric": metric, "lower_is_better": False})
    return tool_response("submit_review", {
        "is_bug": True, "summary": "import failed", "metric": None,
        "lower_is_better": False})


def test_full_search_produces_artifacts():
    data_dir = tempfile.mkdtemp(prefix="swarn_e2e_data_")
    _make_dataset(data_dir)

    code = create_client("mock:e2e-code")
    code.script = [BROKEN, WORKING, WORKING]
    code.fallback = lambda *a: WORKING
    fb = create_client("mock:e2e-fb")
    fb.fallback = _feedback

    cfg = SearchConfig(
        steps=3, num_drafts=2, debug_prob=0.0, exec_timeout=90,
        code_model="mock:e2e-code", feedback_model="mock:e2e-fb",
        runs_dir=tempfile.mkdtemp(prefix="swarn_e2e_runs_"),
        use_knowledge=False,  # keep the test hermetic
    )
    result = run_search("Predict y from x1,x2; metric: accuracy.",
                        data_dir=data_dir, config=cfg)

    assert result.steps_done == 3
    assert result.best is not None, "search should find the working solution"
    assert result.best.metric and result.best.metric > 0.8
    assert os.path.isfile(result.solution_path)
    assert os.path.isfile(result.report_path)
    assert os.path.isfile(os.path.join(result.run_dir, "journal.json"))

    report = open(result.report_path, encoding="utf-8").read()
    assert "Best validation metric" in report and "Solution tree" in report

    journal = result.journal
    assert any(n.is_buggy for n in journal.nodes), "the broken draft must be recorded"
    assert len(journal.good_nodes) >= 1
