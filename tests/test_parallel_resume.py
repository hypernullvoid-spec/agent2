"""
V3 runner features, end-to-end with mock LLMs and real subprocess execution:
  • parallel workers produce a valid journal with num_drafts respected
  • --resume continues a run from its journal.json
  • static gate scores broken code buggy without executing it
  • token budget stops the run early
"""

import os
import re
import tempfile

import numpy as np
import pandas as pd

from agent.llm.base import Usage
from agent.llm.mock_client import tool_response, text_response, MockLLMClient
from agent.llm.router import create_client
from agent.search import SearchConfig, run_search

WORKING = """plan: threshold classifier with holdout
```python
import numpy as np, pandas as pd
df = pd.read_csv('./input/train.csv')
X, y = df[['x1','x2']].values, df['y'].values
rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); cut = int(len(df)*0.8)
tr, va = idx[:cut], idx[cut:]
score = X[:,0] + X[:,1]
pred = (score[va] > np.median(score[tr])).astype(int)
print(f"Final Validation Metric: {(pred == y[va]).mean():.4f}")
```"""

# syntactically broken → the static gate must catch it without executing
GATED = "plan: oops\n```python\ndef broken(:\n    pass\n```"


def _make_dataset(dirpath: str):
    rng = np.random.default_rng(7)
    x1, x2 = rng.normal(size=300), rng.normal(size=300)
    y = ((x1 + x2) > 0).astype(int)
    pd.DataFrame({"x1": x1, "x2": x2, "y": y}).to_csv(
        os.path.join(dirpath, "train.csv"), index=False)


def _feedback(system, messages, tools):
    prompt = str(messages)
    m = re.search(r"Final Validation Metric:\s*([\d.]+)", prompt)
    metric = float(m.group(1)) if m else None
    return tool_response("submit_review", {
        "is_bug": metric is None, "summary": "reviewed",
        "metric": metric, "lower_is_better": False})


def _fresh_clients(tag: str):
    code = create_client(f"mock:{tag}-code")
    code.script, code.fallback = [], lambda *a: WORKING
    code.total_usage = Usage()
    fb = create_client(f"mock:{tag}-fb")
    fb.script, fb.fallback = [], _feedback
    fb.total_usage = Usage()
    return code, fb


def test_parallel_search_valid_journal():
    data_dir = tempfile.mkdtemp(prefix="swarn_par_data_")
    _make_dataset(data_dir)
    _fresh_clients("par")

    cfg = SearchConfig(
        steps=6, num_drafts=3, parallel_workers=3, exec_timeout=90,
        code_model="mock:par-code", feedback_model="mock:par-fb",
        runs_dir=tempfile.mkdtemp(prefix="swarn_par_runs_"),
        use_knowledge=False,
    )
    result = run_search("Predict y; metric accuracy.", data_dir=data_dir, config=cfg)

    assert result.steps_done == 6
    assert len(result.journal.draft_nodes) == 3, \
        "in-flight drafts must count toward num_drafts (no draft explosion)"
    assert result.best is not None and result.best.metric > 0.8
    # steps must be dense and unique even with concurrent appends
    steps = sorted(n.step for n in result.journal.nodes)
    assert steps == list(range(6))
    assert os.path.isfile(result.solution_path)


def test_resume_continues_run():
    data_dir = tempfile.mkdtemp(prefix="swarn_res_data_")
    _make_dataset(data_dir)
    _fresh_clients("res")
    runs_dir = tempfile.mkdtemp(prefix="swarn_res_runs_")

    cfg = SearchConfig(steps=2, num_drafts=2, exec_timeout=90,
                       code_model="mock:res-code", feedback_model="mock:res-fb",
                       runs_dir=runs_dir, use_knowledge=False)
    first = run_search("Predict y; metric accuracy.", data_dir=data_dir, config=cfg)
    assert len(first.journal) == 2

    second = run_search("Predict y; metric accuracy.", config=cfg,
                        resume_run_id=first.run_id)
    assert second.run_id == first.run_id
    assert len(second.journal) == 4, "resume adds cfg.steps MORE nodes"
    assert second.steps_done == 2


def test_static_gate_skips_execution():
    data_dir = tempfile.mkdtemp(prefix="swarn_gate_data_")
    _make_dataset(data_dir)
    code = create_client("mock:gate-code")
    code.script = [GATED, WORKING]
    code.fallback = lambda *a: WORKING
    code.total_usage = Usage()
    fb = create_client("mock:gate-fb")
    fb.script, fb.fallback = [], _feedback
    fb.total_usage = Usage()

    cfg = SearchConfig(steps=2, num_drafts=2, exec_timeout=90,
                       code_model="mock:gate-code", feedback_model="mock:gate-fb",
                       runs_dir=tempfile.mkdtemp(prefix="swarn_gate_runs_"),
                       use_knowledge=False)
    result = run_search("Predict y; metric accuracy.", data_dir=data_dir, config=cfg)

    gated = [n for n in result.journal.nodes if "StaticCheckError" in n.term_out]
    assert gated, "the broken draft must be gated"
    assert gated[0].exec_time == 0.0, "gated code must never reach the sandbox"
    assert gated[0].is_buggy
    # the reviewer LLM must NOT have been called for the gated node:
    # exactly one review call (for the working node)
    assert len(fb.calls) == 1


def test_token_budget_stops_early():
    data_dir = tempfile.mkdtemp(prefix="swarn_tok_data_")
    _make_dataset(data_dir)
    code = create_client("mock:tok-code")
    code.script, code.fallback = [], lambda *a: WORKING
    code.total_usage = Usage()
    # make every mock call cost 1000 tokens
    orig = code._call_api
    def costly(*a, **kw):
        r = orig(*a, **kw)
        r.usage = Usage(input_tokens=800, output_tokens=200, calls=1)
        return r
    code._call_api = costly
    fb = create_client("mock:tok-fb")
    fb.script, fb.fallback = [], _feedback
    fb.total_usage = Usage()

    cfg = SearchConfig(steps=50, num_drafts=2, exec_timeout=90,
                       token_budget=1500,
                       code_model="mock:tok-code", feedback_model="mock:tok-fb",
                       runs_dir=tempfile.mkdtemp(prefix="swarn_tok_runs_"),
                       use_knowledge=False)
    result = run_search("Predict y; metric accuracy.", data_dir=data_dir, config=cfg)
    assert result.steps_done < 50, "token budget must stop the run early"
