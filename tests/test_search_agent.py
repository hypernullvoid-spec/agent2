"""SearchAgent: code extraction, policy transitions, review parsing."""

from agent.llm.mock_client import tool_response
from agent.llm.router import create_client
from agent.search.agent import SearchAgent, extract_code, extract_metric_fallback
from agent.search.config import SearchConfig
from agent.search.journal import Journal, Node


def _agent(journal=None, **cfg_kw):
    cfg = SearchConfig(code_model="mock:code", feedback_model="mock:fb",
                       num_drafts=2, max_debug_depth=2, **cfg_kw)
    return SearchAgent("predict y from x", cfg, journal or Journal())


def test_extract_code_takes_longest_block():
    text = "plan\n```python\nx=1\n```\nmore\n```python\nimport pandas\nprint('full solution')\n```"
    assert "full solution" in extract_code(text)
    assert extract_code("no code here") == ""


def test_extract_metric_fallback():
    out = "epoch 1\nFinal Validation Metric: 0.8734\ndone"
    assert extract_metric_fallback(out) == 0.8734
    assert extract_metric_fallback("Final Validation Metric: 1e-3") == 0.001
    assert extract_metric_fallback("nothing") is None


def test_policy_drafts_until_quota():
    a = _agent()
    stage, parent = a.choose_action()
    assert stage == "draft" and parent is None


def test_policy_improves_best_when_no_debuggable():
    j = Journal()
    for metric in (0.7, 0.9):
        n = j.append(Node())
        n.is_buggy, n.metric = False, metric
    a = _agent(journal=j, debug_prob=0.0, improve_topk=1)
    stage, parent = a.choose_action()
    assert stage == "improve" and parent.metric == 0.9


def test_policy_improve_topk_stays_within_top_k():
    j = Journal()
    for metric in (0.5, 0.7, 0.9):
        n = j.append(Node())
        n.is_buggy, n.metric = False, metric
    a = _agent(journal=j, debug_prob=0.0, improve_topk=2)
    for _ in range(25):
        stage, parent = a.choose_action()
        assert stage == "improve" and parent.metric in (0.9, 0.7), \
            "epsilon-greedy improve must never pick outside the top-k"


def test_policy_reservations_for_parallel_scheduler():
    j = Journal()
    b1 = j.append(Node()); b2 = j.append(Node())   # two buggy drafts (quota met)
    a = _agent(journal=j, debug_prob=1.0)
    stage, parent = a.choose_action(reserved=frozenset({b1.id}))
    assert stage == "debug" and parent.id == b2.id, \
        "a leaf reserved by another worker must not be handed out twice"
    # in-flight drafts count toward the quota: 1 real + 1 pending == num_drafts,
    # so the policy must debug the existing buggy draft instead of over-drafting
    j2 = Journal()
    j2.append(Node())
    a2 = _agent(journal=j2, debug_prob=1.0)
    stage, _ = a2.choose_action(pending_drafts=1)
    assert stage == "debug"


def test_policy_debugs_buggy_leaf():
    j = Journal()
    j.append(Node()); j.append(Node())          # two buggy drafts (quota met)
    a = _agent(journal=j, debug_prob=1.0)
    stage, parent = a.choose_action()
    assert stage == "debug" and parent is not None


def test_policy_respects_max_debug_depth():
    j = Journal()
    root = j.append(Node(stage="draft"))
    d1 = j.append(Node(stage="debug", parent_id=root.id))
    d2 = j.append(Node(stage="debug", parent_id=d1.id))   # depth 2 == max
    j.append(Node(stage="draft"))                          # second draft (quota met)
    a = _agent(journal=j, debug_prob=1.0)
    stage, parent = a.choose_action()
    # d2 is too deep; the only debuggable leaf is the fresh draft
    assert stage == "debug" and parent.id != d2.id


def test_review_uses_forced_tool_call():
    a = _agent()
    fb = create_client("mock:fb")
    fb.script.append(tool_response("submit_review", {
        "is_bug": False, "summary": "solid run", "metric": 0.91, "lower_is_better": False,
    }))
    node = Node(code="print('x')", term_out="Final Validation Metric: 0.91", exit_code=0)
    a.review(node)
    assert not node.is_buggy and node.metric == 0.91 and "solid" in node.analysis


def test_review_marks_timeout_buggy_without_llm():
    a = _agent()
    node = Node(code="while True: pass", timed_out=True, exec_time=600)
    a.review(node)
    assert node.is_buggy and node.metric is None


def test_review_printed_metric_wins():
    a = _agent()
    fb = create_client("mock:fb")
    fb.script.append(tool_response("submit_review", {
        "is_bug": False, "summary": "ok", "metric": 0.5, "lower_is_better": False,
    }))
    node = Node(code="c", term_out="Final Validation Metric: 0.77", exit_code=0)
    a.review(node)
    assert node.metric == 0.77  # trust the script's own print over the reviewer
