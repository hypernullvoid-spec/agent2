"""Journal: tree structure, best-node selection, persistence."""

import os
import tempfile

from agent.search.journal import Journal, Node


def _node(**kw) -> Node:
    return Node(**kw)


def test_append_links_parent_and_children():
    j = Journal()
    root = j.append(_node(stage="draft"))
    child = j.append(_node(stage="debug", parent_id=root.id))
    assert child.id in j.get(root.id).children
    assert j.draft_nodes == [root]
    assert child.step == 1


def test_best_node_higher_is_better():
    j = Journal()
    a = j.append(_node()); a.is_buggy, a.metric = False, 0.80
    b = j.append(_node()); b.is_buggy, b.metric = False, 0.91
    c = j.append(_node()); c.is_buggy = True  # never counted
    assert j.best_node().id == b.id


def test_best_node_lower_is_better():
    j = Journal()
    a = j.append(_node()); a.is_buggy, a.metric, a.lower_is_better = False, 0.42, True
    b = j.append(_node()); b.is_buggy, b.metric, b.lower_is_better = False, 0.13, True
    assert j.best_node().id == b.id


def test_buggy_leaves_excludes_fixed_branches():
    j = Journal()
    root = j.append(_node())                     # buggy root...
    j.append(_node(stage="debug", parent_id=root.id))  # ...already has a fix attempt
    lonely = j.append(_node())                   # buggy, still a leaf
    assert [n.id for n in j.buggy_leaves if n.id != root.id] == [lonely.id] or \
           lonely.id in [n.id for n in j.buggy_leaves]
    assert root.id not in [n.id for n in j.buggy_leaves]


def test_debug_depth():
    j = Journal()
    root = j.append(_node(stage="draft"))
    d1 = j.append(_node(stage="debug", parent_id=root.id))
    d2 = j.append(_node(stage="debug", parent_id=d1.id))
    assert d2.debug_depth(j) == 2 and root.debug_depth(j) == 0


def test_save_load_roundtrip():
    j = Journal()
    n = j.append(_node(plan="p", code="print(1)"))
    n.metric, n.is_buggy = 0.5, False
    path = os.path.join(tempfile.mkdtemp(), "journal.json")
    j.save(path)
    j2 = Journal.load(path)
    assert len(j2) == 1 and j2.nodes[0].metric == 0.5 and not j2.nodes[0].is_buggy


def test_render_tree_marks_best():
    j = Journal()
    a = j.append(_node()); a.is_buggy, a.metric = False, 0.9
    tree = j.render_tree()
    assert "★" in tree and "metric=0.9" in tree


def test_summarize_mentions_good_and_failed():
    j = Journal()
    g = j.append(_node(plan="use xgboost")); g.is_buggy, g.metric = False, 0.88
    f = j.append(_node(plan="bad")); f.analysis = "ImportError: no such module"
    s = j.summarize()
    assert "0.88" in s and "FAILED" in s
