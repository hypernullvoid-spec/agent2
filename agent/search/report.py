"""Markdown run report — the artifact a human actually reads."""

from __future__ import annotations

import os
import time

from agent.search.config import SearchConfig
from agent.search.journal import Journal


def write_report(run_dir: str, task: str, cfg: SearchConfig,
                 journal: Journal, wall_time: float, usage: str) -> str:
    best = journal.best_node()
    good = journal.good_nodes
    buggy = [n for n in journal.nodes if n.is_buggy]

    lines = [
        f"# Swarn run report — {os.path.basename(run_dir)}",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Task",
        task.strip(),
        "",
        "## Outcome",
    ]
    if best:
        direction = "lower is better" if best.lower_is_better else "higher is better"
        lines += [
            f"**Best validation metric: `{best.metric:.6g}`** ({direction}), "
            f"found at step {best.step} via `{best.stage}`.",
            "",
            f"Solution saved to `best_solution.py`.",
        ]
    else:
        lines.append("**No working solution found.** See failed attempts below.")

    lines += [
        "",
        "## Search statistics",
        "",
        f"- Steps executed: {len(journal)} ({len(good)} good, {len(buggy)} buggy)",
        f"- Wall time: {wall_time:.0f}s",
        f"- Models: code=`{cfg.code_model}`, feedback=`{cfg.feedback_model}`",
        f"- LLM usage: {usage}",
        "",
        "## Solution tree",
        "",
        "```",
        journal.render_tree(),
        "```",
        "",
        "## Metric history (good nodes)",
        "",
        "| step | stage | metric | exec time |",
        "|-----:|-------|-------:|----------:|",
    ]
    for n in good:
        lines.append(f"| {n.step} | {n.stage} | {n.metric:.6g} | {n.exec_time:.1f}s |")

    if best:
        lines += ["", "## Best solution plan", "", best.plan or "(none recorded)",
                  "", "## Best solution analysis", "", best.analysis or "(none recorded)"]

    failed_tail = buggy[-5:]
    if failed_tail:
        lines += ["", "## Recent failures (what the search learned to avoid)", ""]
        for n in failed_tail:
            lines.append(f"- step {n.step} [{n.stage}]: {(n.analysis or n.term_out[-160:])[:240]}")

    path = os.path.join(run_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
