"""
Search runner — drives the whole experiment loop:

  prepare run dir -> [propose -> gate -> execute -> review -> journal] x N -> report

V3 upgrades: parallel exploration, static gate, checkpoint/resume,
token budget, cross-run knowledge. See README.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Optional

from agent.execution import make_backend
from agent.search import data_preview as dp
from agent.search.agent import SearchAgent
from agent.search.config import SearchConfig
from agent.search.journal import Journal, Node
from agent.search.report import write_report
from agent.search.static_check import static_check


@dataclass
class SearchResult:
    run_id: str
    run_dir: str
    journal: Journal
    best: Optional[Node]
    steps_done: int
    wall_time: float

    @property
    def report_path(self) -> str:
        return os.path.join(self.run_dir, "report.md")

    @property
    def solution_path(self) -> str:
        return os.path.join(self.run_dir, "best_solution.py")


def _prepare_run(cfg: SearchConfig, data_dir: Optional[str]) -> tuple[str, str, str]:
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = os.path.join(cfg.runs_dir, run_id)
    workspace = os.path.join(run_dir, "workspace")
    os.makedirs(workspace, exist_ok=True)

    if data_dir:
        target = os.path.join(workspace, "input")
        if cfg.copy_data:
            shutil.copytree(data_dir, target, dirs_exist_ok=True)
        else:
            try:
                os.symlink(os.path.abspath(data_dir), target, target_is_directory=True)
            except OSError:
                shutil.copytree(data_dir, target, dirs_exist_ok=True)
    return run_id, run_dir, workspace


def _resume_run(cfg: SearchConfig, resume_run_id: str) -> tuple[str, str, str, Journal]:
    run_dir = os.path.join(cfg.runs_dir, resume_run_id)
    journal_path = os.path.join(run_dir, "journal.json")
    if not os.path.isfile(journal_path):
        raise FileNotFoundError(f"cannot resume: {journal_path} not found")
    workspace = os.path.join(run_dir, "workspace")
    os.makedirs(workspace, exist_ok=True)
    journal = Journal.load(journal_path)
    return resume_run_id, run_dir, workspace, journal


class _Budget:
    """Wall-clock + token budget, shared across workers."""

    def __init__(self, cfg: SearchConfig, agent: SearchAgent):
        self.cfg = cfg
        self.agent = agent
        self.start = time.time()
        self._tok0 = self._tokens_now()

    def _tokens_now(self) -> int:
        u1, u2 = self.agent.code_llm.total_usage, self.agent.feedback_llm.total_usage
        toks = u1.input_tokens + u1.output_tokens
        if u2 is not u1:
            toks += u2.input_tokens + u2.output_tokens
        return toks

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def tokens_used(self) -> int:
        return self._tokens_now() - self._tok0

    def exhausted(self) -> Optional[str]:
        if self.cfg.time_limit_secs and self.elapsed > self.cfg.time_limit_secs:
            return f"time budget reached ({self.cfg.time_limit_secs}s)"
        if self.cfg.token_budget and self.tokens_used() > self.cfg.token_budget:
            return f"token budget reached ({self.tokens_used():,}/{self.cfg.token_budget:,})"
        return None

    def node_timeout(self) -> int:
        timeout = self.cfg.exec_timeout
        if self.cfg.time_limit_secs:
            remaining = self.cfg.time_limit_secs - self.elapsed
            timeout = max(30, min(timeout, int(remaining)))
        return timeout


def _work_one(agent: SearchAgent, backend, budget: _Budget,
              stage: str, parent: Optional[Node], cfg: SearchConfig) -> Node:
    node = agent.propose(stage, parent)

    gate_error = static_check(node.code) if cfg.static_gate else None
    if gate_error:
        node.term_out = gate_error
        node.exit_code = 1
        node.exec_time = 0.0
        node.is_buggy = True
        node.analysis = gate_error.splitlines()[0]
        node.metric = None
        return node

    if node.code:
        res = backend.exec_python(node.code, timeout=budget.node_timeout())
        node.term_out = res.output
        node.exec_time = res.exec_time
        node.exit_code = res.exit_code
        node.timed_out = res.timed_out

    agent.review(node)
    return node


def run_search(
    task: str,
    data_dir: Optional[str] = None,
    config: Optional[SearchConfig] = None,
    evaluation_note: str = "",
    on_step: Optional[Callable[[Node, Journal], None]] = None,
    resume_run_id: Optional[str] = None,
) -> SearchResult:
    cfg = config or SearchConfig()

    if resume_run_id:
        run_id, run_dir, workspace, journal = _resume_run(cfg, resume_run_id)
        print(f"[search] resuming run {run_id} with {len(journal)} existing nodes")
    else:
        run_id, run_dir, workspace = _prepare_run(cfg, data_dir)
        journal = Journal()

    backend = make_backend(workspace)

    input_dir = os.path.join(workspace, "input")
    preview = dp.generate(input_dir) if os.path.isdir(input_dir) else ""

    knowledge_context = ""
    store = None
    if cfg.use_knowledge or cfg.reflect:
        try:
            from agent.knowledge import KnowledgeStore
            store = KnowledgeStore(cfg.knowledge_dir)
            if cfg.use_knowledge:
                knowledge_context = store.context_for_task(task)
        except Exception:  # noqa: BLE001
            store = None

    agent = SearchAgent(task, cfg, journal, data_preview=preview,
                        evaluation_note=evaluation_note,
                        knowledge_context=knowledge_context)

    workers = max(1, cfg.parallel_workers)
    print(f"[search] run {run_id} - backend={backend.name} - workers={workers} - "
          f"budget: {cfg.steps} steps"
          + (f" / {cfg.time_limit_secs}s" if cfg.time_limit_secs else "")
          + (f" / {cfg.token_budget:,} tokens" if cfg.token_budget else ""))

    budget = _Budget(cfg, agent)
    journal_path = os.path.join(run_dir, "journal.json")
    lock = threading.Lock()
    steps_done = 0
    target_nodes = len(journal) + cfg.steps

    def log_node(node: Node):
        status = ("BUGGY" if node.is_buggy
                  else f"metric={node.metric:.5g}")
        best = journal.best_node()
        best_s = f" - best={best.metric:.5g}" if best else ""
        print(f"[search] node {node.step + 1}/{target_nodes} [{node.stage}] "
              f"-> {status} ({node.exec_time:.1f}s){best_s}")

    try:
        if workers == 1:
            while len(journal) < target_nodes:
                reason = budget.exhausted()
                if reason:
                    print(f"[search] {reason} after {steps_done} steps")
                    break
                stage, parent = agent.choose_action()
                print(f"[search] step {len(journal) + 1}/{target_nodes} [{stage}]...")
                node = _work_one(agent, backend, budget, stage, parent, cfg)
                journal.append(node)
                steps_done += 1
                log_node(node)
                journal.save(journal_path)
                if on_step:
                    on_step(node, journal)
        else:
            in_flight: dict[Future, tuple[str, Optional[str]]] = {}
            scheduled = len(journal)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                def launch_until_full():
                    nonlocal scheduled
                    while (len(in_flight) < workers and scheduled < target_nodes
                           and not budget.exhausted()):
                        with lock:
                            reserved = frozenset(
                                pid for _, pid in in_flight.values() if pid)
                            pending_drafts = sum(
                                1 for s, _ in in_flight.values() if s == "draft")
                            stage, parent = agent.choose_action(
                                reserved=reserved, pending_drafts=pending_drafts)
                        fut = pool.submit(_work_one, agent, backend, budget,
                                          stage, parent, cfg)
                        in_flight[fut] = (stage, parent.id if parent else None)
                        scheduled += 1

                launch_until_full()
                while in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        in_flight.pop(fut)
                        try:
                            node = fut.result()
                        except Exception as e:  # noqa: BLE001
                            print(f"[search] worker failed: {e}")
                            continue
                        with lock:
                            journal.append(node)
                            steps_done += 1
                            journal.save(journal_path)
                        log_node(node)
                        if on_step:
                            on_step(node, journal)
                    reason = budget.exhausted()
                    if reason:
                        print(f"[search] {reason}; draining {len(in_flight)} in-flight node(s)")
                        for fut in list(in_flight):
                            try:
                                node = fut.result()
                                with lock:
                                    journal.append(node)
                                    steps_done += 1
                                    journal.save(journal_path)
                                log_node(node)
                            except Exception:  # noqa: BLE001
                                pass
                            in_flight.pop(fut)
                        break
                    launch_until_full()
    finally:
        backend.close()

    wall = budget.elapsed
    best = journal.best_node()
    if best:
        with open(os.path.join(run_dir, "best_solution.py"), "w", encoding="utf-8") as f:
            f.write(best.code)
    journal.save(journal_path)

    usage = agent.code_llm.total_usage.summary()
    write_report(run_dir, task, cfg, journal, wall, usage)

    if store is not None:
        summary = (f"best metric {best.metric} ({'lower' if best.lower_is_better else 'higher'} "
                   f"is better) after {len(journal)} attempts; winning approach: "
                   f"{best.plan[:400]}" if best
                   else f"no working solution in {len(journal)} attempts")
        store.index_run(run_id, task, summary,
                        code=best.code if best else "",
                        metric=best.metric if best else None)
        if cfg.reflect:
            from agent.knowledge import reflect_on_run
            lessons = reflect_on_run(task, journal, agent.feedback_llm, store, run_id)
            if lessons:
                print(f"[search] playbook updated with {len(lessons)} lesson(s)")

    print(f"[search] done in {wall:.0f}s - "
          + (f"best metric {best.metric:.5g} (node {best.step})" if best else "no working solution")
          + f" - report: {os.path.join(run_dir, 'report.md')}")

    return SearchResult(run_id=run_id, run_dir=run_dir, journal=journal,
                        best=best, steps_done=steps_done, wall_time=wall)
