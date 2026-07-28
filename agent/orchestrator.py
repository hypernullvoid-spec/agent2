"""
Phase 11: Multi-Agent Orchestration

Coordinates the four roles defined in roles.py (Planner, Coder,
Reviewer, Tester) through a fixed pipeline:

    Planner → Coder → Reviewer ──approved──→ Tester ──pass──→ done
                 ↑                  │                 │
                 └─ needs_changes ──┘                 │
                 └───────────────── fail ──────────────┘

Each role is its own AgentLoop instance (own system prompt, own tool
subset, own SelfCorrectionPolicy, own Phase 5 Session) — orchestrator.py
does not touch agent_loop.py's control flow at all. It only:
  1. builds a role's AgentLoop from roles.get_role_config()
  2. calls run(task) and reads back {"outcome", "summary", "session_id"}
  3. decides the next role's task from that summary plus a small shared
     BlackboardState, and loops

Why a "blackboard", not a shared message history
─────────────────────────────────────────────────────
A multi-agent system could share one long message history across all
roles (everyone sees everyone's full conversation), or give each role
its own short-lived AgentLoop session and pass forward only a written
summary, the way a real engineering team passes forward a ticket
description and a PR description rather than a full chat transcript.
This codebase does the latter — it's why every role prompt above ends
with "call finish_task with a summary" and explicitly says reviewers/
testers "only see your summary... not your full trace." This keeps each
role's context window focused on its own job and keeps Phase 5's
session traces clean and independently inspectable (list_sessions will
show four separate sessions for one orchestrator run, not one giant
merged one) — but it does mean every role's summary needs to be useful
on its own, which is why the role prompts are explicit about exactly
what the summary must contain.

BlackboardState is the small amount of structured state that doesn't
fit naturally into a free-text summary: the running history of
role → outcome → summary, plus a revision counter, so the orchestrator
itself (not any one role) can decide things like "give up after 3 failed
review cycles" without the Coder or Reviewer needing to track that.

Failure handling
───────────────────
If the Reviewer says NEEDS_CHANGES or the Tester says FAIL, the
orchestrator loops back to the Coder with the reviewer/tester's verdict
folded into the next task description — not back to the Planner, since
re-planning from scratch on every revision would throw away a perfectly
good plan over a small implementation issue. MAX_REVISION_CYCLES caps
this loop so a stuck pipeline doesn't run forever, mirroring
agent_loop.py's own MAX_ITERATIONS guard at the single-agent level.
"""

from dataclasses import dataclass, field
from typing import Optional

from agent import ui
from agent.agent_loop import AgentLoop
from agent.llm import DEFAULT_MODEL  # deployed model (see agent/llm/router.py)
from agent.roles import get_role_config
from agent.self_correction import SelfCorrectionPolicy

MAX_REVISION_CYCLES = 3   # Coder ⇄ Reviewer/Tester loop cap, mirrors AgentLoop.MAX_ITERATIONS


@dataclass
class RoleRun:
    """One role's contribution to the pipeline — kept for the final report."""
    role:       str
    task:       str
    outcome:    str
    summary:    Optional[str]
    session_id: str


@dataclass
class BlackboardState:
    """
    The shared state every role's task gets built from. Deliberately
    small — just enough structure for the orchestrator to make routing
    decisions (approved vs needs_changes, revision count) without
    parsing free text. Everything else flows through each role's
    finish_task summary, by design (see module docstring).
    """
    original_task: str
    history:       list[RoleRun] = field(default_factory=list)
    revisions:     int = 0

    def last(self) -> Optional[RoleRun]:
        return self.history[-1] if self.history else None

    def summary_of(self, role: str) -> Optional[str]:
        """Most recent summary produced by a given role, if any."""
        for run in reversed(self.history):
            if run.role == role:
                return run.summary
        return None


class Orchestrator:
    """
    Drives the Planner → Coder → Reviewer → Tester pipeline.

    One Orchestrator instance is good for one top-level task. Each role
    gets a brand-new AgentLoop (and therefore a brand-new Phase 5
    Session and a brand-new Phase 4 SelfCorrectionPolicy) every time
    it's invoked — including re-invocations of the Coder during a
    revision cycle — so Phase 5's session history shows every attempt
    separately rather than one role's session being silently reused
    and overwritten.
    """

    def __init__(
        self,
        # NOTE: routed to the deployed endpoint (agent/llm/router.py) regardless.
        model: str = DEFAULT_MODEL,
        include_tester: bool = True,
        guardrail_policy: Optional["object"] = None,        # agent.observability.GuardrailPolicy
        observability_hooks: Optional["object"] = None,      # agent.observability.ObservabilityHooks
    ):
        self.model = model
        self.include_tester = include_tester   # set False to stop after Reviewer approval
        # Phase 15: shared across every role's AgentLoop in this run,
        # deliberately NOT re-created per role the way SelfCorrectionPolicy
        # is below — guardrail findings and trace spans are meaningful
        # aggregated across the whole multi-agent run (e.g. "did ANY role
        # in this pipeline encounter an injection attempt"), unlike Phase
        # 4's error budget, which is specifically per-attempt by design.
        self._guardrails = guardrail_policy
        self._observe = observability_hooks

    # ─────────────────────────────────────────────────── internals

    def _run_role(self, role: str, task: str, state: BlackboardState) -> RoleRun:
        """
        Build a fresh AgentLoop for `role`, run it on `task`, record the
        result on the blackboard, and return it. Building a new
        AgentLoop (rather than reusing one) is what gives each role —
        and each retry of the same role — its own isolated Phase 4
        error budget and its own Phase 5 session, matching how a human
        team's retry of a step is a new attempt, not a continuation of
        the failed one. Phase 15's guardrails/observability, by
        contrast, ARE shared across every fresh AgentLoop this method
        builds — see __init__'s note on why that's a deliberate
        asymmetry with Phase 4's per-attempt policy.
        """
        config = get_role_config(role)
        loop = AgentLoop(
            model               = self.model,
            correction_policy   = SelfCorrectionPolicy(),
            system_prompt       = config["system_prompt"],
            tool_names          = config["tool_names"],
            role_name           = role,
            guardrail_policy    = self._guardrails,
            observability_hooks = self._observe,
        )
        result = loop.run(task)
        run = RoleRun(
            role       = role,
            task       = task,
            outcome    = result["outcome"],
            summary    = result["summary"],
            session_id = result["session_id"],
        )
        state.history.append(run)
        return run

    @staticmethod
    def _verdict_is_approval(summary: Optional[str]) -> bool:
        """
        Reviewer/Tester are asked to lead their summary with APPROVED/
        NEEDS_CHANGES or PASS/FAIL. This is intentionally a simple
        substring check, not an attempt to parse free-form judgment —
        the role prompts are written so the agent puts the verdict
        keyword in deliberately, the same way a CI system looks for an
        explicit exit code rather than trying to infer pass/fail from
        log text.
        """
        if not summary:
            return False
        upper = summary.upper()
        if "NEEDS_CHANGES" in upper or "FAIL" in upper:
            return False
        return "APPROVED" in upper or "PASS" in upper

    # ─────────────────────────────────────────────────── public entry point

    def run(self, task: str) -> dict:
        """
        Execute the full pipeline for one top-level task. Returns a
        report dict: {"final_outcome", "state": BlackboardState,
        "report_markdown"} — the markdown is meant to be written to a
        file or printed directly, the same way Phase 5's
        Session.to_markdown() is meant to be read by a person, not
        parsed by code.
        """
        state = BlackboardState(original_task=task)

        ui.section("team pipeline — Planner → Coder → Reviewer → Tester", subtitle=task)

        # ── 1. Planner ───────────────────────────────────────────────
        plan_run = self._run_role("planner", task, state)
        if plan_run.outcome != "complete" or not plan_run.summary:
            return self._finish(state, "planner_failed")

        # ── 2. Coder ⇄ Reviewer ⇄ Tester, with revision loop ──────────
        coder_task = (
            f"Original task: {task}\n\n"
            f"Plan from the Planner:\n{plan_run.summary}\n\n"
            f"Execute this plan."
        )

        while True:
            coder_run = self._run_role("coder", coder_task, state)
            if coder_run.outcome != "complete" or not coder_run.summary:
                return self._finish(state, "coder_failed")

            reviewer_task = (
                f"Original task: {task}\n\n"
                f"The Coder reports the following work was completed:\n"
                f"{coder_run.summary}\n\n"
                f"Independently verify this is correct and complete."
            )
            reviewer_run = self._run_role("reviewer", reviewer_task, state)
            if reviewer_run.outcome != "complete" or not reviewer_run.summary:
                return self._finish(state, "reviewer_failed")

            if not self._verdict_is_approval(reviewer_run.summary):
                state.revisions += 1
                if state.revisions >= MAX_REVISION_CYCLES:
                    return self._finish(state, "max_revisions_reached")
                ui.warn("revision", "Reviewer requested changes — sending back to Coder "
                        f"({state.revisions}/{MAX_REVISION_CYCLES})")
                coder_task = (
                    f"Original task: {task}\n\n"
                    f"Your previous summary:\n{coder_run.summary}\n\n"
                    f"The Reviewer found issues:\n{reviewer_run.summary}\n\n"
                    f"Fix these specific issues."
                )
                continue   # back to Coder with reviewer feedback

            # Reviewer approved — stop here unless a Tester stage is configured
            if not self.include_tester:
                return self._finish(state, "approved_no_tester")

            tester_task = (
                f"Original task: {task}\n\n"
                f"The Coder completed this work:\n{coder_run.summary}\n\n"
                f"The Reviewer approved it:\n{reviewer_run.summary}\n\n"
                f"Run/verify the actual result."
            )
            tester_run = self._run_role("tester", tester_task, state)
            if tester_run.outcome != "complete" or not tester_run.summary:
                return self._finish(state, "tester_failed")

            if self._verdict_is_approval(tester_run.summary):
                return self._finish(state, "complete")

            state.revisions += 1
            if state.revisions >= MAX_REVISION_CYCLES:
                return self._finish(state, "max_revisions_reached")
            ui.warn("revision", "Tester reported failure — sending back to Coder "
                    f"({state.revisions}/{MAX_REVISION_CYCLES})")
            coder_task = (
                f"Original task: {task}\n\n"
                f"Your previous summary:\n{coder_run.summary}\n\n"
                f"The Tester found a failure:\n{tester_run.summary}\n\n"
                f"Fix this and ensure the actual run succeeds."
            )
            # back to Coder with tester feedback → re-review → re-test

    # ─────────────────────────────────────────────────── reporting

    def _finish(self, state: BlackboardState, final_outcome: str) -> dict:
        report = self._render_report(state, final_outcome)
        ui.section(f"team pipeline — finished: {final_outcome}",
                   style="green" if final_outcome == "complete" else "red")
        return {
            "final_outcome":  final_outcome,
            "state":          state,
            "report_markdown": report,
        }

    def _render_report(self, state: BlackboardState, final_outcome: str) -> str:
        lines = [
            f"# Multi-agent run report",
            "",
            f"**Task:** {state.original_task}",
            f"**Final outcome:** {final_outcome}",
            f"**Revision cycles used:** {state.revisions}/{MAX_REVISION_CYCLES}",
            "",
            "## Timeline",
        ]
        for i, run in enumerate(state.history, 1):
            lines += [
                "",
                f"### {i}. {run.role.upper()}  (session `{run.session_id[:8]}`, outcome: {run.outcome})",
                "",
                run.summary or "_(no summary produced)_",
            ]
        return "\n".join(lines)
