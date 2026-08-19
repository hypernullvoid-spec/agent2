"""
Core agent loop — Phase 1 foundation, Phase 4 + Phase 5 integrated.

What changed from Phase 1
──────────────────────────
  Phase 4  SelfCorrectionPolicy (optional, passed from main.py)
           • assess() is called after every tool result
           • enriches the tool_result sent to Claude with a diagnostic hint
           • tracks consecutive errors; aborts when the budget runs out

  Phase 5  SessionStore replaces the old flat trace list + runs/*.json
           • each run() creates a Session and logs every step as a typed
             StepKind record (PLAN / TOOL_CALL / TOOL_RESULT / CORRECTION /
             COMPLETE / ERROR)
           • on exit, saves sessions/<id>/trace.json + summary.md and
             updates sessions/index.json

  Phase 11 AgentLoop is now reusable per-role, not just as the single
           top-level loop main.py drives directly:
           • system_prompt and tool_names are now optional constructor
             args — pass a role-specific prompt + a restricted tool
             subset and you get an isolated single-role agent
           • run() now returns the session's outcome/summary instead of
             only printing, so a caller (orchestrator.py) can read the
             result without scraping stdout or reaching into private
             session internals

  Phase 15 GuardrailPolicy + ObservabilityHooks (both optional, passed
           from main.py/orchestrator.py, same pattern as Phase 4's
           correction_policy)

  V3       Doom-loop detection + deterministic context compaction:
           • every tool call is recorded with a result-hash-aware
             signature; same-call-same-result repetition triggers a
             corrective note appended to the tool result
           • once the conversation crosses a char budget, old tool
             results are truncated head+tail (the last few turns are
             always kept verbatim) — no extra LLM call, fully testable
           • MAX_ITERATIONS is env-configurable (SWARN_MAX_ITERATIONS)

Everything else (the ReAct loop, tool dispatch, Claude API call) is
identical to Phase 1. agent_loop.py is the one file that deliberately
evolves as new phases add to it.
"""

import os

from agent.utils            import ui
from agent.llm            import DEFAULT_MODEL  # deployed model (see agent/llm/router.py)
from agent.llm.llm_client     import LLMClient
from agent.runtime.tools          import get_tool_definitions, run_tool
from agent.messaging.prompts        import SYSTEM_PROMPT
from agent.memory.memory         import get_session_store, Session, StepKind
from agent.core.self_correction import SelfCorrectionPolicy
from agent.core.doom_loop       import DoomLoopDetector, WARNING as DOOM_WARNING

from typing import Callable, Optional

from agent.config import MAX_ITERATIONS, CONTEXT_CHAR_BUDGET
from agent.core.approval_policy import requires_approval

# How many steps from the cap to start telling the model to wrap up. Hitting the
# cap yields NOTHING — no summary, no answer — however much useful work was done,
# so the model needs warning while it still has calls left to write one.
BUDGET_WARN_AT = int(os.environ.get("SWARN_BUDGET_WARN_AT", "8"))
BUDGET_FINAL_AT = int(os.environ.get("SWARN_BUDGET_FINAL_AT", "3"))


def _review_summary(summary: str) -> list:
    """Check the final summary against the evidence actually gathered."""
    try:
        from agent.data_analysis import review_conclusions
        return review_conclusions(summary or "")
    except Exception:  # noqa: BLE001 — never block finishing on a checker fault
        return []


def _budget_notice(step_num: int, cap: int) -> str:
    """A note appended to the last tool result as the step budget runs down."""
    left = cap - step_num
    if left > BUDGET_WARN_AT or left <= 0:
        return ""
    if left <= BUDGET_FINAL_AT:
        return (f"\n\n[BUDGET] {left} tool call(s) left before this run is cut off with NO "
                f"answer delivered. Stop gathering data. Call finish_task NOW with a summary "
                f"of what you already found — an incomplete summary is worth far more than "
                f"none.")
    return (f"\n\n[BUDGET] {left} of {cap} tool call(s) remaining. Start consolidating: do only "
            f"what materially changes your conclusion, then call finish_task while you still "
            f"can. Reaching the cap delivers nothing to the user.")

# Context compaction (V3): long runs accumulate huge tool results in the
# message history. When the total conversation size crosses this budget,
# old tool results are truncated head+tail — deterministic, no extra LLM
# call, and the last few turns are always kept verbatim.
_KEEP_RECENT_MESSAGES = 6
_TRUNC_HEAD, _TRUNC_TAIL = 700, 500


def _message_chars(messages: list) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    total += len(str(b.get("content", b.get("text", ""))))
                else:
                    total += len(getattr(b, "text", "") or "")
    return total


def compact_messages(messages: list) -> int:
    """Truncate old tool results in place once the conversation exceeds the
    char budget. Returns the number of results truncated."""
    if _message_chars(messages) <= CONTEXT_CHAR_BUDGET:
        return 0
    truncated = 0
    for m in messages[:-_KEEP_RECENT_MESSAGES]:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                text = str(b.get("content", ""))
                if len(text) > _TRUNC_HEAD + _TRUNC_TAIL + 100:
                    b["content"] = (text[:_TRUNC_HEAD]
                                    + f"\n… [{len(text) - _TRUNC_HEAD - _TRUNC_TAIL} chars compacted] …\n"
                                    + text[-_TRUNC_TAIL:])
                    truncated += 1
    return truncated


class AgentLoop:
    """
    The core ReAct loop:
      for each iteration:
        1. Call the LLM with the conversation + tool definitions
        2. Log + display any reasoning text (PLAN)
        3. For each tool_use block:
           a. Log the TOOL_CALL
           b. Execute via run_tool()
           c. Run through correction policy (Phase 4)
           d. Log the TOOL_RESULT (and CORRECTION if applicable)
           e. If finish_task → log COMPLETE and exit
           f. If abort signal from policy → log ERROR and exit
        4. Append tool_results to messages and loop
      On max_iterations → log ERROR and exit

    Phase 11 note: this class is also the building block for multi-agent
    roles — orchestrator.py creates one AgentLoop per role with its own
    system_prompt and a restricted tool_names subset.
    """

    def __init__(
        self,
        # NOTE: all LLM calls are hard-routed to the deployed endpoint
        # configured in agent/llm/router.py — this param is display/log only.
        model:               str                           = DEFAULT_MODEL,
        correction_policy:   Optional[SelfCorrectionPolicy] = None,
        system_prompt:       Optional[str]                  = None,
        tool_names:          Optional[list[str]]             = None,
        role_name:           Optional[str]                   = None,
        guardrail_policy:    Optional["object"]              = None,   # agent.observability.observability.GuardrailPolicy
        observability_hooks: Optional["object"]              = None,   # agent.observability.observability.ObservabilityHooks
        on_tool_result:      Optional[Callable[[str, dict, str], None]] = None,
        # Interactive mode passes a callback taking (tool_name, tool_input)
        # and returning True to run the call, False to refuse it. None — the
        # default, and what headless mode uses — auto-approves everything,
        # which is what makes a headless run scriptable: a prompt on stdin
        # nobody is watching would simply hang.
        approval_callback:   Optional[Callable[[str, dict], bool]] = None,
        # Per-instance override of the SWARN_MAX_ITERATIONS default, so
        # --max-iterations can be an ordinary CLI flag.
        max_iterations:      Optional[int]                   = None,
        # When True, run() carries the message history across calls — that is
        # what makes the REPL a conversation rather than a series of
        # unrelated one-shots, so "now add tests for that" resolves against
        # the previous turn.
        keep_history:        bool                            = False,
        # When True, every run() records into ONE session for the life of this
        # AgentLoop instead of opening a new one per call — one interactive
        # sitting, one session. Kept separate from keep_history because they
        # answer different questions (what the model sees vs. what gets
        # recorded); the REPL happens to want both.
        single_session:      bool                            = False,
    ):
        self.llm           = LLMClient(model=model)
        self.model         = model
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        # tool_names (the allow-list filter) is stored, not resolved into a
        # fixed tool-definitions list here: Phase 12's connect_mcp_server can
        # register brand-new tools into TOOL_REGISTRY *during* a run, so
        # get_tool_definitions(self._tool_names) is called fresh on every
        # loop iteration (see run() below) — a tool registered mid-run
        # becomes visible on the very next LLM call.
        self._tool_names   = tool_names
        self._policy       = correction_policy
        self._guardrails   = guardrail_policy
        self._observe      = observability_hooks
        self._store        = get_session_store()
        self.role_name     = role_name   # purely cosmetic — prefixes log lines, e.g. "[planner]"
        # Called with (tool_name, tool_input, raw_result) after every tool
        # call, before the model sees the result. The CLI uses it to print a
        # tool's own grounded output (doc_qa's verified evidence, say) instead
        # of leaving only the agent's later paraphrase on screen. Given the
        # RAW result deliberately — self-correction hints and guardrail
        # banners are messages to the model, not to the reader.
        self._on_tool_result = on_tool_result
        self._approve       = approval_callback
        self.max_iterations = max_iterations or MAX_ITERATIONS
        self.keep_history   = keep_history
        self.single_session = single_session
        # Carried across run() calls only when keep_history is set.
        self.history: list = []
        # The sitting's Session when single_session is set; None otherwise and
        # between sittings. See _acquire_session().
        self._session = None

    # ─────────────────────────────────────────────────── session management

    def _acquire_session(self, task: str):
        """Return the Session this run() should record into.

        Default (headless, one task per process): a fresh session per run,
        which is what a scripted invocation wants — one command, one trace,
        one exit code.

        single_session (the interactive REPL): the whole sitting is ONE
        session, and each question becomes a USER_TURN step inside it. A REPL
        conversation is one continuous piece of work — the follow-ups only
        make sense against what came before, and `keep_history` already feeds
        them to the model that way — so splitting it into one session per
        question produced a history full of fragments that could not be read
        on their own. New REPL, new session.
        """
        if not self.single_session:
            return self._store.new_session(task=task, model=self.model)

        if self._session is None:
            self._session = self._store.new_session(task=task, model=self.model)
            return self._session

        # A continuing sitting: mark the turn boundary so the trace can be
        # read back as a conversation rather than one long run of tool calls.
        self._session.add_step(
            StepKind.USER_TURN,
            task=task,
            turn=len(self._session.user_turns()) + 1,
        )
        # Each turn reopens the session: outcome/summary describe the turn in
        # flight, and a stale "complete" from the previous one would otherwise
        # persist if this turn died before setting its own.
        self._session.outcome = None
        self._session.summary = None
        return self._session

    def close_session(self) -> None:
        """Finalize the current sitting's session (interactive shutdown, /new).

        Only meaningful with single_session: _finish() closes the session
        itself in the default one-run-per-session mode. Safe to call when
        there is nothing open, so the REPL's exit paths need not check.
        """
        if self._session is None:
            return
        session, self._session = self._session, None
        # A sitting that ended without the last turn setting an outcome (the
        # user just typed /quit) is neither a success nor a failure.
        if session.outcome is None:
            session.outcome = "ended"
        self._store.close_session(session)

    # ───────────────────────────────────────────── conversation management

    def reset_conversation(self) -> None:
        """Drop the carried-over message history (interactive `/new`)."""
        self.history = []

    def compact_conversation(self) -> tuple[int, int]:
        """Compact the carried-over history in place (interactive `/compact`).

        Unlike the automatic pass in run(), this ignores CONTEXT_CHAR_BUDGET
        and always compacts: the user asked for it explicitly. Returns
        (chars_before, chars_after) so the caller can report the saving.
        """
        before = _message_chars(self.history)
        for message in self.history[:-_KEEP_RECENT_MESSAGES]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                text = str(block.get("content", ""))
                if len(text) > _TRUNC_HEAD + _TRUNC_TAIL + 100:
                    block["content"] = (
                        text[:_TRUNC_HEAD]
                        + f"\n… [{len(text) - _TRUNC_HEAD - _TRUNC_TAIL} chars compacted] …\n"
                        + text[-_TRUNC_TAIL:]
                    )
        return before, _message_chars(self.history)

    def _finish(self, session, messages: list) -> dict:
        """Close the session and hand back the result.

        keep_history is applied HERE rather than at each return site so every
        exit path — completion, abort, cancellation, iteration cap — leaves
        the conversation in the same consistent state.
        """
        if self.keep_history:
            self.history = messages
        if self.single_session:
            # The sitting is still open — the next question belongs in this
            # same session. Checkpoint so the turn survives a crash, and let
            # close_session() finalize when the REPL exits.
            self._store.checkpoint_session(session)
        else:
            self._store.close_session(session)
        return {"outcome": session.outcome, "summary": session.summary,
                "session_id": session.id}

    # ─────────────────────────────────────────────────── public entry point

    def run(self, task: str, stop_event=None, on_session=None) -> dict:
        """
        Run the ReAct loop to completion (or until aborted/iteration-capped).

        Returns a small dict describing what happened:
          {"outcome": str, "summary": str | None, "session_id": str}
        so a caller — main.py for single-agent use, or Phase 11's
        Orchestrator for multi-role use — can branch on the result
        without reaching into Session internals or parsing stdout.

        stop_event: optional threading.Event — checked between iterations;
        when set, the run stops with outcome "cancelled" (the current LLM
        call / tool batch finishes first; nothing is interrupted mid-flight).
        on_session: optional callback invoked with the Session right after
        it's created, so an async caller (the dashboard's job registry) can
        learn the session id long before run() returns.
        """
        session  = self._acquire_session(task)
        # [frontend-api] NEW: report the session id to the caller immediately.
        # The dashboard's job registry passes on_session so the web frontend
        # can match this run's live websocket steps to its job id right away,
        # instead of waiting until run() returns at the very end.
        if on_session is not None:
            try:
                on_session(session)
            except Exception:  # noqa: BLE001 — an observer must not kill the run
                pass
        # keep_history: earlier turns stay in the prompt, so a follow-up like
        # "now add tests for it" resolves against what just happened.
        messages = ([*self.history] if self.keep_history else []) + [
            {"role": "user", "content": task}
        ]
        doom     = DoomLoopDetector()

        ui.session_header(session.id, task, role=self.role_name)

        summary_revised = False
        try:
            from agent.data_analysis import reset_evidence
            reset_evidence()
        except Exception:  # noqa: BLE001
            pass

        for step_num in range(1, self.max_iterations + 1):

            # [frontend-api] NEW: cooperative cancellation (dashboard job API).
            # POST /api/jobs/{id}/cancel sets this threading.Event; we check it
            # once per loop iteration, so the run stops cleanly BETWEEN steps —
            # the in-flight LLM call / tool batch is never killed halfway.
            if stop_event is not None and stop_event.is_set():
                ui.warn("cancelled", "stop requested — ending run before next step")
                session.outcome = "cancelled"
                session.add_step(StepKind.ERROR, reason="cancelled_by_user", step=step_num)
                return self._finish(session, messages)

            # ── V3: context compaction ──────────────────────────────
            n_compacted = compact_messages(messages)
            if n_compacted:
                ui.info(f"context: compacted {n_compacted} old tool result(s)")

            # ── LLM call ────────────────────────────────────────────
            current_tools = get_tool_definitions(self._tool_names)
            # The spinner covers the one part of a step with nothing to
            # show: waiting on the model. It clears itself before any
            # reasoning/tool output is printed below.
            with ui.thinking(role=self.role_name):
                if self._observe:
                    with self._observe.llm_call_span(step_num, self.model):
                        response = self.llm.call(
                            system   = self.system_prompt,
                            messages = messages,
                            tools    = current_tools,
                        )
                else:
                    response = self.llm.call(
                        system   = self.system_prompt,
                        messages = messages,
                        tools    = current_tools,
                    )

            # ── reasoning text (PLAN) ───────────────────────────────
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    text = block.text.strip()
                    ui.agent_text(text, role=self.role_name)
                    session.add_step(StepKind.PLAN, step=step_num, text=text)

            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            # ── no tool calls → the model is done (or stuck) ────────
            if not tool_use_blocks:
                final_text = "\n".join(
                    b.text.strip() for b in response.content
                    if b.type == "text" and b.text.strip()
                )
                # Single-agent run (no role tool subset): a substantive
                # plain-text turn is a legitimate answer to the task, not
                # necessarily a stuck model. Accept it as completion so
                # Q&A-style tasks (e.g. "is this data dirty?") succeed
                # instead of failing with no_tool_use. Role-based runs in
                # the orchestrator keep the strict finish_task contract.
                if final_text and self._tool_names is None:
                    session.summary   = final_text
                    session.outcome   = "complete"
                    session.add_step(StepKind.COMPLETE, summary=session.summary)
                else:
                    session.outcome = "no_tool_use"
                return self._finish(session, messages)

            # ── execute each tool call ───────────────────────────────
            tool_results = []
            finished     = False
            abort        = False

            for block in tool_use_blocks:
                ui.tool_call(block.name, block.input, role=self.role_name)

                # Log call BEFORE running — so we have a record even if
                # execution hangs or the process is killed mid-task
                session.add_step(
                    StepKind.TOOL_CALL,
                    step  = step_num,
                    tool  = block.name,
                    input = block.input,
                )

                # ── approval gate (interactive mode) ────────────────
                # Only side-effecting tools are gated; read-only calls run
                # unprompted, or the REPL would ask about every list_files.
                # A refusal is fed back as a tool result rather than raised:
                # the model should learn the call was declined and pick
                # another route, not have the whole run torn down.
                if self._approve and requires_approval(block.name):
                    if not self._approve(block.name, block.input):
                        refusal = (
                            f"Error: the user declined to approve "
                            f"'{block.name}'. Do not retry it — take a "
                            f"different approach, or ask what to do instead."
                        )
                        session.add_step(
                            StepKind.TOOL_RESULT,
                            step   = step_num,
                            tool   = block.name,
                            result = refusal,
                        )
                        ui.warn("declined", f"{block.name} was not approved")
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     refusal,
                        })
                        continue

                # Execute — wrapped in an observability span if configured
                if self._observe:
                    with self._observe.tool_call_span(block.name, step_num) as span_ctx:
                        raw_result = run_tool(block.name, block.input)
                        if raw_result.startswith("Error"):
                            span_ctx.mark_failed()
                else:
                    raw_result = run_tool(block.name, block.input)
                final_result = raw_result    # may be enriched by Phase 4 / Phase 15 below

                # A presentation hook, never a control-flow one: a caller that
                # raises in here must not kill a run that otherwise succeeded.
                if self._on_tool_result:
                    try:
                        self._on_tool_result(block.name, block.input, raw_result)
                    except Exception as exc:  # noqa: BLE001
                        print(f"{tag}[warn] on_tool_result hook failed: {exc}")

                # ── Phase 4: self-correction policy ─────────────────
                # Runs on raw_result, NOT a guardrail-annotated string —
                # _is_error() relies on result.startswith("Error") checks
                # that would break if a guardrail banner were prepended
                # first. Phase 15's guardrail scan therefore runs AFTER
                # this, layering its banner onto whatever Phase 4 already
                # produced, never the other way around.
                if self._policy:
                    is_error, final_result = self._policy.assess(
                        block.name, raw_result
                    )
                    if is_error:
                        session.corrections += 1
                        session.add_step(
                            StepKind.CORRECTION,
                            tool       = block.name,
                            error_kind = self._policy.last_error_kind(),
                            attempt    = self._policy.consecutive_errors,
                        )
                        ui.warn(
                            "correction",
                            f"error in '{block.name}' — attempt "
                            f"{self._policy.consecutive_errors}/{self._policy.max_consecutive}, "
                            "hint sent back to the model",
                        )

                    if self._policy.should_abort():
                        abort = True

                # ── Phase 15: guardrail scan ─────────────────────────
                if self._guardrails:
                    flagged, final_result = self._guardrails.scan_tool_result(block.name, final_result)
                    if flagged:
                        ui.warn("guardrail", f"possible prompt injection in '{block.name}' result — flagged for the model")

                # ── V3: doom-loop detection ──────────────────────────
                # Signature = (tool, canonicalized args, result hash) —
                # polling with changing results never trips this; true
                # same-call-same-result repetition does. The corrective
                # note rides on the result so the model sees it exactly
                # where the loop lives.
                if doom.record(block.name, block.input, raw_result):
                    final_result = str(final_result) + DOOM_WARNING
                    ui.warn("doom-loop", f"repetition guard triggered on '{block.name}'")

                # ── Phase 5: log the result ──────────────────────────
                # Store the *raw* result (not the enriched one) so the
                # session trace stays clean and factual.
                result_for_log = raw_result[:3000]
                session.add_step(
                    StepKind.TOOL_RESULT,
                    step   = step_num,
                    tool   = block.name,
                    result = result_for_log,
                )

                # Display (truncated for terminal readability)
                ui.tool_result(final_result)

                # Send the enriched result to the model so it sees the hints
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     str(final_result),
                })

                if block.name == "finish_task":
                    summary = block.input.get("summary", "")
                    # Every other guard protects one tool's output; this is the
                    # only check on the CONCLUSION, where true results can be
                    # assembled into a false claim. One chance to revise, then
                    # accept regardless so a stubborn model cannot loop forever.
                    issues = _review_summary(summary) if not summary_revised else []
                    if issues:
                        summary_revised = True
                        note = ("\n\n[REVIEW] Your summary contradicts what was measured. Fix "
                                "these and call finish_task again — this check runs once:\n"
                                + "\n".join(f"  • {i}" for i in issues))
                        tool_results[-1]["content"] = str(tool_results[-1]["content"]) + note
                        for issue in issues:
                            ui.error(f"[review] {issue}")
                        session.add_step(StepKind.ERROR, reason="summary_review_failed")
                    else:
                        finished        = True
                        session.summary = summary
                        session.add_step(StepKind.COMPLETE, summary=summary)

                if abort:
                    break   # don't run remaining tool calls in the batch

            # ── handle exit conditions ───────────────────────────────
            budget = _budget_notice(step_num, self.max_iterations)
            if budget and tool_results and not finished:
                # Without this the loop just stops dead at the cap and the user
                # gets nothing, however much good work was already done.
                tool_results[-1]["content"] = str(tool_results[-1]["content"]) + budget
                ui.info(budget.strip())
            messages.append({"role": "user", "content": tool_results})

            if abort:
                ui.error(
                    f"Stopped — {self._policy.max_consecutive} consecutive "
                    "errors with no successful step between them."
                )
                session.outcome = "max_corrections"
                session.add_step(
                    StepKind.ERROR,
                    reason = "max_consecutive_errors_reached",
                    step   = step_num,
                )
                return self._finish(session, messages)

            if finished:
                session.outcome = "complete"
                return self._finish(session, messages)

        # ── iteration cap ────────────────────────────────────────────────
        ui.error(f"Stopped after {self.max_iterations} iterations without finishing.")
        session.outcome = "max_iterations"
        session.add_step(StepKind.ERROR, reason="max_iterations_reached")
        return self._finish(session, messages)
