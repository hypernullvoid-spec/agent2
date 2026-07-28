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

from agent            import ui
from agent.llm            import DEFAULT_MODEL  # deployed model (see agent/llm/router.py)
from agent.llm_client     import LLMClient
from agent.tools          import get_tool_definitions, run_tool
from agent.prompts        import SYSTEM_PROMPT
from agent.memory         import get_session_store, Session, StepKind
from agent.self_correction import SelfCorrectionPolicy
from agent.doom_loop       import DoomLoopDetector, WARNING as DOOM_WARNING

from typing import Optional

MAX_ITERATIONS = int(os.environ.get("SWARN_MAX_ITERATIONS", "30"))

# Context compaction (V3): long runs accumulate huge tool results in the
# message history. When the total conversation size crosses this budget,
# old tool results are truncated head+tail — deterministic, no extra LLM
# call, and the last few turns are always kept verbatim.
CONTEXT_CHAR_BUDGET = int(os.environ.get("SWARN_CONTEXT_CHAR_BUDGET", "400000"))
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
        guardrail_policy:    Optional["object"]              = None,   # agent.observability.GuardrailPolicy
        observability_hooks: Optional["object"]              = None,   # agent.observability.ObservabilityHooks
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

    # ─────────────────────────────────────────────────── public entry point

    def run(self, task: str) -> dict:
        """
        Run the ReAct loop to completion (or until aborted/iteration-capped).

        Returns a small dict describing what happened:
          {"outcome": str, "summary": str | None, "session_id": str}
        so a caller — main.py for single-agent use, or Phase 11's
        Orchestrator for multi-role use — can branch on the result
        without reaching into Session internals or parsing stdout.
        """
        session  = self._store.new_session(task=task, model=self.model)
        messages = [{"role": "user", "content": task}]
        doom     = DoomLoopDetector()

        ui.session_header(session.id, task, role=self.role_name)

        for step_num in range(1, MAX_ITERATIONS + 1):

            # ── V3: context compaction ──────────────────────────────
            n_compacted = compact_messages(messages)
            if n_compacted:
                ui.info(f"context: compacted {n_compacted} old tool result(s)")

            # ── LLM call ────────────────────────────────────────────
            current_tools = get_tool_definitions(self._tool_names)
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
                session.outcome = "no_tool_use"
                self._store.close_session(session)
                return {"outcome": session.outcome, "summary": session.summary, "session_id": session.id}

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

                # Execute — wrapped in an observability span if configured
                if self._observe:
                    with self._observe.tool_call_span(block.name, step_num) as span_ctx:
                        raw_result = run_tool(block.name, block.input)
                        if raw_result.startswith("Error"):
                            span_ctx.mark_failed()
                else:
                    raw_result = run_tool(block.name, block.input)
                final_result = raw_result    # may be enriched by Phase 4 / Phase 15 below

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
                    finished          = True
                    session.summary   = block.input.get("summary", "")
                    session.add_step(StepKind.COMPLETE, summary=session.summary)

                if abort:
                    break   # don't run remaining tool calls in the batch

            # ── handle exit conditions ───────────────────────────────
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
                self._store.close_session(session)
                return {"outcome": session.outcome, "summary": session.summary, "session_id": session.id}

            if finished:
                session.outcome = "complete"
                self._store.close_session(session)
                return {"outcome": session.outcome, "summary": session.summary, "session_id": session.id}

        # ── iteration cap ────────────────────────────────────────────────
        ui.error(f"Stopped after {MAX_ITERATIONS} iterations without finishing.")
        session.outcome = "max_iterations"
        session.add_step(StepKind.ERROR, reason="max_iterations_reached")
        self._store.close_session(session)
        return {"outcome": session.outcome, "summary": session.summary, "session_id": session.id}
