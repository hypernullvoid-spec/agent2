"""
Submission Loop - Core agent execution loop with event-based architecture.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from agent.core.session import OpType, Operation, Submission
from agent.core.tools import ToolRouter
from agent.core.agent_loop import AgentLoop
from agent.utils.terminal_display import get_console


@dataclass
class Event:
    event_type: str
    data: dict[str, Any] | None = None


class SubmissionLoop:
    """Main agent loop that processes submissions and emits events."""

    def __init__(
        self,
        submission_queue: asyncio.Queue,
        event_queue: asyncio.Queue,
        config: Any,
        tool_router: ToolRouter,
        session_holder: list,
        hf_token: str | None,
        user_id: str | None,
        hf_username: str | None,
        user_plan: str,
        local_mode: bool,
        autonomous_mode: bool,
        stream: bool,
        notification_gateway: Any,
        notification_destinations: list[str],
        defer_turn_complete_notification: bool,
    ):
        self.submission_queue = submission_queue
        self.event_queue = event_queue
        self.config = config
        self.tool_router = tool_router
        self.session_holder = session_holder
        self.hf_token = hf_token
        self.user_id = user_id
        self.hf_username = hf_username
        self.user_plan = user_plan
        self.local_mode = local_mode
        self.autonomous_mode = autonomous_mode
        self.stream = stream
        self.notification_gateway = notification_gateway
        self.notification_destinations = notification_destinations
        self.defer_turn_complete_notification = defer_turn_complete_notification

        self._cancelled = asyncio.Event()
        self._agent: AgentLoop | None = None
        self._current_submission_id: str | None = None
        self.turn_count = 0
        self.context_manager = None  # Will be initialized with agent

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self):
        self._cancelled.set()

    async def send_deferred_turn_complete_notification(self, event: Event):
        """Send deferred notification for turn completion."""
        pass

    async def run(self):
        """Main loop processing submissions."""
        # Initialize agent
        self._agent = AgentLoop(
            model=self.config.model_name,
            correction_policy=None,  # Could add from config
            guardrail_policy=None,   # Could add from config
        )
        self.session_holder[0] = self

        # Send ready event
        await self.event_queue.put(Event(
            event_type="ready",
            data={"tool_count": len(self.tool_router.get_tool_definitions())}
        ))

        while True:
            try:
                submission = await self.submission_queue.get()
                self._current_submission_id = submission.id

                if submission.operation.op_type == OpType.SHUTDOWN:
                    await self.event_queue.put(Event(event_type="shutdown"))
                    break

                if submission.operation.op_type == OpType.USER_INPUT:
                    await self._process_user_input(submission)

                elif submission.operation.op_type == OpType.EXEC_APPROVAL:
                    await self._process_approval(submission)

                elif submission.operation.op_type == OpType.UNDO:
                    await self.event_queue.put(Event(event_type="undo_complete"))

                elif submission.operation.op_type == OpType.COMPACT:
                    await self.event_queue.put(Event(event_type="compacted", data={"old_tokens": 0, "new_tokens": 0}))

                elif submission.operation.op_type == OpType.NEW:
                    await self.event_queue.put(Event(event_type="new_complete", data={"clear_screen": True}))

                elif submission.operation.op_type == OpType.RESUME:
                    await self.event_queue.put(Event(event_type="resume_complete", data={"path": submission.operation.data.get("path", "")}))

            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.event_queue.put(Event(event_type="error", data={"error": str(e)}))

    async def _process_user_input(self, submission: Submission):
        """Process user input through the agent."""
        text = submission.operation.data.get("text", "") if submission.operation.data else ""

        # Emit processing event
        await self.event_queue.put(Event(event_type="processing"))

        try:
            # Run the agent
            result = self._agent.run(text)

            # Emit assistant message
            if result.get("summary"):
                await self.event_queue.put(Event(
                    event_type="assistant_message",
                    data={"content": result["summary"]}
                ))

            # Emit turn complete
            self.turn_count += 1
            await self.event_queue.put(Event(
                event_type="turn_complete",
                data={"history_size": self.turn_count}
            ))

        except Exception as e:
            await self.event_queue.put(Event(event_type="error", data={"error": str(e)}))

    async def _process_approval(self, submission: Submission):
        """Process approval response."""
        # For now, just continue - the agent would need to be paused/resumed
        await self.event_queue.put(Event(event_type="turn_complete"))


async def submission_loop(
    submission_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    config: Any,
    tool_router: ToolRouter,
    session_holder: list,
    hf_token: str | None,
    user_id: str | None,
    hf_username: str | None,
    user_plan: str,
    local_mode: bool,
    autonomous_mode: bool,
    stream: bool,
    notification_gateway: Any,
    notification_destinations: list[str],
    defer_turn_complete_notification: bool,
):
    """Entry point for the submission loop."""
    loop = SubmissionLoop(
        submission_queue=submission_queue,
        event_queue=event_queue,
        config=config,
        tool_router=tool_router,
        session_holder=session_holder,
        hf_token=hf_token,
        user_id=user_id,
        hf_username=hf_username,
        user_plan=user_plan,
        local_mode=local_mode,
        autonomous_mode=autonomous_mode,
        stream=stream,
        notification_gateway=notification_gateway,
        notification_destinations=notification_destinations,
        defer_turn_complete_notification=defer_turn_complete_notification,
    )
    await loop.run()