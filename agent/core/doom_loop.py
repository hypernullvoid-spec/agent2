"""
Doom-loop detection — rescue the agent from repeating itself.

The most common failure mode of long autonomous runs is not a crash, it's
a loop: the agent calls the same tool with the same arguments, gets the
same result, and calls it again. Ported from huggingface/ml-intern's
detector, including its key subtlety: the signature hashes the tool name,
the *canonicalized* arguments, AND the result — so legitimate polling
(same call, changing results) is never flagged, while true no-progress
repetition (same call, same result) is.

Detects two shapes:
  • 3+ identical consecutive signatures      (A, A, A)
  • a repeating pair pattern                 (A, B, A, B)

On detection, a corrective system note is appended to the tool result so
the model sees it exactly where the loop lives.
"""

from __future__ import annotations

import hashlib
import json

WINDOW = 30          # signatures remembered
REPEAT_THRESHOLD = 3  # identical consecutive calls that count as a loop

WARNING = (
    "\n\n[SYSTEM: REPETITION GUARD] You have made the same tool call with the "
    "same arguments and received the same result multiple times in a row. "
    "Repeating it again will not produce a different outcome. Stop, state "
    "what you learned, and either (a) try a materially different approach, "
    "(b) use a different tool, or (c) finish the task with finish_task, "
    "reporting honestly what worked and what did not."
)


def _canon(obj) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


class DoomLoopDetector:
    def __init__(self, window: int = WINDOW, threshold: int = REPEAT_THRESHOLD):
        self.window = window
        self.threshold = threshold
        self._sigs: list[str] = []

    def record(self, tool_name: str, tool_input: dict, result: str) -> bool:
        """Record one tool call; returns True if the agent is looping."""
        sig = hashlib.md5(
            f"{tool_name}|{_canon(tool_input)}|{result[:2000]}".encode()
        ).hexdigest()
        self._sigs.append(sig)
        self._sigs = self._sigs[-self.window:]
        return self._identical_run() or self._pair_cycle()

    def _identical_run(self) -> bool:
        if len(self._sigs) < self.threshold:
            return False
        tail = self._sigs[-self.threshold:]
        return len(set(tail)) == 1

    def _pair_cycle(self) -> bool:
        """A,B,A,B over the last 4 calls (with A != B)."""
        if len(self._sigs) < 4:
            return False
        a, b, c, d = self._sigs[-4:]
        return a == c and b == d and a != b

    def reset(self) -> None:
        self._sigs.clear()
