"""
Session types and operations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class OpType(Enum):
    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    UNDO = "undo"
    COMPACT = "compact"
    NEW = "new"
    RESUME = "resume"
    SHUTDOWN = "shutdown"


@dataclass
class Operation:
    op_type: OpType
    data: Optional[dict[str, Any]] = None


@dataclass
class Submission:
    id: str
    operation: Operation