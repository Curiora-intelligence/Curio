"""Model output contract independent of MLX, Torch and orchestration."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: Any
    recipient: str


@dataclass
class ModelTurn:
    final: str | None = None
    tool_calls: list[ToolRequest] = field(default_factory=list)
    # Internal Harmony messages, including analysis, retained only within this turn.
    continuation: list[dict] = field(default_factory=list)
