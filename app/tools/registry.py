from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Awaitable, Callable

from pydantic import BaseModel, ValidationError

from app.services.ephemeral import EphemeralStore


class Permission(str, Enum):
    READ_ONLY = "READ_ONLY"
    USER_DATA_WRITE = "USER_DATA_WRITE"
    EXTERNAL_ACTION = "EXTERNAL_ACTION"


@dataclass
class ToolContext:
    user_id: str
    statement: str
    source_message_id: str
    # Server-owned confirmation fingerprints bind permission to exact tool + arguments.
    confirmed_actions: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: type[BaseModel]
    permission: Permission
    handler: Callable[[BaseModel, ToolContext], Awaitable[dict]]
    cache_ttl: int = 0

    @property
    def alias(self) -> str:
        return self.name.replace(".", "_")


def action_fingerprint(name: str, arguments: dict) -> str:
    return hashlib.sha256(json.dumps([name, arguments], sort_keys=True).encode()).hexdigest()


class ToolRegistry:
    def __init__(self, cache: EphemeralStore | None = None):
        self.tools: dict[str, Tool] = {}
        self.cache = cache or EphemeralStore()

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools or any(t.alias == tool.alias for t in self.tools.values()):
            raise ValueError("Duplicate tool name or Harmony alias")
        self.tools[tool.name] = tool

    def definitions(self) -> list[dict]:
        return [{"name": t.name, "alias": t.alias, "description": t.description,
                 "parameters": t.schema.model_json_schema(), "permission": t.permission.value} for t in self.tools.values()]

    async def execute(self, name: str, arguments, context: ToolContext) -> dict:
        tool = self.tools.get(name)
        if not tool:
            return {"error": "unknown_tool", "message": "Choose an available tool."}
        try:
            args = tool.schema.model_validate(arguments)
        except ValidationError as exc:
            return {"error": "invalid_arguments", "message": str(exc)[:2000]}
        values = args.model_dump(mode="json")
        if tool.permission == Permission.EXTERNAL_ACTION:
            if action_fingerprint(name, values) not in context.confirmed_actions:
                return {"error": "confirmation_required", "tool": name, "arguments": values}
        if tool.permission == Permission.USER_DATA_WRITE:
            # The model cannot invent a user assertion. Handler applies the memory grammar too.
            if values.get("statement", "").strip() != context.statement.strip():
                return {"error": "permission_denied", "message": "Memory writes must use the exact current user statement."}
        key = "tool:" + context.user_id + ":" + action_fingerprint(name, values)
        if tool.cache_ttl and tool.permission == Permission.READ_ONLY:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached
        try:
            result = await asyncio.wait_for(tool.handler(args, context), timeout=30)
            if tool.cache_ttl and tool.permission == Permission.READ_ONLY and not result.get("error"):
                await self.cache.set(key, result, tool.cache_ttl)
            return result
        except Exception:
            # Keep secrets/implementation exceptions out of model-visible errors.
            return {"error": "tool_failed", "message": f"{name} failed. Try another approach or explain the limitation."}
