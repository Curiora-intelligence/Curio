from fastapi.concurrency import run_in_threadpool
import json

from app.persistence.repositories import AgentRepository
from app.tools.registry import ToolContext, ToolRegistry


class AgentRuntime:
    maximum_iterations = 6

    def __init__(self, gateway, registry: ToolRegistry, database, cache):
        self.gateway, self.registry, self.database, self.cache = gateway, registry, database, cache

    async def run(self, messages: list[dict], context: ToolContext, run_id: str,
                  max_tokens: int = 1024, temperature: float = .2) -> tuple[str, int, str]:
        messages = list(messages)
        for iteration in range(1, self.maximum_iterations + 1):
            await self.cache.set("run:" + run_id, {"iteration": iteration, "status": "reasoning"}, 600)
            turn = await run_in_threadpool(self.gateway.generate_turn, messages=messages,
                                          tools=self.registry.definitions(), max_tokens=max_tokens, temperature=temperature)
            if turn.tool_calls:
                messages.extend(turn.continuation)
                # A malformed model response cannot multiply the bounded loop into unbounded work.
                for call in turn.tool_calls[:8]:
                    arguments = call.arguments if isinstance(call.arguments, dict) else {"raw": str(call.arguments)[:8000]}
                    async with self.database.sessions.begin() as session:
                        call_id = await AgentRepository(session).begin_tool(run_id, call.name, arguments)
                    result = await self.registry.execute(call.name, call.arguments, context)
                    async with self.database.sessions.begin() as session:
                        await AgentRepository(session).finish_tool(call_id, result)
                    messages.append({"role": "tool", "name": call.recipient,
                                     "content": json.dumps(result, ensure_ascii=False)[:24000]})
            elif turn.final:
                return turn.final, iteration, "completed"
            else:
                raise RuntimeError("Model returned neither a tool call nor a final response.")
        return "I reached the tool-step limit. Please narrow the request so I can continue.", self.maximum_iterations, "iteration_limit"
