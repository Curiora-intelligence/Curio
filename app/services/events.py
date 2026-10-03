"""Safe transient progress. Redis Streams are optional; PostgreSQL owns results."""
import json
from redis.exceptions import RedisError

STAGES = {
    "understanding": "Understanding your request…",
    "memory": "Checking relevant context…",
    "reasoning": "Reasoning…",
    "tool_execution": "Checking available options…",
    "verification": "Verifying the result…",
    "response": "Preparing response…",
}
STREAM_TTL = 3600


def event_number(value: str) -> int:
    # IDs are explicit integer Redis Stream IDs, local to one execution.
    first, separator, last = value.partition("-")
    if not first.isdigit() or (separator and last != "0") or len(first) > 16:
        raise ValueError("Invalid event ID")
    return int(first)


def encode_event(number, kind, data):
    return f"id: {number}-0\nevent: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class RunEvents:
    def __init__(self, client, request_id):
        self.client, self.request_id = client, request_id
        self.sequence = 0
        self.key = "curio:events:" + request_id

    async def _send(self, kind, data):
        self.sequence += 1
        if self.client:
            try:
                await self.client.xadd(self.key, {"event": kind, "data": json.dumps(data)},
                                       id=f"{self.sequence}-0", maxlen=256)
                await self.client.expire(self.key, STREAM_TTL)
            except (RedisError, OSError):
                pass

    async def started(self):
        await self._send("run_started", {"request_id": self.request_id})

    async def stage(self, name):
        await self._send("stage", {"stage": name, "message": STAGES[name]})

    async def memory(self, memories):
        labels = []
        titles = {"food": "Food preference", "style": "Style preference", "career": "Career goal", "interview": "Interview focus"}
        for memory in memories:
            key = memory.get("key", "")
            if key in titles:
                labels.append(titles[key])
            elif "budget" in key:
                value = str(memory.get("value", ""))
                labels.append("₹" + value + " budget" if value.replace(".", "", 1).isdigit() and len(value) <= 12 else "Budget preference")
        await self._send("stage", {"stage": "memory", "message": STAGES["memory"], "remembered": list(dict.fromkeys(labels))[:6]})

    async def tool(self, name, status):
        # Called only by the registry, after validation/permission and before execution.
        await self._send("tool_" + status, {"tool": name, "status": status})

    async def final(self, answer, conversation_id, user_id):
        await self._send("final", {"answer": answer, "conversation_id": conversation_id, "user_id": user_id})

    async def error(self):
        await self._send("error", {"message": "Curio could not finish this request. Please try again."})

    async def done(self):
        await self._send("done", {})


async def read_events(client, request_id, after):
    if client:
        try:
            return await client.xrange("curio:events:" + request_id, min=f"({after}-0", max="+", count=256)
        except (RedisError, OSError):
            pass
    return []
