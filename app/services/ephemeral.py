"""Redis is disposable; every operation tolerates absence or outage."""
import json
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError


class EphemeralStore:
    def __init__(self, url: str | None = None, client=None):
        self.client = client or (Redis.from_url(url, decode_responses=True, socket_connect_timeout=0.3,
                                              socket_timeout=0.3) if url else None)

    async def get(self, key: str):
        if not self.client:
            return None
        try:
            value = await self.client.get("curio:" + key)
            return json.loads(value) if value else None
        except (RedisError, OSError, ValueError):
            return None

    async def set(self, key: str, value, ttl: int = 60) -> None:
        if self.client:
            try:
                await self.client.set("curio:" + key, json.dumps(value), ex=ttl)
            except (RedisError, OSError):
                pass

    async def acquire(self, key: str, ttl: int = 600) -> str | None:
        token = str(uuid4())
        if not self.client:
            return token  # PostgreSQL provides the authoritative turn lock.
        try:
            return token if await self.client.set("curio:lock:" + key, token, nx=True, ex=ttl) else None
        except (RedisError, OSError):
            return token

    async def release(self, key: str, token: str) -> None:
        if self.client:
            try:
                await self.client.eval("if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) end",
                                       1, "curio:lock:" + key, token)
            except (RedisError, OSError):
                pass

    async def health(self) -> str:
        if not self.client:
            return "disabled"
        try:
            return "ok" if await self.client.ping() else "unavailable"
        except (RedisError, OSError):
            return "unavailable"

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()
