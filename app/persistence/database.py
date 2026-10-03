from contextlib import asynccontextmanager
import hashlib
import asyncio

from sqlalchemy import text
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.settings import Settings


class Database:
    def __init__(self, url: str | None = None) -> None:
        url = url or Settings().database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        options = {"connect_args": {"timeout": 3}} if url.startswith("postgresql+") else {}
        self.engine = create_async_engine(url, pool_pre_ping=True, **options)
        self.lock_engine = create_async_engine(url, poolclass=NullPool, **options) if url.startswith("postgresql+") else self.engine
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def health(self) -> str:
        try:
            async with asyncio.timeout(2):
                async with self.engine.connect() as conn:
                    await conn.execute(text("SELECT 1 FROM users LIMIT 1"))
            return "ok"
        except Exception:
            return "unavailable"

    @asynccontextmanager
    async def user_lock(self, user_id: str):
        """Cross-worker turn ordering, including when Redis is down."""
        if self.engine.dialect.name != "postgresql":
            yield  # SQLite is only used in isolated unit tests.
            return
        key = int.from_bytes(hashlib.sha256(user_id.encode()).digest()[:8], "big", signed=True)
        async with self.lock_engine.connect() as conn:
            await conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
            try:
                yield
            finally:
                await conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})

    async def close(self) -> None:
        await self.engine.dispose()
        if self.lock_engine is not self.engine:
            await self.lock_engine.dispose()
