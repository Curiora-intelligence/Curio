import os
import subprocess
import sys
from unittest.mock import Mock
from uuid import uuid4

import pytest
from redis.exceptions import ConnectionError
from sqlalchemy import select

from app.persistence.database import Database
from app.persistence.models import Memory
from app.persistence.repositories import MemoryRepository
from app.services.curio import CurioService
from app.runtimes.turn import ModelTurn
from app.services.ephemeral import EphemeralStore
from app.services.memory import extract_explicit


def gateway():
    return Mock(generate_turn=Mock(return_value=ModelTurn(final="Okay")), generate_vision=Mock(return_value="A red square"))


async def test_new_conversation_recalls_memory_and_filters_irrelevant(db):
    model = gateway()
    first = CurioService(db, model)
    _, cid = await first.respond("I prefer spicy chicken biryani and usually stay under ₹300.", user_id="alice")
    await first.respond("I struggle with database indexing.", user_id="alice")
    second = CurioService(db, model)
    _, new_cid = await second.respond("I'm hungry.", user_id="alice")
    assert new_cid != cid
    context = str(model.generate_turn.call_args.kwargs["messages"])
    assert "spicy chicken biryani" in context and '300' in context
    assert "indexing" not in context
    await second.respond("What is 2 + 2?", user_id="alice")
    assert not any(m["content"].startswith("Retrieved context") for m in model.generate_turn.call_args.kwargs["messages"])


async def test_conflicts_supersede_and_user_isolation(db):
    service = CurioService(db, gateway())
    _, cid = await service.respond("I prefer spicy chicken biryani.", user_id="alice")
    await service.respond("I prefer vegetarian biryani.", user_id="alice")
    async with db.sessions() as session:
        rows = list((await session.scalars(select(Memory))).all())
        assert len(rows) == 2 and sum(m.active for m in rows) == 1
        assert [m.value for m in rows if m.active] == ["vegetarian biryani"]
        assert not await MemoryRepository(session).list("bob")
    with pytest.raises(LookupError):
        await service.respond("hello", conversation_id=cid, user_id="bob")


async def test_redis_outage_falls_back_to_database(db):
    class BrokenRedis:
        async def get(self, *args): raise ConnectionError()
        async def set(self, *args, **kwargs): raise ConnectionError()
        async def ping(self): raise ConnectionError()
    cache = EphemeralStore(client=BrokenRedis())
    service = CurioService(db, gateway(), cache)
    await service.respond("I struggle with database indexing.", user_id="alice")
    await service.respond("Mock interview me for a backend internship.", user_id="alice")
    assert "database indexing" in str(service.gateway.generate_turn.call_args.kwargs["messages"])
    assert await cache.health() == "unavailable"


@pytest.mark.parametrize("statement", ["It is raining.", "She prefers chicken biryani.", 'He said "I prefer biryani".', "Can you remember my name?", "I saw a black product today."])
def test_conservative_extraction(statement):
    assert not extract_explicit(statement)


@pytest.mark.postgres
async def test_postgres_across_repository_and_process_instances():
    url = os.getenv("CURIO_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set CURIO_TEST_DATABASE_URL to a migrated disposable PostgreSQL database")
    user_id = "test-" + str(uuid4())
    db = Database(url)
    await CurioService(db, gateway()).respond("I prefer spicy chicken biryani and usually stay under ₹300.", user_id=user_id)
    await db.close()
    code = '''import asyncio,sys
from app.persistence.database import Database
from app.persistence.repositories import MemoryRepository
async def check():
 db=Database(sys.argv[1])
 async with db.sessions() as session:
  rows=await MemoryRepository(session).list(sys.argv[2])
  assert {m.value for m in rows} == {"spicy chicken biryani", "300"}
 await db.close()
asyncio.run(check())'''
    result = subprocess.run([sys.executable, "-c", code, url, user_id], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


async def test_optional_embeddings_failure_keeps_keyword_memory(db):
    from app.services.memory import MemoryService
    class UnavailableEmbeddings:
        async def embed(self, text): raise RuntimeError('model missing')
    service = CurioService(db, gateway())
    service.memory = MemoryService(embeddings=UnavailableEmbeddings())
    await service.respond('I struggle with database indexing.', user_id='alice')
    await service.respond('Mock interview me.', user_id='alice')
    assert 'database indexing' in str(service.gateway.generate_turn.call_args.kwargs['messages'])
