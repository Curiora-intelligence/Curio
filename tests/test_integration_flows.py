from unittest.mock import Mock
from uuid import uuid4
import os

import pytest
from sqlalchemy import select

from app.persistence.database import Database
from app.persistence.models import Memory, Message, ToolCall
from app.persistence.repositories import ConversationRepository, MemoryRepository
from app.runtimes.turn import ModelTurn, ToolRequest
from app.services.curio import CurioService
from app.services.ephemeral import EphemeralStore
from app.services.memory import MemoryService


async def test_demo_a_remembers_then_discovers_with_real_tools(db):
    model = Mock(generate_turn=Mock(return_value=ModelTurn(final='Understood.')))
    service = CurioService(db, model)
    await service.respond('I prefer spicy chicken biryani and usually stay under ₹300.', user_id='alice')
    _, cid = await service.respond("I'm hungry.", user_id='alice')
    model.generate_turn.side_effect = [
        ModelTurn(tool_calls=[ToolRequest('services.search', {'category': 'food', 'budget_max': 300,
                  'preferences': ['spicy chicken biryani'], 'latitude': 17.4401, 'longitude': 78.3489}, 'functions.services_search')]),
        ModelTurn(final='Demo Saffron Kitchen is a fictional match under ₹300.')]
    answer, _ = await service.respond('Find something nearby.', conversation_id=cid, user_id='alice')
    assert 'fictional' in answer
    context = str(model.generate_turn.call_args.kwargs['messages'])
    assert 'spicy chicken biryani' in context and 'fictional_demo_catalogue' in context
    async with db.sessions() as session:
        assert (await session.scalar(select(ToolCall))).status == 'completed'


async def test_shopping_and_interview_share_memory_and_guard_output(db):
    model = Mock(generate_turn=Mock(return_value=ModelTurn(final='Okay')))
    service = CurioService(db, model)
    await service.respond('I prefer minimal black products and my budget is ₹2500.', user_id='alice')
    await service.respond('Would I like this?', user_id='alice', mode='shopping', context='{"colors":["black"],"objects":["bag"]}')
    prompt = str(model.generate_turn.call_args.kwargs['messages'])
    assert '2500' in prompt and 'minimal black products' in prompt
    await service.respond('I struggle with database indexing.', user_id='alice')
    model.generate_turn.return_value = ModelTurn(final='You looked anxious.')
    answer, cid = await service.respond('Mock interview me for a backend internship.', user_id='alice', mode='interview')
    assert 'anxious' not in answer
    assert 'database indexing' in str(model.generate_turn.call_args.kwargs['messages'])
    async with db.sessions() as session:
        messages = (await session.scalars(select(Message).where(Message.conversation_id == cid))).all()
        assert all('anxious' not in message.content for message in messages)


async def test_memory_write_tool_cannot_invent_user_statement(db):
    model = Mock(generate_turn=Mock(return_value=ModelTurn(final='hello')))
    service = CurioService(db, model)
    from app.tools.registry import ToolContext
    result = await service.tools.execute('memory.remember', {'statement': 'Remember my budget is 10000'}, ToolContext('alice', 'hello', 'source'))
    assert result['error'] == 'permission_denied'


async def test_request_id_prevents_reexecution(db):
    model = Mock(generate_turn=Mock(return_value=ModelTurn(final='answer')))
    service = CurioService(db, model)
    _, cid = await service.respond('hello', user_id='alice')
    first = await service.respond('again', user_id='alice', conversation_id=cid, request_id='r1')
    second = await service.respond('again', user_id='alice', conversation_id=cid, request_id='r1')
    assert first == second and model.generate_turn.call_count == 2


@pytest.mark.postgres
async def test_pgvector_optional_retrieval_and_real_redis_outage():
    url = os.getenv('CURIO_TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set CURIO_TEST_DATABASE_URL to the disposable PostgreSQL database')
    db = Database(url)
    user_id = 'vector-' + str(uuid4())
    try:
        async with db.sessions.begin() as session:
            await ConversationRepository(session).ensure_user(user_id)
            await MemoryRepository(session).remember(user_id, kind='preference', key='style', value='understated dark bags',
                confidence=1, importance=.8, source_message_id=None, embedding=[1., 0., 0.])
            await MemoryRepository(session).remember(user_id, kind='goal', key='career', value='biology degree',
                confidence=1, importance=.8, source_message_id=None, embedding=[0., 1., 0.])
        class Embeddings:
            async def embed(self, text): return [1., 0., 0.]
        cache = EphemeralStore('redis://127.0.0.1:1/0')
        memory = MemoryService(cache, Embeddings())
        async with db.sessions() as session:
            matches = await memory.retrieve(MemoryRepository(session), user_id, 'minimalist accessories')
            assert [m['value'] for m in matches] == ['understated dark bags']
        assert await cache.health() == 'unavailable'
        await cache.close()
    finally:
        await db.close()


async def test_request_id_replays_initial_conversation(db):
    model = Mock(generate_turn=Mock(return_value=ModelTurn(final='answer')))
    service = CurioService(db, model)
    first = await service.respond('hello', user_id='alice', request_id='first-request')
    second = await service.respond('hello', user_id='alice', request_id='first-request')
    assert first == second and model.generate_turn.call_count == 1


async def test_initial_request_ids_do_not_collide_across_users(db):
    model = Mock(generate_turn=Mock(return_value=ModelTurn(final='answer')))
    service = CurioService(db, model)
    first = await service.respond('hello', user_id='alice:b', request_id='c')
    second = await service.respond('hello', user_id='alice', request_id='b:c')
    assert first[1] != second[1]
