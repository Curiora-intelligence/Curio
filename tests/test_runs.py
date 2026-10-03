import asyncio
import json
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from redis.exceptions import ConnectionError
from sqlalchemy import select

from main import app
from app.persistence.models import AgentRun, Message
from app.routers.curio import curio_service
from app.routers.runs import run_manager
from app.runtimes.turn import ModelTurn, ToolRequest
from app.services.ephemeral import EphemeralStore
from app.services.events import RunEvents, STREAM_TTL
from app.services.runs import RunManager
from app.tools.registry import Tool, Permission
from tests.test_agent_tools import Args


class Streams:
    def __init__(self):
        self.rows = {}
        self.ttls = {}

    async def ping(self):
        return True

    async def xadd(self, key, fields, id, maxlen):
        self.rows.setdefault(key, []).append((id, fields))
        self.rows[key] = self.rows[key][-maxlen:]

    async def expire(self, key, ttl):
        self.ttls[key] = ttl

    async def xrange(self, key, min, max, count):
        number = int(min.lstrip('(').split('-')[0])
        return [(id, fields) for id, fields in self.rows.get(key, []) if int(id.split('-')[0]) > number][:count]


def parse_events(text):
    result = []
    for block in text.split('\n\n'):
        fields = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line and not line.startswith(':'))
        if 'event' in fields:
            result.append((fields['id'], fields['event'], json.loads(fields['data'])))
    return result


@pytest.fixture
async def runs_client(db, monkeypatch):
    monkeypatch.setattr(curio_service, 'database', db)
    monkeypatch.setattr(curio_service.agent, 'database', db)
    cache = EphemeralStore()
    # Existing cache operations are intentionally disabled; this fixture tests streams separately.
    cache.client = Streams()
    cache.set = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    monkeypatch.setattr(curio_service, 'cache', cache)
    monkeypatch.setattr(curio_service.gateway, 'generate_turn', Mock(return_value=ModelTurn(final='Hello.')))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        yield client
    await run_manager.close()


async def submit(client, **data):
    response = await client.post('/curio/runs', data={'user_id': 'alice', 'message': 'Hello', **data})
    assert response.status_code == 202
    assert response.json()['status'] == 'accepted'
    return response.json()['request_id']


async def events(client, rid, **kwargs):
    response = await client.get(f'/curio/runs/{rid}/events?user_id=alice', **kwargs)
    assert response.headers['content-type'].startswith('text/event-stream')
    assert response.headers['x-accel-buffering'] == 'no'
    assert response.headers['cache-control'] == 'no-cache'
    assert response.headers['connection'] == 'keep-alive'
    return parse_events(response.text)


async def test_accepts_without_waiting_and_durable_status(runs_client, db, monkeypatch):
    release = threading.Event()
    def generate(**kwargs):
        assert release.wait(5)
        return ModelTurn(final='Safe final')
    monkeypatch.setattr(curio_service.gateway, 'generate_turn', generate)
    try:
        rid = await asyncio.wait_for(submit(runs_client), 1)
        status = await runs_client.get(f'/curio/runs/{rid}?user_id=alice')
        assert status.json()['status'] in {'accepted', 'running'}
        assert (await runs_client.get(f'/curio/runs/{rid}?user_id=bob')).status_code == 404
    finally:
        release.set()
    await run_manager.close()
    status = await RunManager(curio_service).status(rid, 'alice')
    assert status['status'] == 'completed' and status['answer'] == 'Safe final'
    async with db.sessions() as session:
        assert (await session.get(AgentRun, rid)).answer == 'Safe final'
        assert len((await session.scalars(select(Message))).all()) == 2


async def test_stage_order_final_and_reconnect(runs_client):
    rid = await submit(runs_client)
    await run_manager.close()
    rows = await events(runs_client, rid)
    assert [kind for _, kind, _ in rows] == ['run_started', 'stage', 'stage', 'stage', 'stage', 'stage', 'final', 'done']
    assert [data['stage'] for _, kind, data in rows if kind == 'stage'] == ['understanding', 'memory', 'reasoning', 'verification', 'response']
    assert [int(id.split('-')[0]) for id, _, _ in rows] == list(range(1, 9))
    assert rows[-2][2]['answer'] == 'Hello.'
    resumed = await events(runs_client, rid, headers={'Last-Event-ID': rows[3][0]})
    assert resumed == rows[4:]
    assert (await runs_client.get(f'/curio/runs/{rid}/events?user_id=alice', headers={'Last-Event-ID':'bad'})).status_code == 400


@pytest.mark.parametrize('failed', [False, True])
async def test_real_tool_events_and_private_fields(runs_client, monkeypatch, failed):
    handler = AsyncMock(side_effect=RuntimeError('SECRET') if failed else None, return_value={'private': 'PRIVATE_RESULT'})
    tool = Tool('test.search', 'Search', Args, Permission.READ_ONLY, handler)
    monkeypatch.setitem(curio_service.tools.tools, tool.name, tool)
    monkeypatch.setattr(curio_service.gateway, 'generate_turn', Mock(side_effect=[
        ModelTurn(tool_calls=[ToolRequest('test.search', {'query':'PRIVATE_ARGUMENT'}, 'functions.test_search')],
                  continuation=[{'role':'assistant', 'content':'PRIVATE_HARMONY_ANALYSIS'}]), ModelTurn(final='Finished')]))
    rid = await submit(runs_client, message='PRIVATE_USER_TEXT')
    await run_manager.close()
    rows = await events(runs_client, rid)
    kinds = [kind for _, kind, _ in rows]
    assert 'tool_started' in kinds and ('tool_failed' if failed else 'tool_finished') in kinds
    for _, kind, data in rows:
        if kind.startswith('tool_'):
            assert set(data) == {'tool', 'status'} and data['tool'] == 'test.search'
    assert all(secret not in str(rows) for secret in ['SECRET', 'PRIVATE_', '<|channel|>', 'system', 'arguments'])
    handler.assert_awaited_once()


async def test_invalid_tool_does_not_invent_execution(runs_client, monkeypatch):
    monkeypatch.setattr(curio_service.gateway, 'generate_turn', Mock(side_effect=[
        ModelTurn(tool_calls=[ToolRequest('memory.search', {}, 'functions.memory_search'),
                              ToolRequest('secret.unknown', {}, 'functions.unknown')]), ModelTurn(final='Done')]))
    rid = await submit(runs_client)
    await run_manager.close()
    assert not any(kind.startswith('tool_') for _, kind, _ in await events(runs_client, rid))


async def test_failed_run_is_durable_and_safe(runs_client, monkeypatch):
    monkeypatch.setattr(curio_service.gateway, 'generate_turn', Mock(side_effect=RuntimeError('PRIVATE_SYSTEM_PROMPT')))
    rid = await submit(runs_client)
    await run_manager.close()
    assert (await runs_client.get(f'/curio/runs/{rid}?user_id=alice')).json()['status'] == 'failed'
    rows = await events(runs_client, rid)
    assert [r[1] for r in rows[-2:]] == ['error', 'done']
    assert 'PRIVATE' not in str(rows)


@pytest.mark.parametrize('outage', [False, True])
async def test_no_redis_and_expired_stream_recover_from_database(runs_client, monkeypatch, outage):
    if outage:
        broken = Mock(xadd=AsyncMock(side_effect=ConnectionError()), xrange=AsyncMock(side_effect=ConnectionError()))
        monkeypatch.setattr(curio_service.cache, 'client', broken)
    rid = await submit(runs_client)
    await run_manager.close()
    # Simulate a different worker with no transient stream left.
    monkeypatch.setattr(curio_service.cache, 'client', None)
    rows = await events(runs_client, rid)
    assert [r[1] for r in rows] == ['final', 'done']
    assert rows[0][2]['answer'] == 'Hello.'
    assert await events(runs_client, rid, headers={'Last-Event-ID': rows[-1][0]}) == []


async def test_idempotent_acceptance_and_iteration_limit(runs_client, monkeypatch, db):
    monkeypatch.setattr(curio_service.gateway, 'generate_turn', Mock(return_value=ModelTurn(tool_calls=[
        ToolRequest('unknown', {}, 'functions.unknown')])))
    rid = await submit(runs_client, request_id='once')
    assert await submit(runs_client, request_id='once') == rid
    await run_manager.close()
    async with db.sessions() as session:
        run = await session.get(AgentRun, rid)
        assert run.iterations == 6 and run.status == 'iteration_limit'
    assert (await runs_client.get(f'/curio/runs/{rid}?user_id=alice')).json()['status'] == 'completed'
    assert curio_service.gateway.generate_turn.call_count == 6


@pytest.mark.parametrize('failed', [False, True])
async def test_async_image_lifetime(runs_client, monkeypatch, failed):
    paths = []
    def generate(**kwargs):
        path = Path(kwargs['image_path']); paths.append(path)
        assert path.read_bytes() == b'image'
        if failed:
            raise RuntimeError('vision failed')
        return 'A red square'
    monkeypatch.setattr(curio_service.gateway, 'generate_vision', generate)
    response = await runs_client.post('/curio/runs', data={'user_id':'alice'}, files={'image':('a.png', b'image', 'image/png')})
    assert response.status_code == 202
    await run_manager.close()
    assert paths and not paths[0].exists()
    state = (await runs_client.get('/curio/runs/' + response.json()['request_id'] + '?user_id=alice')).json()
    assert state['status'] == ('failed' if failed else 'completed')


async def test_heartbeat_and_cross_origin(runs_client, monkeypatch):
    import app.routers.runs as router
    monkeypatch.setattr(router, 'POLL_SECONDS', .01)
    monkeypatch.setattr(router, 'HEARTBEAT_SECONDS', .01)
    def slow(**kwargs):
        import time
        time.sleep(.12)
        return ModelTurn(final='Done')
    monkeypatch.setattr(curio_service.gateway, 'generate_turn', slow)
    rid = await submit(runs_client)
    result = await runs_client.get(f'/curio/runs/{rid}/events?user_id=alice', headers={'Origin':'http://127.0.0.1:8000'})
    assert ': keep-alive' in result.text
    assert result.headers['access-control-allow-origin'] == 'http://127.0.0.1:8000'
    result = await runs_client.get('/health', headers={'Origin':'https://untrusted.example'})
    assert 'access-control-allow-origin' not in result.headers


async def test_real_redis_stream_replay_and_ttl():
    url = os.getenv('CURIO_TEST_REDIS_URL')
    if not url:
        pytest.skip('Set CURIO_TEST_REDIS_URL for real Redis integration')
    from redis.asyncio import Redis
    client = Redis.from_url(url, decode_responses=True)
    publisher = RunEvents(client, str(uuid4()))
    try:
        await publisher.started(); await publisher.stage('memory'); await publisher.done()
        rows = await client.xrange(publisher.key, min='(1-0')
        assert [r[0] for r in rows] == ['2-0', '3-0']
        assert 0 < await client.ttl(publisher.key) <= STREAM_TTL
    finally:
        await client.delete(publisher.key)
        await client.aclose()


@pytest.mark.postgres
async def test_postgres_run_survives_new_manager_and_new_conversation():
    url = os.getenv('CURIO_TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set CURIO_TEST_DATABASE_URL for durable run integration')
    from app.persistence.database import Database
    from app.services.curio import CurioService
    from app.persistence.repositories import MemoryRepository
    db = Database(url)
    model = Mock(generate_turn=Mock(return_value=ModelTurn(final='Remembered.')))
    service = CurioService(db, model, EphemeralStore())
    manager = RunManager(service)
    user_id = 'sse-' + str(uuid4())
    try:
        rid = await manager.start(user_id=user_id, message='I prefer spicy chicken biryani and usually stay under ₹300.')
        await manager.close()
        first = await RunManager(service).status(rid, user_id)
        assert first['status'] == 'completed' and first['answer'] == 'Remembered.'
        second = await manager.start(user_id=user_id, message="I'm hungry.")
        await manager.close()
        assert (await manager.status(second, user_id))['conversation_id'] != first['conversation_id']
        assert 'spicy chicken biryani' in str(model.generate_turn.call_args.kwargs['messages'])
        async with db.sessions() as session:
            assert len(await MemoryRepository(session).list(user_id)) >= 2
    finally:
        await db.close()


async def test_shopping_verification_does_not_treat_budget_as_product_price(runs_client, monkeypatch):
    monkeypatch.setattr(curio_service.gateway, 'generate_turn', Mock(return_value=ModelTurn(
        final="It matches your style and is within your ₹2500 budget. I don't have the price.")))
    rid = await submit(runs_client, mode='shopping', message='Would I like this?')
    await run_manager.close()
    status = (await runs_client.get(f'/curio/runs/{rid}?user_id=alice')).json()
    assert 'is within' not in status['answer'] and "don't have the price" in status['answer']
    from app.services.coaching import shopping_price_evidence, safe_shopping_text
    assert not shopping_price_evidence('{"memories":[{"value":"2500"}]}', [])
    assert shopping_price_evidence('{"visual_observation":{"visible_text":["₹2200"]}}', [])
    assert shopping_price_evidence('', [{'recommendations':[{'price':2200}]}])
    assert safe_shopping_text('It fits your budget.', True) == 'It fits your budget.'
    assert safe_shopping_text("I cannot confirm whether it fits your budget.", False) == "I cannot confirm whether it fits your budget."
