"""Coverage preserved from deleted legacy tests, adapted to durable async state."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import httpx
import pytest

from main import app
from app.routers.curio import curio_service
from app.runtimes.turn import ModelTurn
from app.services.vision import VisionService


@pytest.fixture
async def client(db, monkeypatch):
    monkeypatch.setattr(curio_service, "database", db)
    monkeypatch.setattr(curio_service.agent, "database", db)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_health_and_openapi(client):
    response = (await client.get('/health')).json()
    assert response['status'] == 'ok' and response['postgres'] == 'ok'
    assert '/curio/analyze' in (await client.get('/openapi.json')).json()['paths']


async def test_text_history(client, monkeypatch):
    model = Mock(return_value=ModelTurn(final="4"))
    monkeypatch.setattr(curio_service.gateway, "generate_turn", model)
    assert (await client.post('/curio/analyze', data={'message': ' '})).status_code == 400
    first = await client.post('/curio/analyze', data={'message': '2 + 2?', 'user_id': 'alice'})
    assert first.status_code == 200
    cid = first.json()['conversation_id']
    second = await client.post('/curio/analyze', data={'message': 'Again?', 'conversation_id': cid, 'user_id': 'alice'})
    assert second.json()['conversation_id'] == cid and second.json()['mode'] == 'text'
    assert [m['role'] for m in model.call_args.kwargs['messages']] == ['system', 'user', 'assistant', 'user']


@pytest.mark.parametrize("failure", [False, True])
async def test_image_cleanup(client, monkeypatch, failure):
    paths = []
    def vision(**kwargs):
        paths.append(Path(kwargs['image_path']))
        assert paths[-1].is_file()
        if failure:
            raise RuntimeError('test failure')
        return 'red'
    monkeypatch.setattr(curio_service.gateway, 'generate_vision', vision)
    result = await client.post('/curio/analyze', files={'image': ('test.png', b'fake-image', 'image/png')})
    assert result.status_code == (500 if failure else 200)
    assert not paths[0].exists()


async def test_image_validation(client, monkeypatch):
    for content, mime, status in [(b'', 'image/png', 400), (b'x', 'text/plain', 415)]:
        assert (await client.post('/curio/analyze', files={'image': ('test', content, mime)})).status_code == status
    monkeypatch.setattr('app.routers.curio.MAX_IMAGE_SIZE', 2)
    assert (await client.post('/curio/analyze', files={'image': ('test.png', b'123', 'image/png')})).status_code == 413


def test_vision_helper_contract():
    gateway = Mock(generate_vision=Mock(return_value='ok'))
    VisionService(gateway).analyze(__file__, 'Describe it')
    assert 'messages' in gateway.generate_vision.call_args.kwargs
    assert 'prompt' not in gateway.generate_vision.call_args.kwargs


def test_lazy_model_imports():
    result = subprocess.run([sys.executable, '-c', 'import sys; import main; assert "mlx.core" not in sys.modules; assert "torch" not in sys.modules'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


async def test_memory_api(client):
    result = await client.post('/memory/remember', json={'user_id': 'alice', 'statement': 'I struggle with database indexing.'})
    assert result.status_code == 200
    memory_id = result.json()['memories'][0]['id']
    assert (await client.get('/memory/bob')).json()['memories'] == []
    assert (await client.post('/memory/forget', json={'user_id': 'bob', 'memory_id': memory_id})).status_code == 404
    assert (await client.post('/memory/forget', json={'user_id': 'alice', 'memory_id': memory_id})).status_code == 200
    assert (await client.get('/memory/alice')).json()['memories'] == []
