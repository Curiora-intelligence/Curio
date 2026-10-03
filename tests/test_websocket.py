from unittest.mock import AsyncMock, Mock
from fastapi.testclient import TestClient

from main import app
from app.services.ephemeral import EphemeralStore


def test_websocket_protocol_and_shared_curio(monkeypatch):
    service = Mock(cache=EphemeralStore(), respond=AsyncMock(return_value=('Try an indexing question.', 'cid')))
    monkeypatch.setattr('app.routers.live.curio_service', service)
    with TestClient(app) as client:
        assert client.get('/live').status_code == 200
        with client.websocket_connect('/live/ws/alice') as websocket:
            assert websocket.receive_json()['type'] == 'ready'
            websocket.send_json({'type': 'configure', 'mode': 'interview'})
            assert websocket.receive_json()['mode'] == 'interview'
            websocket.send_json({'type': 'transcript', 'text': 'Mock interview me.'})
            response = websocket.receive_json()
            assert response['type'] == 'response' and response['conversation_id'] == 'cid'
            websocket.send_json({'type': 'unknown'})
            assert websocket.receive_json()['type'] == 'error'
    assert service.respond.call_args.kwargs['user_id'] == 'alice'
    assert service.respond.call_args.kwargs['mode'] == 'interview'
