"""Actual Qwen observation → persistent Curio shopping answer over WebSocket."""
import base64
import io
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from dotenv import load_dotenv


def main():
    load_dotenv()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CURIO_MAX_TEXT_TOKENS'] = '1024'
    from app.core.models import cached_model
    for mode, repo in [('TEXT', 'mlx-community/gpt-oss-20b-MXFP4-Q8'), ('VISION', 'mlx-community/Qwen3-VL-8B-Instruct-8bit')]:
        os.environ['CURIO_MLX_' + mode + '_MODEL'] = cached_model(repo)
    from fastapi.testclient import TestClient
    from PIL import Image, ImageDraw
    from main import app
    from app.routers.curio import curio_service
    image = Image.new('RGB', (384, 384), 'white')
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 130, 314, 330), radius=12, fill='black')
    draw.arc((124, 60, 260, 200), 180, 360, fill='black', width=18)
    buffer = io.BytesIO(); image.save(buffer, format='JPEG')
    user_id = 'live-smoke-' + str(uuid4())
    report = {}
    with TestClient(app) as client:
        remembered = client.post('/memory/remember', json={'user_id': user_id, 'statement': 'I prefer minimal black products and my shopping budget is ₹2500.'})
        assert remembered.status_code == 200, remembered.status_code
        with client.websocket_connect('/live/ws/' + user_id) as ws:
            assert ws.receive_json()['type'] == 'ready'
            ws.send_json({'type': 'configure', 'mode': 'shopping'})
            assert ws.receive_json()['type'] == 'configured'
            start = time.monotonic()
            ws.send_json({'type': 'frame', 'jpeg': base64.b64encode(buffer.getvalue()).decode(), 'captured_at': time.time()})
            observation = ws.receive_json()
            report['observation'] = observation
            report['observation_seconds'] = round(time.monotonic() - start, 2)
            print(json.dumps(observation), flush=True)
            assert observation['type'] == 'observation' and 'unavailable' not in observation['observation']
            start = time.monotonic()
            ws.send_json({'type': 'transcript', 'text': 'Would I like this? Explain using my preferences. State whether you can determine the price.'})
            response = ws.receive_json()
            report['response'] = response
            report['response_seconds'] = round(time.monotonic() - start, 2)
            print(json.dumps(response), flush=True)
            assert response['type'] == 'response'
            assert 'black' in response['answer'].lower()
            report['passed'] = True
    report['released_on_shutdown'] = curio_service.gateway._runtime is None
    Path('test-results').mkdir(exist_ok=True)
    Path('test-results/live-runtime.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
