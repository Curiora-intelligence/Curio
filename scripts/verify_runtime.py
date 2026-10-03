"""Real cached MLX inference through the API; never downloads model weights."""
import argparse
import asyncio
import io
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['text', 'vision', 'tools', 'both'], default='both')
    parser.add_argument('--output', default='test-results/persistent-runtime.json')
    args = parser.parse_args()
    load_dotenv()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CURIO_MAX_TEXT_TOKENS'] = '1024'
    os.environ['CURIO_MAX_VISION_TOKENS'] = '96'
    from app.core.models import cached_model
    for mode, repo in [('TEXT', 'mlx-community/gpt-oss-20b-MXFP4-Q8'), ('VISION', 'mlx-community/Qwen3-VL-8B-Instruct-8bit')]:
        if args.mode in ['both', mode.lower()] or (args.mode == 'tools' and mode == 'TEXT'):
            os.environ.setdefault('CURIO_MLX_' + mode + '_MODEL', cached_model(repo))
    from fastapi.testclient import TestClient
    from PIL import Image, ImageDraw
    from sqlalchemy import select
    from app.persistence.database import Database
    from app.persistence.models import ToolCall, AgentRun
    from app.routers.curio import curio_service
    from main import app
    report = {'tests': []}
    # Record call-site metadata on failure without dumping model inputs or reasoning.
    original_generate = curio_service.gateway.generate_turn
    def diagnostic_generate(**kwargs):
        try:
            return original_generate(**kwargs)
        except Exception as exc:
            import traceback
            report['runtime_error'] = {'type': type(exc).__name__, 'frames': [
                {'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                for frame in traceback.extract_tb(exc.__traceback__)]}
            if isinstance(exc, RuntimeError):
                report['runtime_error']['message'] = str(exc)[:500]
            print(json.dumps(report['runtime_error']), flush=True)
            raise
    curio_service.gateway.generate_turn = diagnostic_generate
    user_id = 'runtime-smoke-' + str(uuid4())
    async def calls_for(cid):
        db = Database()
        async with db.sessions() as session:
            calls = (await session.scalars(select(ToolCall.name).join(AgentRun, AgentRun.id == ToolCall.agent_run_id)
                                           .where(AgentRun.conversation_id == cid))).all()
        await db.close()
        return list(calls)
    modes = ['text', 'vision', 'text', 'tools'] if args.mode == 'both' else [args.mode]
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    with TestClient(app) as client:
        for mode in modes:
            start = time.monotonic()
            print('Starting real ' + mode + ' inference', flush=True)
            try:
                if mode == 'tools':
                    remembered = client.post('/memory/remember', json={'user_id': user_id, 'statement': 'I prefer spicy chicken biryani and usually stay under ₹300.'})
                    assert remembered.status_code == 200, remembered.text
                    response = client.post('/curio/analyze', data={'user_id': user_id, 'message': 'Use services.search to find fictional demo food providers. My coordinates are 17.4401, 78.3489. Apply my food preferences and budget, and summarize the first result in two sentences.'})
                elif mode == 'vision':
                    image = Image.new('RGB', (256, 256), 'white')
                    ImageDraw.Draw(image).rectangle((64, 64, 192, 192), fill='red')
                    buffer = io.BytesIO(); image.save(buffer, format='PNG')
                    response = client.post('/curio/analyze', data={'user_id': user_id, 'message': 'What color is the square? Answer with one word.'}, files={'image': ('square.png', buffer.getvalue(), 'image/png')})
                else:
                    response = client.post('/curio/analyze', data={'user_id': user_id, 'message': 'What is 2 + 2? Answer with only the number.'})
                body = response.json()
                answer = body.get('answer', '').lower().strip().rstrip('.! ')
                calls = asyncio.run(calls_for(body['conversation_id'])) if mode == 'tools' and response.status_code == 200 else []
                passed = response.status_code == 200 and (('services.search' in calls and 'fictional' in answer) if mode == 'tools' else answer == ('red' if mode == 'vision' else '4'))
                entry = {'mode': mode, 'status': response.status_code, 'response': body, 'tool_calls': calls, 'passed': passed}
            except Exception as exc:
                entry = {'mode': mode, 'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)}
            entry['seconds'] = round(time.monotonic() - start, 2)
            report['tests'].append(entry)
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(entry), flush=True)
    report['released_on_shutdown'] = curio_service.gateway._runtime is None
    output.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if all(t['passed'] for t in report['tests']) else 1)


if __name__ == '__main__':
    main()
