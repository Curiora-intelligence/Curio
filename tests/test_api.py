"""API contract tests use a stub gateway; real inference is tested separately."""
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from app.routers.curio import curio_service
from app.services.curio import CONVERSATIONS
from app.services.vision import VisionService


class APITests(unittest.TestCase):
    def setUp(self):
        CONVERSATIONS.clear()
        self.client = TestClient(app)

    def test_health_and_openapi(self):
        self.assertEqual(self.client.get('/health').json()['status'], 'ok')
        self.assertIn('/curio/analyze', self.client.get('/openapi.json').json()['paths'])

    def test_missing_input(self):
        self.assertEqual(self.client.post('/curio/analyze', data={'message': ' '}).status_code, 400)

    def test_text_and_conversation_history(self):
        with patch.object(curio_service.gateway, 'generate_text', return_value='4') as generate:
            first = self.client.post('/curio/analyze', data={'message': '2 + 2?'})
            self.assertEqual(first.status_code, 200)
            cid = first.json()['conversation_id']
            second = self.client.post('/curio/analyze', data={'message': 'Again?', 'conversation_id': cid})
            self.assertEqual(second.json()['conversation_id'], cid)
            self.assertEqual(second.json()['mode'], 'text')
            self.assertEqual([x['role'] for x in generate.call_args.kwargs['messages']],
                             ['system', 'user', 'assistant', 'user'])

    def test_vision_routing_and_temp_cleanup(self):
        paths = []
        def generate(**kwargs):
            path = Path(kwargs['image_path'])
            self.assertTrue(path.is_file())
            paths.append(path)
            return 'red'
        with patch.object(curio_service.gateway, 'generate_vision', side_effect=generate):
            response = self.client.post('/curio/analyze', files={'image': ('test.png', b'fake-image', 'image/png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['mode'], 'vision')
        self.assertFalse(paths[0].exists())

    def test_cleanup_on_inference_failure(self):
        paths = []
        def fail(**kwargs):
            paths.append(Path(kwargs['image_path']))
            raise RuntimeError('test failure')
        with patch.object(curio_service.gateway, 'generate_vision', side_effect=fail):
            response = self.client.post('/curio/analyze', files={'image': ('test.png', b'fake-image', 'image/png')})
        self.assertEqual(response.status_code, 500)
        self.assertFalse(paths[0].exists())

    def test_upload_validation(self):
        for content, mime, status in [(b'', 'image/png', 400), (b'x', 'text/plain', 415)]:
            with self.subTest(mime=mime, status=status):
                response = self.client.post('/curio/analyze', files={'image': ('test', content, mime)})
                self.assertEqual(response.status_code, status)
        with patch('app.routers.curio.MAX_IMAGE_SIZE', 2):
            response = self.client.post('/curio/analyze', files={'image': ('test.png', b'123', 'image/png')})
            self.assertEqual(response.status_code, 413)

    def test_vision_helper_uses_messages_contract(self):
        with patch.object(curio_service.gateway, 'generate_vision', return_value='ok') as generate:
            VisionService(curio_service.gateway).analyze(__file__, 'Describe it')
            self.assertIn('messages', generate.call_args.kwargs)
            self.assertNotIn('prompt', generate.call_args.kwargs)

    def test_runtime_import_is_lazy(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-c',
            'import sys; import main; assert "mlx.core" not in sys.modules; assert "torch" not in sys.modules'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
