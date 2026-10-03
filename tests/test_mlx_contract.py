"""Exercise the real adapter with fake MLX imports, without allocating a GPU."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from app.runtimes.harmony import encoding


def test_mlx_keeps_action_stop_token_and_restores_tokenizer(monkeypatch):
    imports = {
        'mlx_lm': {'stream_generate': Mock(), 'load': Mock()},
        'mlx_lm.sample_utils': {'make_sampler': Mock(return_value='sampler')},
        'mlx_vlm': {'generate': Mock(), 'load': Mock()},
        'mlx_vlm.prompt_utils': {'apply_chat_template': Mock()},
        'mlx_vlm.utils': {'load_config': Mock()},
    }
    for name, values in imports.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location('isolated_mlx_runtime', Path(__file__).parents[1] / 'app/runtimes/mlx_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.MLXRuntime()
    runtime._ensure_text_model = Mock()
    runtime._tokenizer = SimpleNamespace(eos_token_ids={123})
    runtime._model = object()
    tokens = encoding().encode('<|channel|>final<|message|>4<|return|>', allowed_special='all')
    def stream(*args, **kwargs):
        assert set(args[1].eos_token_ids) == set(encoding().stop_tokens_for_assistant_actions())
        assert 32 <= kwargs['prefill_step_size'] <= 512
        for token in tokens:
            yield SimpleNamespace(token=token)
    module.llm_stream_generate = stream
    result = runtime.generate_turn(model_id='cached-model', messages=[{'role': 'user', 'content': '2+2?'}], tools=[], max_tokens=32, temperature=0)
    assert result.final == '4'
    assert runtime._tokenizer.eos_token_ids == {123}
