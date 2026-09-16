# Curio

Standalone local inference backend extracted from **Curiora Campus**. Text requests go to **GPT-OSS 20B**; image-only and image-plus-text requests go to **Qwen3-VL 8B**. This folder contains the actual model gateway, runtime adapters, conversation service, vision helper, and FastAPI endpoint. Campus remains intact.

## Run on this Mac

From this folder, use the existing environment and cached model weights:

```sh
conda activate aiml
python -m scripts.run_local
```

Open <http://127.0.0.1:8001/docs> for the interactive API. The launcher binds to localhost, uses one worker, and disables Hugging Face network access. It resolves the cached model snapshots before starting. Model weights load on the first inference request, not at server startup.

If the environment is not activated, use:

```sh
/Users/saiganeshsattenapalli/miniforge3/envs/aiml/bin/python -m scripts.run_local
```

On another Apple Silicon Mac, create an environment and install `requirements-mlx.txt`. Acquire the two checkpoints separately, or set `CURIO_MLX_TEXT_MODEL` and `CURIO_MLX_VISION_MODEL` to existing local model directories. The local launcher does not download weights.

## API

- `GET /health`: server liveness; does not claim either model is loaded or ready.
- `POST /curio/analyze`: multipart form with `message`, optional `image`, and optional `conversation_id`.
- `GET /docs`, `GET /openapi.json`: generated API documentation.

```sh
curl http://127.0.0.1:8001/curio/analyze \
  -F 'message=What is 2 + 2?'

curl http://127.0.0.1:8001/curio/analyze \
  -F 'message=Describe the visible image.' \
  -F 'image=@/absolute/path/to/image.png'
```

Successful responses contain `success`, `mode`, `answer`, and `conversation_id`. Send the returned conversation ID with the next request to reuse text history. Images are temporary and are not retained for follow-up turns. JPEG, PNG, WebP and GIF MIME types are accepted, up to 15 MB. Temporary images are removed after successful or failed inference.

## Architecture

```text
Input → validation → modality routing → model gateway
      → runtime → inference → response / temporary-file cleanup

Text  → GPT-OSS 20B
Image → Qwen3-VL 8B
```

`ModelGateway` serializes inference and the adapters keep one model resident at a time. Automatic runtime selection prefers Apple Silicon/MLX, then NVIDIA CUDA, then PyTorch CPU. MLX imports are lazy so importing the API does not require an Apple GPU.

| Runtime | Text model | Vision model |
| --- | --- | --- |
| MLX | `mlx-community/gpt-oss-20b-MXFP4-Q8` | `mlx-community/Qwen3-VL-8B-Instruct-8bit` |
| PyTorch adapters | `openai/gpt-oss-20b` | `Qwen/Qwen3-VL-8B-Instruct` |

The PyTorch adapters are retained from Campus; they have **not** been validated with real inference on this Mac. Their checkpoints and NVIDIA hardware are unavailable here. `requirements-torch.txt` describes their dependencies, not a promise of CPU/CUDA compatibility.

Product direction remains **Perceive → Understand → Route → Act → Verify**. This extraction provides inference APIs; it does not implement an agent action or verification loop.

## Tests

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -v
python -m scripts.smoke_test --mode both
```

Contract tests use stubs for model output. The separate smoke test uses actual cached weights and FastAPI requests: arithmetic via GPT-OSS, a synthetic red-square image via Qwen, then arithmetic after switching back to GPT-OSS. It writes responses, HTTP status, timing, package versions and shutdown-release status to `test-results/inference.json`. A failure exits nonzero. Run it in a normal local shell with GPU access.

See [runtime test report](docs/runtime-test-report.md) for observed results and limitations.

## Configuration and limits

Settings are loaded automatically from `.env` in this folder by both the local launcher and `main.py`. Explicit shell environment variables take precedence. This Mac's `.env` contains the cached checkpoint paths; it is ignored by Git. Use `.env.example` as the template on another machine. No API key is needed for these local models.

- `CURIO_MLX_TEXT_MODEL` / `CURIO_MLX_VISION_MODEL`: local model directories or model IDs when running `uvicorn main:app` directly.
- `CURIO_MAX_TEXT_TOKENS`: output cap, default 1024.
- `CURIO_MAX_VISION_TOKENS`: output cap, default 384.
- The smoke test uses smaller output caps (256/96) and no downloads.
- These models are large for 16 GB unified memory. Cold requests and model switches include loading time. Use one worker and avoid loading both simultaneously.
- Conversations live in process memory, with no persistence, authentication, expiry or per-user isolation. Run locally; production hosting needs those controls and bounded history.
- The API validates MIME and size; decoding is left to the inference processor. File uploads are read into memory before the size check.
- GPT-OSS responses require an explicit final channel. Truncated/malformed output is treated as an error instead of a successful fallback answer.

## Source

Copied from local `Curiora-Campus`, commit `fe8b1e9a916b2cec5f294179c66754b2c0fe61ac`. No Campus source files, credentials, database setup, frontend assets or weight files were moved or changed. The empty Campus `cpu_runtime.py` placeholder was omitted; CPU routing uses `TorchRuntime`.

The standalone copy adds a FastAPI entry point, local launcher, dependencies and tests. It fixes the vision helper's outdated `prompt=` argument, makes runtime imports lazy, allows model paths/output caps to be configured, and removes raw GPT-OSS fallback logging.
