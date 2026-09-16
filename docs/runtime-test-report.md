# Curio runtime verification — 16 September 2026

**Both cached models passed real inference through the extracted Curio API on this Mac using MLX.** These are smoke-test results, not comprehensive model-quality or production-load benchmarks.

## Observed results

| Request, in execution order | Runtime | Expected / actual answer | HTTP | Time including loading |
| --- | --- | --- | --- | --- |
| GPT-OSS 20B: `What is 2 + 2? Answer with only the number.` | MLX | `4` / `4` | 200 | 40.30 s |
| Qwen3-VL 8B: white 256×256 image containing a red square; ask its color | MLX | `red` / `red` | 200 | 16.64 s |
| GPT-OSS 20B after switching back from Qwen | MLX | `4` / `4` | 200 | 65.41 s |

All three requests ran in the same process and gateway. Resident mode changed `text → vision → text`. Runtime logs showed Metal active/cache memory returning to zero between models and on shutdown. The API lifespan released the gateway at shutdown. Raw structured results: [`inference.json`](../test-results/inference.json).

An earlier standalone GPT-OSS API request also passed in 41.38 seconds: [`text-initial.json`](../test-results/text-initial.json).

## Environment and checkpoints

- Apple Silicon, macOS 27.0, 16 GB unified memory, Python 3.14.7.
- MLX 0.32.2, MLX-LM 0.31.3, MLX-VLM 0.6.17, Transformers 5.16.1, FastAPI 0.141.1.
- GPT-OSS: `mlx-community/gpt-oss-20b-MXFP4-Q8`, cached revision `773a7da77e569019bb0fd17a554b263738d669a3`.
- Qwen: `mlx-community/Qwen3-VL-8B-Instruct-8bit`, cached revision `a0093b9b5fda6f76ddd4a462c6830ae7c4fe47ec`.
- No model download or model-cache mutation was performed.
- Text output cap 256 tokens; vision output cap 96 tokens. Campus system prompt and temperature 0.2 retained.

## API and extraction checks

Eight contract tests passed: health/OpenAPI, missing input, text history, vision routing, temporary-file cleanup on success/failure, upload type/empty/size rejection, corrected vision-helper argument, and lazy imports without MLX or Torch initialization. Contract tests stub model generation and are separate from the real inference tests above. Python compilation passed.

Campus remained unchanged. The empty CPU placeholder was excluded; the existing Torch adapter provides the CPU route. Curio now has its own entry point, requirements, offline launcher, test suite and documentation.

## Limits and warnings observed

- **CUDA and PyTorch CPU inference are unverified.** This Mac has no NVIDIA CUDA GPU and only the MLX checkpoints are cached. No full-precision 20B/8B checkpoints were downloaded or loaded on CPU. Retained adapters are not evidence those paths work.
- The execution sandbox could not access Metal. Actual GPU tests succeeded outside the sandbox with authorized local GPU access.
- GPT-OSS used about 11,520 MiB of Metal model memory; Qwen about 9,400 MiB. The loader warned GPT-OSS approaches the recommended Metal working-set limit. Timings include loading and memory pressure; they are not tokens-per-second benchmarks.
- The Qwen cached index names four shards, while the cache has two differently named shards. The installed MLX-VLM loader falls back to the available safetensors files, loads the model and passes inference. The shared cache was left intact; compatibility with other loader versions is unverified.
- Hugging Face's full-snapshot offline check reports missing GPT-OSS README/attributes files. Local-path model loading succeeds because inference assets are present. The launcher uses local snapshot paths.
- Nonfatal MLX memory-API deprecation notices and a Transformers processor-kwargs warning appeared. Neither prevented successful inference.
- Multi-turn multimodal behavior, long-context quality, concurrent users and sustained throughput were not tested. Conversation history retains text only.
