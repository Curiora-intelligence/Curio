# Curio

Curio is one local multimodal intelligence with persistent user memory and typed tools. GPT-OSS 20B handles text and agent reasoning; Qwen3-VL 8B handles images and sampled camera observations. Both use the existing `ModelGateway` and runtime adapters. Only one heavy model is resident at a time.

The same identity, PostgreSQL memory and agent loop support service discovery, shopping and mock interviews. Providers in the included service catalogue are **fictional demo data**, not real businesses or bookings.

## Setup on this Mac

```sh
conda activate aiml
python -m pip install -r requirements-mlx.txt -r requirements-test.txt
cp .env.example .env  # only if you do not already have a .env
createdb curio
# Set DATABASE_URL in .env to your own PostgreSQL database.
python -m alembic upgrade head
# Cache the official Harmony vocabulary once while online (not model weights):
python -c 'from openai_harmony import *; load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)'
python -m scripts.serve
```

PostgreSQL must be running. Configure credentials through environment variables; no database credentials are embedded in the application. Existing `.env` values and model checkpoints are preserved. The offline launcher resolves existing cached MLX checkpoints and binds to `127.0.0.1:8001`, with one worker. It does not download weights. For another machine, use `uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1 --ws-max-size 740000` after configuring model paths.

Open [the live demo](http://127.0.0.1:8001/live) or [API documentation](http://127.0.0.1:8001/docs). Camera and microphone permissions are optional. Speech recognition availability depends on the browser and may use its vendor's speech service.

## Configuration

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL URL; `postgresql+asyncpg://…` recommended. PostgreSQL is the durable source of truth. |
| `REDIS_URL` | Optional ephemeral state, caches and locks. Missing/unreachable Redis does not stop durable chat. |
| `CURIO_BROWSER_BACKEND` | `disabled` (default) or `exa`. |
| `EXA_API_KEY` | Required for Exa. Without it browser tools return `tool_unavailable`. |
| `CURIO_MLX_TEXT_MODEL` | Existing local GPT-OSS directory or MLX model ID. |
| `CURIO_MLX_VISION_MODEL` | Existing local Qwen3-VL directory or MLX model ID. |
| `CURIO_MLX_PREFILL_STEP_SIZE` | Text prompt processing chunk, default 128; bounded to 32–512 to reduce peak unified memory. |
| `CURIO_MAX_TEXT_TOKENS` | Per-agent-iteration cap, default 1024. |
| `CURIO_MAX_VISION_TOKENS` | Regular image answer cap, default 384. |
| `CURIO_FRAME_INTERVAL` | Live cooldown after VLM inference, minimum/default 5 seconds. |
| `CURIO_EMBEDDING_MODEL` | Optional existing local SentenceTransformer directory. Unset disables embeddings. |

No OpenAI API key is needed. Runtime selection remains Apple Silicon/MLX → NVIDIA CUDA → PyTorch CPU. CPU/CUDA adapters are retained but require their own suitable checkpoints and hardware; automated tests do not establish their real inference performance.

For optional semantic retrieval, install `requirements-semantic.txt`, provide an already downloaded embedding model directory, and have a database administrator enable `CREATE EXTENSION vector` in the Curio database. Vectors are stored alongside memory and scored using pgvector cosine distance. The small demo uses casts over JSON vectors rather than an ANN index. Without embeddings or the extension, keyword/full-text retrieval continues. Existing memories gain embeddings when learned anew; no background backfill/download occurs.

## APIs

| Endpoint | Input / behavior |
| --- | --- |
| `POST /curio/analyze` | Multipart `message`, optional `image`, `conversation_id`, `user_id`, `request_id`, `mode`, and paired `latitude`/`longitude`. |
| `GET /memory/{user_id}` | Active durable memories. |
| `POST /memory/remember` | JSON `user_id`, `statement`; explicit preference/goal/constraint/skill gap or `remember …`. |
| `POST /memory/forget` | JSON `user_id`, `memory_id`; deactivates memory. |
| `POST /discovery/` | JSON `category`, optional paired coordinates, `budget_max`, `preferences`, `availability` (`today` or `any`). |
| `WS /live/ws/{user_id}` | Sampled vision, transcripts, audio metrics and agent responses. Protocol below. |
| `GET /health` | Separate `postgres`, `redis`, `models/runtime`, and `service` fields. Model status is loaded/not loaded, not a readiness claim. |

Chat response fields remain `success`, `mode`, `answer`, `conversation_id`, with added `user_id`. Image requests retain the direct Qwen path; use live observations plus a transcript when agent tools are needed for visual input. Uploads accept JPEG, PNG, WebP and GIF up to 15 MB and are removed after success or failure.

Send a stable `user_id` for cross-conversation memory. Omitting it uses `local-demo` for backwards-compatible local requests. A new conversation omits `conversation_id`; send the returned ID to continue. `request_id` replays a completed response without re-executing tools, including first requests. An interrupted run stays recorded and is not silently re-executed; use a new request ID after reviewing its status.

This is a **local hackathon API**: user IDs are client-supplied, not authentication. Bind to localhost. Before multi-user/public deployment, bind IDs to authenticated principals and add authorization/rate limits. Forgetting removes a memory from active retrieval; it does not erase source chat messages or audit history.

## Demo flows

**A — persistent biryani memory**

```sh
curl http://127.0.0.1:8001/curio/analyze -F 'user_id=demo-alice' \
  -F 'message=I prefer spicy chicken biryani and usually stay under ₹300.'
# Omit conversation_id to start a new conversation with the same person:
curl http://127.0.0.1:8001/curio/analyze -F 'user_id=demo-alice' -F "message=I'm hungry."
# Use the new conversation_id and your actual location on the next request:
curl http://127.0.0.1:8001/curio/analyze -F 'user_id=demo-alice' \
  -F 'conversation_id=REPLACE_WITH_RETURNED_ID' -F 'message=Find something nearby.' \
  -F 'latitude=17.4401' -F 'longitude=78.3489'
```

The fictional catalogue has Hyderabad-area sample coordinates. Without coordinates, distance is marked unknown; Curio must not invent your location.

**B — visual discovery:** Open `/live`, use the same user ID, choose discovery, connect, and start the camera. Wait for a structured observation of a leaking tap; ask “This is leaking. Find someone nearby who can fix it today.” Optionally share location. Curio uses the observation and the shared agent tools. Recommendations remain fictional.

**C — shopping:** Send “I prefer minimal black products and my shopping budget is ₹2500.” Start a new conversation in shopping mode, show an object, wait for an observation, then ask “Would I like this?” Curio compares visible evidence against remembered preferences; unknown prices/specifications remain unknown.

**D — interview:** Send “I struggle with database indexing.” Start a new conversation in interview mode and ask “Mock interview me for a backend internship.” Optional mic metrics and camera samples provide observable communication feedback. The system must not infer psychological state from face or voice.

## Live protocol

Client JSON messages:

```json
{"type":"configure","mode":"shopping"}
{"type":"configure","latitude":17.4401,"longitude":78.3489}
{"type":"frame","jpeg":"BASE64_JPEG","captured_at":1791020000.0}
{"type":"transcript","text":"Would I like this?"}
{"type":"metrics","metrics":{"duration_seconds":30,"speaking_seconds":22,"pause_count":3,"rms":[0.04,0.05]}}
{"type":"reason","text":"Give feedback on my answer."}
{"type":"ping"}
```

`captured_at` is Unix seconds and optional; supplied stale timestamps are rejected. Frames must be JPEG, ≤512 KB, ≤1280 pixels per side. The client shows a 30 FPS preview but samples at most once per 5-second cooldown. The server drops frames during inference and returns `busy` for concurrent questions (retry after the response). No raw audio is sent. Final browser transcripts are sent on pressing **Send**, avoiding a GPT turn for every partial utterance.

Server events: `ready`, `configured`, `observation`, `metrics`, `response`, `frame_skipped`, `busy`, `error`, `pong`. Only transcript/reason messages trigger GPT-OSS. Visual observations expire after 60 seconds; Redis session metadata expires automatically. On a 16 GB Mac, a model switch can be slow; pause camera sampling while having a longer text conversation.

## Tests

```sh
python -m pytest -q
# Real PostgreSQL integration: use a disposable migrated database with pgvector enabled.
DATABASE_URL=postgresql+asyncpg://localhost/curio_test python -m alembic upgrade head
psql curio_test -c 'CREATE EXTENSION IF NOT EXISTS vector'
CURIO_TEST_DATABASE_URL=postgresql+asyncpg://localhost/curio_test python -m pytest -q
DATABASE_URL=postgresql+asyncpg://localhost/curio_test python -m alembic check
# Actual cached MLX inference, model switching, and a persisted tool call:
python -m scripts.verify_runtime --mode both
# Actual sampled Qwen observation → persistent shopping answer over WebSocket:
python -m scripts.verify_live
```

Unit/API tests use stub model output and temporary SQLite databases. PostgreSQL tests separately verify persistence in another process, optional vector retrieval, and a real Redis connection failure. The model smoke test requires local PostgreSQL, cached weights, Harmony vocabulary, and Metal access. Results are written to `test-results/persistent-runtime.json`. No CUDA/CPU model smoke is implied.

See [architecture](docs/architecture.md) for flow, contracts and boundaries, and [implementation report](docs/implementation-report.md) for observed verification results.

## Asynchronous chat

Normal clients can now use `POST /curio/runs` followed by SSE at `/curio/runs/{request_id}/events?user_id=...`. Poll `/curio/runs/{request_id}?user_id=...` to recover the durable answer. `/curio/analyze` is unchanged. See [SSE contract and deployment notes](docs/sse.md).
