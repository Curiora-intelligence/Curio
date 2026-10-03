# Asynchronous chat and progress

`POST /curio/runs` accepts the same multipart fields as `/curio/analyze` and returns HTTP 202 with `{request_id, status: "accepted"}`. The returned UUID identifies the existing PostgreSQL AgentRun row. An optional input `request_id` is an idempotency key scoped to the conversation; without a conversation it deterministically identifies the initial conversation for that user, matching analyze semantics. Reposting the same key returns the original run. Image files remain available until the background execution finishes.

`GET /curio/runs/{request_id}?user_id=demo-sai` returns accepted/running/completed/failed plus answer, conversation_id and user_id. The internal six-step `iteration_limit` result maps to completed and retains its explanatory final answer. Unknown or differently owned runs return 404. User IDs are demo scoping, not authentication.

`GET /curio/runs/{request_id}/events?user_id=demo-sai` serves named SSE events with Cache-Control: no-cache, Connection: keep-alive, X-Accel-Buffering: no, and keep-alive comments every approximately 15 seconds. EventSource handles Last-Event-ID automatically. IDs are monotonically increasing `N-0` Redis Stream IDs. Redis keys are `curio:events:{request_id}`, bounded to 256 entries, renewed with a one-hour TTL including completion. Stage messages come from a fixed allowlist; actual tool-handler calls expose name and status only. No tool arguments, results, prompts or model analysis enter progress events.

Without Redis, execution and polling still work. If transient events are absent/expired, SSE recovers the committed final answer (or safe error) and done from PostgreSQL, using reserved IDs 1000000-0 and 1000001-0. Clients should handle a repeated final idempotently because a stream can fail around the durable commit. No database migration is required.

The single-process in-memory scheduler is suitable for this local hackathon demo. Graceful shutdown waits for tasks and their image/model resources. Hard-killed accepted/running requests are not automatically resumed; their final answer cannot be recovered unless it was committed before the interruption. Do not restart the backend during a demo run. A durable worker queue/lease is future production work, not part of this scheduler.

Set `CURIO_FRONTEND_ORIGINS=http://127.0.0.1:8000,http://localhost:8000` for browser access (these are the defaults). Use one backend worker for local MLX. Existing analyze and Live routes remain available.

Validation (2026-10-03): `CURIO_TEST_DATABASE_URL=postgresql+asyncpg://curio_test@127.0.0.1:55432/curio_test CURIO_TEST_REDIS_URL=redis://127.0.0.1:56379/0 python -m pytest -q` → **72 passed in 2.11s**, including all original 57 tests. The test instances are isolated from the application's default database.

Real-model HTTP smoke: accepted in 0.055 seconds, final/done in 52.61 seconds, durable status and Last-Event-ID replay verified. The existing Live protocol was also checked with a synthetic frame (19.41 seconds to a Qwen observation), busy/frame-skipped responses, shopping and interview. A narrow final shopping check now rejects affordability claims without a visible price or a successful price-bearing discovery result; direct image analysis remains unchanged.
