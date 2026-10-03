# Implementation and verification report

Date: 3 October 2026. Work performed in the existing Curio checkout, preserving the gateway/runtime architecture and local edits. No commits, pushes, model downloads, credentials, or production database changes were made.

## Milestone 1 — PostgreSQL memory

Changed: `app/services/curio.py`, `app/routers/curio.py`, `main.py`, dependencies, `.env.example`; added `app/persistence/`, `app/services/memory.py`, `app/services/ephemeral.py`, memory routes and Alembic migration `0001`.

Verification: first migration applied to an isolated PostgreSQL 18 server on port 55432. **9 tests passed**, including reading memories in another Python process, new-conversation recall, irrelevant-memory filtering, supersession, user isolation and Redis outage fallback.

Risks: extraction intentionally covers a narrow explicit grammar. User IDs are scoped but not authenticated. PostgreSQL is required; there is no silent in-memory durability fallback.

## Milestone 2 — GPT-OSS agent loop

Changed: MLX/Torch runtime adapters, RuntimeAdapter, ModelGateway and CurioService; added `app/runtimes/harmony.py`, `turn.py`, `app/agent/runtime.py`, registry and memory tools.

Verification: **26 tests passed** with PostgreSQL enabled. Includes one and multiple tool calls, failures returned to the model, malformed arguments, permission checks, six-iteration exhaustion, final-only output and Harmony continuation rendering. Legacy API behavior coverage was recreated under new test filenames because the checkout already had the original tests deleted.

Risk discovered by real inference: larger prompts containing tool schemas/context exhausted Metal memory during default-size prefill. Fixed by processing prompts in 128-token chunks and releasing models on generation errors. This preserves the model, quantization, gateway lock and one-model residency architecture.

## Milestone 3 — service discovery and browser

Changed: `app/tools/discovery.py`, `app/tools/browser.py`, discovery router and service registration.

Verification: **34 passed, 1 PostgreSQL test skipped** in the sandbox at this stage. Added deterministic ranking, exact weights, hard budget/availability filtering, reasons, fictional labels, Exa search/open/find adapter contract and private-URL rejection. Browser responses were tested with a mock transport.

Risks: catalogue data is fictional; real provider access needs a configured Exa key. No external bookings/actions are implemented. Real Exa account access was not exercised.

## Milestone 4 — live multimodal mode

Changed: live router/service, coaching schemas/metrics/guards, `/live` HTML client, mode/location inputs, documentation and live tests.

Verification: **45 passed, 1 PostgreSQL test skipped** at the initial live milestone. Tests cover sampled-frame throttling, temporary-file cleanup on success/failure, stale/invalid frames, mode/user propagation, observable metrics and deterministic rejection of mental-state labels. Subsequent WebSocket and integration tests cover the shared Curio service.

Risks: browser camera/microphone permissions and speech recognition are browser-dependent. Automated tests use synthetic frames rather than a physical camera, real interview participant or leaking tap. Sparse observations cannot establish psychological state or unsupported product facts. Model adherence needs continued evaluation.

## Final regression and real inference

- **57 tests passed, zero skipped** against the isolated PostgreSQL database with pgvector enabled.
- PostgreSQL durability verified across repository and separate process instances.
- pgvector similarity retrieval and an actual refused Redis connection verified.
- `alembic check`: **No new upgrade operations detected.**
- Python compilation, `git diff --check`, and live JavaScript syntax check passed.
- Real cached-model API smoke: **4/4 passed**, with successful gateway release at shutdown.

| Real API check | Result | Seconds |
| --- | --- | ---: |
| GPT-OSS arithmetic | HTTP 200, `4` | 39.89 |
| Qwen synthetic red square | HTTP 200, `red` | 13.41 |
| GPT-OSS after switching back | HTTP 200, `4` | 30.35 |
| Remembered food preferences/budget → `services.search` → final | HTTP 200; persisted call; fictional ₹280 recommendation within ₹300 budget | 87.00 |

Timings include model loading/switching and memory pressure; they are not throughput benchmarks. See `test-results/persistent-runtime.json`. Earlier failed runs identified the Metal prefill allocation issue; the table records the successful run after that fix. CUDA/CPU inference remains unverified.

The original deleted `tests/test_api.py`, launcher/smoke scripts, runtime report and old inference artifacts were left deleted. A separate in-session edit to the vision helper's token cap was preserved. New launcher and verification scripts are `scripts/serve.py`, `scripts/verify_runtime.py` and `scripts/verify_live.py`.


## Real live WebSocket smoke

`python -m scripts.verify_live` passed using a synthetic black silhouette, actual cached Qwen and GPT-OSS, the live WebSocket route, and PostgreSQL memories. Qwen produced structured visible features; GPT-OSS recalled minimal black products and the ₹2,500 ceiling, and explicitly stated that the item's price was unknown. The gateway released both models at shutdown.

The silhouette was intended as a simple bag-like shape but Qwen classified it as a padlock. This establishes pipeline integration and preference/price grounding, not broad object-recognition accuracy. No physical camera, microphone, leaking tap or interview participant was used. See `test-results/live-runtime.json`.

Observed timings: visual observation 21.68 seconds; reasoning/model switch 32.7 seconds.
