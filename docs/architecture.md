# Persistent Curio architecture

## Perceive → Remember → Understand → Route → Act → Verify

1. **Perceive:** text enters the existing multipart API. Images retain the direct Qwen3-VL path. Live camera frames are compressed and sampled by the browser; Qwen returns bounded observations. Transcripts and aggregate RMS/timing metrics arrive separately.
2. **Remember:** PostgreSQL commits the current user message before inference. Explicit durable statements are extracted conservatively. Active memories are retrieved with user-scoped keywords/full-text search, optionally augmented by pgvector.
3. **Understand:** relevant memories and live observations enter as lower-priority context data. Curio's identity stays in application instructions. GPT-OSS interprets intent through the existing gateway.
4. **Route:** the model selects typed tools from a registry. Public dotted names such as `services.search` map to valid Harmony function names such as `functions.services_search`.
5. **Act:** the agent validates names, Pydantic arguments and permission class, commits a ToolCall audit record, executes the handler, commits its result, and appends a Harmony tool response. Errors become tool results.
6. **Verify:** only a complete final channel reaches the user. Final answers and AgentRun status are committed together. Exhaustion returns a bounded-step message. Interview output passes an additional deterministic label guard. Tests separately check model-independent behavior and real cached inference.

## Boundaries

```text
FastAPI routers / live demo
  ├─ CurioService
  │    ├─ ConversationRepository / MessageRepository / MemoryRepository
  │    │    └─ SQLAlchemy async sessions → PostgreSQL
  │    ├─ MemoryService → optional EmbeddingProvider
  │    └─ AgentRuntime (at most 6 model iterations)
  │         ├─ ToolRegistry → typed handlers
  │         │    ├─ memory tools → MemoryService / repositories
  │         │    ├─ service tools → deterministic fictional catalogue
  │         │    └─ web tools → BrowserBackend → Exa
  │         └─ ModelGateway → MLX / CUDA / CPU adapters
  │              ├─ GPT-OSS 20B (Harmony)
  │              └─ Qwen3-VL 8B
  └─ LiveSession → sampled VLM observations → shared CurioService

Redis: disposable live metadata, run state, retrieval/tool/page caches, frame locks
```

No LangChain/LangGraph, arbitrary shell execution, model-generated SQL, or alternate model stack is introduced. Runtime detection in `app/core/runtime.py` and the one-heavy-model residency design remain intact. A gateway thread lock serializes model loading, inference, and release. Production should use one inference worker on this Mac; Redis does not coordinate model residency across independent processes.

## Durable state and transaction boundaries

`User`, `Conversation`, `Message`, `Memory`, `AgentRun`, and `ToolCall` are SQLAlchemy entities. Alembic owns schema changes; startup never silently creates tables or enables extensions. Conversations verify the owning user ID, and memory queries always scope by user.

A PostgreSQL advisory lock orders a user's turns across workers. Its dedicated unpooled connection avoids consuming the transaction pool while waiting for model inference. Short database transactions commit input/memories, tool attempts/results, and final response/status separately. The assistant does not hold a write transaction while generating. A crashed process releases its database session lock; incomplete runs/tool calls remain visible in the audit tables. They are not silently replayed.

Request IDs are unique within a conversation. An initial request ID deterministically selects its conversation so an initial retry can also replay a completed answer. Retries of failed/incomplete runs are rejected rather than repeating potentially consequential work. There are currently no external-action tools.

Redis is never authoritative for memory or execution audit. Retrieval cache keys include the durable user's memory version; writes/forget operations bump the version, so a Redis outage during invalidation cannot resurrect stale cached memories when Redis recovers. Cache errors/timeouts fall back to PostgreSQL. Browser pages have a bounded, expiring in-process fallback when Redis is absent.

## Memory policy and retrieval

The extractor accepts bounded explicit first-person patterns for preferences, budgets, career goals and interview skill gaps. Explicit `remember …` statements can retain a user-provided fact. Quoted/code/control-token content is excluded. Model-inferred memory is deliberately not enabled for the hackathon; no extra model switch is needed to extract memory.

Conflicts on `(user_id, kind, key)` deactivate the previous value and create a new sourced record. A partial unique index permits only one active value. Exact repeated values are deduplicated. `source_message_id`, confidence, importance and timestamps preserve provenance. Forget is a soft deactivation, not transcript erasure.

Retrieval first filters for matching terms and a small set of intent domains (e.g. hungry → food, internship → career/interview). Short referential follow-ups can use bounded recent user context. Zero-relevance records are excluded. Scores combine 55% relevance, 20% importance, 15% recency and 10% memory-type weight. Only six results are injected; prior chat history is bounded by message count and character budget. Current user input stays intact.

`EmbeddingProvider` is an async interface. `LocalEmbeddings` optionally loads an existing CPU SentenceTransformer. Stored JSON vectors allow the base schema to start without pgvector; the optional repository query casts vectors and computes cosine distance when the extension exists. Missing/incompatible vectors, extension, package or model fall back to keyword retrieval. The demo has no ANN index or embedding backfill; a larger deployment should use a dimensioned vector column/index and version its embedding model.

## Harmony compatibility

The official `openai-harmony` package renders prompts to token IDs and parses generated token IDs. Function schemas belong in a developer message; tool results use the exact recipient name as their author. Tool/analysis messages remain in transient continuation state, with automatic analysis dropping disabled during an active agent turn. They are never copied into user-visible answers or exposed by APIs.

A completed tool handoff and a completed final answer have different terminal tokens; ordinary message-end tokens do not end generation. MLX `stream_generate` includes the action stop token in its last response chunk, which must be retained even though the decoded text omits it. Torch passes the same action stop IDs to `generate`. Prompt prefill uses 128-token chunks by default to avoid the measured GPU memory exhaustion with larger tool/context prompts on this 16 GB Mac. Generation errors release the resident model before retry. A length stop, missing final channel, wrong role, or recipient/stop mismatch is an error rather than a guessed answer.

Special-token strings in user/context/tool content are escaped before rendering to prevent forged Harmony headers. Raw model continuation is internal structured data. Tool names and arguments still pass validation even when Harmony parsing succeeds. The vocabulary must be cached once before offline use.

Sources: [official Harmony format](https://developers.openai.com/cookbook/articles/openai-harmony), [official GPT-OSS inference example](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers).

## Tools and permissions

Each registry entry has a name, description, argument schema, permission class and async handler. Names/aliases are unique. Unknown tools, invalid schemas, failures and timeouts yield structured tool errors. The loop allows six model iterations and at most eight calls per generated turn. SQL and shell execution are not exposed.

- `READ_ONLY`: automatic execution; eligible results may be cached.
- `USER_DATA_WRITE`: current user statement must exactly match tool arguments, then pass conservative extraction. Tool arguments cannot choose another user or source-message ID.
- `EXTERNAL_ACTION`: denied unless a server-owned confirmation fingerprint matches the exact tool and validated arguments. No external-action handler or confirmation UI is shipped; future action tools must implement a user confirmation flow before adding their fingerprints.

Service ranking is deterministic: 30% rating, 25% distance, 20% budget fit, 15% availability, 10% preference match. Explicit budget ceilings and same-day requests also filter candidates. Missing distance/budget/preferences receive documented neutral scores. Distance decreases linearly to zero at 20 km. Ties sort by provider ID. Every result includes components and reasons; all provider data is fictional.

Browser tools use Exa's documented [search](https://exa.ai/docs/reference/search) and [contents](https://exa.ai/docs/reference/get-contents) APIs. Only this adapter makes web requests. Arbitrary URLs are submitted to Exa; Curio itself never fetches those hosts or follows redirects. Local/private URL literals are rejected. Returned text is bounded, labeled untrusted and used only as context. No Google HTML scraping is performed.

## Live mode

The browser owns camera preview and speech recognition. It sends compressed JPEG samples, final transcripts and RMS/timing aggregates; no continuous video/audio stream reaches a model. One session schedules one inference at a time. While it runs, the WebSocket receiver discards incoming frames and rejects concurrent questions with `busy`. A five-second cooldown begins after VLM inference. Redis adds a cross-session per-user frame lock; the gateway serializes all heavy inference even without Redis.

A frame produces an observation without invoking GPT-OSS. A transcript/reason event adds recent observations and communication metrics to the normal persistent agent turn. Observations expire after 60 seconds. Shopping uses visible objects/colors/text and remembered preferences; price, brand, material, quality and specifications require visible or tool evidence. Interview observations are enum-valued gaze/head/posture/facial-movement signals; audio code calculates pace, pauses, speaking ratio and RMS statistics. Neither deterministic path classifies psychological state. The model prompt and output guard add defense, but model quality still requires empirical evaluation.

## Deployment limits

The API assumes trusted localhost callers; client-supplied IDs are not authentication. Shared/public deployment requires authenticated principals, authorization, quotas, transport security and resource management. The health endpoint separates storage and cache connectivity from lazy model load state. A Redis outage does not make the service dead. A PostgreSQL outage prevents durable chat and is reported separately.

The 16 GB Mac incurs substantial cold-load and switching costs. Frame sampling is intentionally sparse and does not imply five-second end-to-end latency. Stub tests cannot prove visual recognition quality, model adherence, browser camera compatibility or real provider access. Use the real smoke script and manual demos to evaluate those boundaries.
