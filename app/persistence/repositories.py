from __future__ import annotations

from uuid import uuid5, NAMESPACE_URL
import json

from sqlalchemy import cast, func, or_, select, String, Text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.models import AgentRun, Conversation, Memory, Message, ToolCall, User, now


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def ensure_user(self, user_id: str) -> User:
        if self.session.bind.dialect.name == "postgresql":
            await self.session.execute(pg_insert(User).values(id=user_id).on_conflict_do_nothing())
        elif await self.session.get(User, user_id) is None:
            self.session.add(User(id=user_id))
            await self.session.flush()
        return await self.session.get(User, user_id)

    async def get_or_create(self, user_id: str, conversation_id: str | None, request_id: str | None = None) -> Conversation:
        await self.ensure_user(user_id)
        if conversation_id:
            conversation = await self.session.get(Conversation, conversation_id)
            if not conversation or conversation.user_id != user_id:
                raise LookupError("Conversation not found for this user.")
            return conversation
        new_id = str(uuid5(NAMESPACE_URL, json.dumps(["curio", user_id, request_id]))) if request_id else None
        if new_id:
            existing = await self.session.get(Conversation, new_id)
            if existing:
                return existing
        conversation = Conversation(user_id=user_id, **({"id": new_id} if new_id else {}))
        self.session.add(conversation)
        await self.session.flush()
        return conversation


class MessageRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, conversation_id: str, role: str, content: str) -> Message:
        message = Message(conversation_id=conversation_id, role=role, content=content)
        self.session.add(message)
        await self.session.flush()
        return message

    async def recent(self, conversation_id: str, limit: int = 16) -> list[Message]:
        rows = await self.session.scalars(select(Message).where(Message.conversation_id == conversation_id)
                                         .order_by(Message.created_at.desc(), Message.id.desc()).limit(limit))
        return list(reversed(rows.all()))


class MemoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def version(self, user_id: str) -> int:
        return (await self.session.scalar(select(User.memory_version).where(User.id == user_id))) or 0

    async def list(self, user_id: str) -> list[Memory]:
        return list((await self.session.scalars(select(Memory).where(Memory.user_id == user_id, Memory.active)
                                               .order_by(Memory.updated_at.desc()))).all())

    async def candidates(self, user_id: str, terms: set[str]) -> list[Memory]:
        if not terms:
            return []
        document = Memory.kind + " " + Memory.key + " " + Memory.value
        # Keyword retrieval is portable; PostgreSQL also matches lexical forms with FTS.
        predicates = [document.ilike("%" + term.replace("%", "").replace("_", " ") + "%") for term in sorted(terms)]
        if self.session.bind.dialect.name == "postgresql":
            predicates.append(func.to_tsvector("english", document).op("@@")(
                func.plainto_tsquery("english", " ".join(sorted(terms)))))
        rows = await self.session.scalars(select(Memory).where(Memory.user_id == user_id, Memory.active, or_(*predicates))
                                         .order_by(Memory.updated_at.desc()).limit(100))
        return list(rows.all())

    async def semantic_candidates(self, user_id: str, embedding: list[float]) -> list[tuple[Memory, float]]:
        if self.session.bind.dialect.name != "postgresql":
            return []
        # Optional pgvector casts over JSON vectors keep the base migration extension-independent.
        from pgvector.sqlalchemy import Vector
        try:
            async with self.session.begin_nested():
                distance = cast(cast(Memory.embedding, Text), Vector(len(embedding))).cosine_distance(embedding)
                rows = await self.session.execute(select(Memory, distance).where(
                    Memory.user_id == user_id, Memory.active,
                    func.json_array_length(Memory.embedding) == len(embedding)).order_by(distance).limit(20))
                return [(row[0], max(0., 1 - row[1])) for row in rows if row[1] is not None]
        except SQLAlchemyError:
            # Missing vector extension or incompatible stored vectors cannot break keyword retrieval.
            return []

    async def remember(self, user_id: str, *, kind: str, key: str, value: str, confidence: float,
                       importance: float, source_message_id: str | None, embedding=None) -> Memory:
        # Serializes concurrent updates to a user's memory, preserving one active value per key.
        await self.session.execute(select(User).where(User.id == user_id).with_for_update())
        previous = await self.session.scalar(select(Memory).where(Memory.user_id == user_id, Memory.kind == kind,
                                                                 Memory.key == key, Memory.active))
        if previous and previous.value.casefold() == value.casefold():
            return previous
        if previous:
            previous.active = False
            previous.updated_at = now()
            await self.session.flush()
        memory = Memory(user_id=user_id, kind=kind, key=key, value=value, confidence=confidence,
                        importance=importance, source_message_id=source_message_id, embedding=embedding)
        self.session.add(memory)
        await self.session.execute(update(User).where(User.id == user_id).values(memory_version=User.memory_version + 1))
        await self.session.flush()
        return memory

    async def forget(self, user_id: str, memory_id: str) -> bool:
        result = await self.session.execute(update(Memory).where(Memory.id == memory_id, Memory.user_id == user_id, Memory.active)
                                            .values(active=False, updated_at=now()))
        if result.rowcount:
            await self.session.execute(update(User).where(User.id == user_id).values(memory_version=User.memory_version + 1))
        return bool(result.rowcount)


class AgentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, conversation_id: str, request_id: str | None = None) -> AgentRun:
        run = AgentRun(conversation_id=conversation_id, request_id=request_id)
        self.session.add(run)
        await self.session.flush()
        return run

    async def by_request(self, conversation_id: str, request_id: str) -> AgentRun | None:
        return await self.session.scalar(select(AgentRun).where(AgentRun.conversation_id == conversation_id,
                                                               AgentRun.request_id == request_id))

    async def finish(self, run_id: str, status: str, iterations: int, answer: str | None) -> None:
        await self.session.execute(update(AgentRun).where(AgentRun.id == run_id).values(
            status=status, iterations=iterations, answer=answer, updated_at=now()))

    async def begin_tool(self, run_id: str, name: str, arguments: dict) -> str:
        call = ToolCall(agent_run_id=run_id, name=name, arguments=arguments)
        self.session.add(call)
        await self.session.flush()
        return call.id

    async def finish_tool(self, call_id: str, result: dict) -> None:
        await self.session.execute(update(ToolCall).where(ToolCall.id == call_id).values(
            result=result, status="error" if result.get("error") else "completed"))
