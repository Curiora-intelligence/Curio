from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.persistence.repositories import ConversationRepository, MemoryRepository, MessageRepository
from app.routers.curio import curio_service
from app.services.memory import extract_explicit, serialize

memory_router = APIRouter(prefix="/memory", tags=["Memory"])


class RememberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=128)
    statement: str = Field(min_length=1, max_length=4000)


class ForgetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=128)
    memory_id: str = Field(min_length=1, max_length=36)


@memory_router.get("/{user_id}")
async def list_memories(user_id: str):
    async with curio_service.database.sessions() as session:
        return {"memories": [serialize(m) for m in await MemoryRepository(session).list(user_id)]}


@memory_router.post("/remember")
async def remember(request: RememberRequest):
    if not extract_explicit(request.statement):
        raise HTTPException(422, "Provide an explicit preference, goal, constraint, skill gap, or 'remember ...' statement.")
    async with curio_service.database.user_lock(request.user_id):
        async with curio_service.database.sessions.begin() as session:
            conversation = await ConversationRepository(session).get_or_create(request.user_id, None)
            source = await MessageRepository(session).add(conversation.id, "user", request.statement)
            memories = await curio_service.memory.learn(MemoryRepository(session), request.user_id, request.statement, source.id)
    return {"memories": memories}


@memory_router.post("/forget")
async def forget(request: ForgetRequest):
    async with curio_service.database.user_lock(request.user_id):
        async with curio_service.database.sessions.begin() as session:
            forgotten = await MemoryRepository(session).forget(request.user_id, request.memory_id)
    if not forgotten:
        raise HTTPException(404, "Memory not found for this user.")
    return {"forgotten": True}
