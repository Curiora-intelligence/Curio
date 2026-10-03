from pydantic import BaseModel, ConfigDict, Field
from app.persistence.repositories import MemoryRepository
from app.services.memory import extract_explicit
from app.tools.registry import Permission, Tool


class SearchMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)


class RememberMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(min_length=1, max_length=4000)


def register_memory_tools(registry, database, memory_service):
    async def search(args, context):
        async with database.sessions() as session:
            return {"memories": await memory_service.retrieve(MemoryRepository(session), context.user_id, args.query)}

    async def remember(args, context):
        if not extract_explicit(args.statement):
            return {"error": "not_explicit", "message": "No explicit durable preference, goal, constraint or remember request."}
        async with database.sessions.begin() as session:
            return {"memories": await memory_service.learn(MemoryRepository(session), context.user_id,
                                                           args.statement, context.source_message_id)}
    registry.register(Tool("memory.search", "Retrieve the user's relevant durable memories.", SearchMemory,
                           Permission.READ_ONLY, search))
    registry.register(Tool("memory.remember", "Store an explicit durable user statement. Pass the exact current user statement.",
                           RememberMemory, Permission.USER_DATA_WRITE, remember))
