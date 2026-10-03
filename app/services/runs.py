"""Background scheduling around the existing service, with durable AgentRun state."""
import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from app.persistence.models import AgentRun, Conversation
from app.persistence.repositories import AgentRepository, ConversationRepository
from app.services.events import RunEvents

logger = logging.getLogger(__name__)


class RunManager:
    def __init__(self, service):
        self.service = service
        self.tasks = set()

    async def start(self, *, user_id, conversation_id=None, request_id=None, image_path=None, **kwargs):
        # A separate short acceptance lock never waits for a model turn's user lock.
        async with self.service.database.user_lock("run-accept:" + user_id):
            async with self.service.database.sessions.begin() as session:
                conversation = await ConversationRepository(session).get_or_create(user_id, conversation_id, request_id)
                repository = AgentRepository(session)
                previous = await repository.by_request(conversation.id, request_id) if request_id else None
                if previous:
                    if image_path:
                        Path(image_path).unlink(missing_ok=True)
                    return previous.id
                run = await repository.create(conversation.id, request_id or str(uuid4()))
                run.status = "accepted"
        task = asyncio.create_task(self._execute(run.id, user_id, conversation.id, image_path, kwargs))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return run.id

    async def _execute(self, run_id, user_id, conversation_id, image_path, kwargs):
        events = RunEvents(self.service.cache.client, run_id)
        try:
            await events.started()
            await events.stage("understanding")
            answer, cid = await self.service.respond(user_id=user_id, conversation_id=conversation_id,
                                                     image_path=image_path, run_id=run_id, events=events, **kwargs)
            await events.final(answer, cid, user_id)
        except Exception as exc:
            logger.warning("Background Curio run failed: %s", type(exc).__name__)
            try:
                async with self.service.database.sessions.begin() as session:
                    await AgentRepository(session).finish(run_id, "failed", 0, None)
            except Exception:
                logger.error("Could not persist failed run status")
            await events.error()
        finally:
            if image_path:
                Path(image_path).unlink(missing_ok=True)
            await events.done()

    async def status(self, run_id, user_id):
        async with self.service.database.sessions() as session:
            run = await session.scalar(select(AgentRun).join(Conversation).where(
                AgentRun.id == run_id, Conversation.user_id == user_id))
            if not run:
                raise LookupError("Request not found for this user.")
            return {"request_id": run.id, "status": "completed" if run.status == "iteration_limit" else run.status,
                    "answer": run.answer, "conversation_id": run.conversation_id, "user_id": user_id}

    async def close(self):
        # Finish outstanding model threads before removing images or releasing the gateway.
        if self.tasks:
            await asyncio.gather(*self.tasks)
