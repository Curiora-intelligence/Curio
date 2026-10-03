import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.routers.curio import curio_service
from app.services.live import LiveEvent, LiveSession

live_router = APIRouter(prefix="/live", tags=["Live"])


@live_router.websocket("/ws/{user_id}")
async def live(websocket: WebSocket, user_id: str):
    if not 1 <= len(user_id) <= 128:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    session = LiveSession(curio_service, user_id)
    task = None
    send_lock = asyncio.Lock()

    async def send(payload):
        async with send_lock:
            await websocket.send_json(payload)

    async def process(event):
        try:
            await send(await session.handle(event))
        except (ValueError, ValidationError):
            await send({"type": "error", "message": "Invalid event fields or frame. Check the live protocol."})
        except Exception:
            await send({"type": "error", "message": "Curio could not process this event. Please retry."})

    try:
        await send({"type": "ready", "session_id": session.id, "frame_interval_seconds": session.interval})
        while True:
            raw = await websocket.receive_text()
            if len(raw) > 740000:
                await send({"type": "error", "message": "Event too large"})
                continue
            try:
                event = LiveEvent.model_validate_json(raw)
            except ValueError:
                await send({"type": "error", "message": "Invalid live event"})
                continue
            if task and not task.done():
                # Read and discard frames during inference; never queue video for later inference.
                await send({"type": "frame_skipped" if event.type == "frame" else "busy", "reason": "inference_in_progress"})
                continue
            if task:
                await task
            task = asyncio.create_task(process(event))
    except WebSocketDisconnect:
        pass
    finally:
        if task:
            # Do not cancel a thread running MLX or unlink an image still being used by it.
            try:
                await task
            except (WebSocketDisconnect, RuntimeError):
                pass
