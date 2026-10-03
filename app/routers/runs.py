import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.routers.curio import curio_service, ALLOWED_IMAGE_FORMATS, MAX_IMAGE_SIZE
from app.services.events import encode_event, event_number, read_events
from app.services.runs import RunManager

runs_router = APIRouter(prefix="/curio/runs", tags=["Runs"])
run_manager = RunManager(curio_service)
POLL_SECONDS = .5
HEARTBEAT_SECONDS = 15
# Reserved terminal recovery IDs are above every bounded agent-loop event ID.
RECOVERY_FINAL = 1000000
RECOVERY_DONE = 1000001


@runs_router.post("", status_code=202)
async def start_run(image: UploadFile | None = File(None), message: str = Form("", max_length=16000),
                    conversation_id: str | None = Form(None, max_length=36),
                    user_id: str = Form("local-demo", min_length=1, max_length=128),
                    request_id: str | None = Form(None, min_length=1, max_length=128),
                    mode: Literal["general", "discovery", "shopping", "interview"] = Form("general"),
                    latitude: float | None = Form(None, ge=-90, le=90),
                    longitude: float | None = Form(None, ge=-180, le=180)):
    path = None
    transferred = False
    try:
        if image is None and not message.strip():
            raise HTTPException(400, "Please provide a message or an image.")
        if (latitude is None) != (longitude is None):
            raise HTTPException(422, "Provide both latitude and longitude.")
        if image:
            if image.content_type not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(415, "Please upload JPEG, PNG, WEBP, or GIF.")
            data = await image.read(MAX_IMAGE_SIZE + 1)
            if not data:
                raise HTTPException(400, "The uploaded image is empty.")
            if len(data) > MAX_IMAGE_SIZE:
                raise HTTPException(413, "Image is too large. Maximum size is 15 MB.")
            with tempfile.NamedTemporaryFile(suffix=ALLOWED_IMAGE_FORMATS[image.content_type], delete=False) as file:
                path = file.name
                file.write(data)
        context = json.dumps({"location": {"latitude": latitude, "longitude": longitude}}) if latitude is not None else ""
        run_id = await run_manager.start(message=message, image_path=path, user_id=user_id,
                                         conversation_id=conversation_id, request_id=request_id, mode=mode, context=context)
        transferred = True
        return {"request_id": run_id, "status": "accepted"}
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    finally:
        if image:
            await image.close()
        if path and not transferred:
            Path(path).unlink(missing_ok=True)


async def owned_status(request_id, user_id):
    try:
        return await run_manager.status(request_id, user_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@runs_router.get("/{request_id}")
async def run_status(request_id: str, user_id: str = Query("local-demo", min_length=1, max_length=128)):
    return await owned_status(request_id, user_id)


@runs_router.get("/{request_id}/events")
async def run_events(request_id: str, request: Request,
                     user_id: str = Query("local-demo", min_length=1, max_length=128),
                     last_event_id: str = Header("0", alias="Last-Event-ID")):
    await owned_status(request_id, user_id)
    try:
        after = event_number(last_event_id)
    except ValueError as exc:
        raise HTTPException(400, "Invalid Last-Event-ID") from exc

    async def stream():
        nonlocal after
        heartbeat = time.monotonic()
        yield ": connected\n\n"
        while not await request.is_disconnected():
            rows = await read_events(curio_service.cache.client, request_id, after)
            for event_id, fields in rows:
                number = event_number(event_id)
                after = number
                yield encode_event(number, fields["event"], json.loads(fields["data"]))
                if fields["event"] == "done":
                    return
            status = await owned_status(request_id, user_id)
            if status["status"] in {"completed", "failed"}:
                # Durable recovery also works after stream TTL, outages or server restarts.
                if after < RECOVERY_FINAL:
                    if status["status"] == "completed":
                        yield encode_event(RECOVERY_FINAL, "final", {k: status[k] for k in ("answer", "conversation_id", "user_id")})
                    else:
                        yield encode_event(RECOVERY_FINAL, "error", {"message": "Curio could not finish this request. Please try again."})
                if after < RECOVERY_DONE:
                    yield encode_event(RECOVERY_DONE, "done", {})
                return
            if time.monotonic() - heartbeat >= HEARTBEAT_SECONDS:
                yield ": keep-alive\n\n"
                heartbeat = time.monotonic()
            await asyncio.sleep(POLL_SECONDS)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
