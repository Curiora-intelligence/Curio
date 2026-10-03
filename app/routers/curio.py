from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Literal
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError

from app.services.curio import CurioService
from app.core.errors import RequestConflictError

curio_service = CurioService()
curio_router = APIRouter(prefix="/curio", tags=["Curio"])
ALLOWED_IMAGE_FORMATS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}
MAX_IMAGE_SIZE = 15 * 1024 * 1024
logger = logging.getLogger(__name__)


@curio_router.post("/analyze")
async def analyze_curio_image(image: UploadFile | None = File(None), message: str = Form("", max_length=16000),
                              conversation_id: str | None = Form(None, max_length=36),
                              user_id: str = Form("local-demo", min_length=1, max_length=128),
                              request_id: str | None = Form(None, max_length=128),
                              mode: Literal["general", "discovery", "shopping", "interview"] = Form("general"),
                              latitude: float | None = Form(None, ge=-90, le=90),
                              longitude: float | None = Form(None, ge=-180, le=180)):
    if image is None and not message.strip():
        raise HTTPException(400, "Please provide a message or an image.")
    if (latitude is None) != (longitude is None):
        raise HTTPException(422, "Provide both latitude and longitude.")
    context = json.dumps({"location": {"latitude": latitude, "longitude": longitude}}) if latitude is not None else ""
    temporary_path = None
    try:
        if image:
            if not image.content_type:
                raise HTTPException(400, "Image content type is missing.")
            if image.content_type not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(415, "Please upload JPEG, PNG, WEBP, or GIF.")
            data = await image.read(MAX_IMAGE_SIZE + 1)
            if not data:
                raise HTTPException(400, "The uploaded image is empty.")
            if len(data) > MAX_IMAGE_SIZE:
                raise HTTPException(413, "Image is too large. Maximum size is 15 MB.")
            with tempfile.NamedTemporaryFile(suffix=ALLOWED_IMAGE_FORMATS[image.content_type], delete=False) as file:
                temporary_path = Path(file.name)
                file.write(data)
        answer, cid = await curio_service.respond(message, str(temporary_path) if temporary_path else None,
                                                  conversation_id, user_id, request_id=request_id, mode=mode, context=context)
        return {"success": True, "mode": "vision" if image else "text", "answer": answer,
                "conversation_id": cid, "user_id": user_id}
    except HTTPException:
        raise
    except RequestConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("Curio persistence unavailable: %s", type(exc).__name__)
        raise HTTPException(503, "Persistent storage unavailable. Please retry.") from exc
    except Exception as exc:
        logger.warning("Curio inference failed: %s", type(exc).__name__)
        raise HTTPException(500, "Curio could not process the request.") from exc
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        if image:
            await image.close()
