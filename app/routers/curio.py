from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.requests import Request

from app.services.curio import CurioService

curio_service = CurioService()

curio_router = APIRouter(prefix="/curio",tags=["Curio routings"])


ALLOWED_IMAGE_FORMATS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

MAX_IMAGE_SIZE = 15 * 1024 * 1024


@curio_router.post("/analyze")
async def analyze_curio_image(request: Request,image: UploadFile | None = File(None),message: str = Form(""),conversation_id: str | None = Form(None)):
    """
    Unified Curio endpoint.

    Cases:

        1. Text only
           message -> GPT-OSS

        2. Image + text
           image + message -> Qwen3-VL

        3. Image only
           image -> Qwen3-VL
    """

    del request  # Reserved for auth/session middleware.

    message = message.strip()

    # ---------------------------------------------------------
    # Validate that something was provided
    # ---------------------------------------------------------

    if image is None and not message:
        raise HTTPException(
            status_code=400,
            detail="Please provide a message or an image.",
        )

    # ---------------------------------------------------------
    # TEXT-ONLY REQUEST
    # ---------------------------------------------------------

    if image is None:

        try:
            answer, new_conversation_id = await run_in_threadpool(
                curio_service.respond,
                message,
                None,
                conversation_id,
            )

            return {
                "success": True,
                "mode": "text",
                "answer": answer,
                "conversation_id": new_conversation_id,
            }

        except Exception as exc:
            print(
                f"Curio text request failed: {exc}"
            )

            raise HTTPException(
                status_code=500,
                detail="Curio could not process the request.",
            ) from exc

    # ---------------------------------------------------------
    # IMAGE REQUEST
    # ---------------------------------------------------------

    if not image.content_type:
        raise HTTPException(
            status_code=400,
            detail="Image content type is missing.",
        )

    if image.content_type not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image type. "
                "Please upload JPEG, PNG, WEBP, or GIF."
            ),
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum size is 15 MB."
            ),
        )

    suffix = ALLOWED_IMAGE_FORMATS[
        image.content_type
    ]

    temporary_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            delete=False,
        ) as temporary_file:

            temporary_file.write(
                image_bytes
            )

            temporary_path = Path(
                temporary_file.name
            )

        answer, new_conversation_id = await run_in_threadpool(
            curio_service.respond,
            message,
            str(temporary_path),
            conversation_id,
        )

        return {
            "success": True,
            "mode": "vision",
            "answer": answer,
            "conversation_id": new_conversation_id,
        }

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=400,
            detail="The uploaded image could not be processed.",
        ) from exc

    except Exception as exc:

        print(
            f"Curio image request failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Curio could not analyze the image."
            ),
        ) from exc

    finally:

        if temporary_path is not None:

            try:
                os.remove(temporary_path)
            except OSError:
                pass