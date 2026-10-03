from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
from pathlib import Path
import tempfile
import time
from typing import Literal
from uuid import uuid4

from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from app.core.settings import Settings
from app.services.coaching import AudioMetrics, InterviewObservation, communication_metrics

Mode = Literal["discovery", "shopping", "interview", "general"]
MAX_FRAME_BYTES = 512 * 1024


class VisualObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objects: list[str] = Field(default_factory=list, max_length=8)
    colors: list[str] = Field(default_factory=list, max_length=8)
    visible_text: list[str] = Field(default_factory=list, max_length=8)
    visible_details: list[str] = Field(default_factory=list, max_length=8)


class LiveEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    type: Literal["configure", "frame", "transcript", "metrics", "reason", "ping"]
    mode: Mode | None = None
    conversation_id: str | None = Field(default=None, max_length=36)
    jpeg: str | None = Field(default=None, max_length=720000)
    captured_at: float | None = None
    text: str = Field(default="", max_length=16000)
    metrics: AudioMetrics | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


def parse_observation(raw: str, mode: str) -> dict:
    schema = InterviewObservation if mode == "interview" else VisualObservation
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        observation = schema.model_validate_json(raw).model_dump()
        if mode != "interview" and any(len(value) > 500 for values in observation.values() for value in values):
            raise ValueError("Observation too large")
        return observation
    except (ValueError, TypeError):
        return {"unavailable": "The frame did not produce a reliable structured observation."}


class LiveSession:
    def __init__(self, service, user_id: str, clock=time.monotonic):
        self.service, self.user_id, self.clock = service, user_id, clock
        self.id = str(uuid4())
        self.mode: Mode = "general"
        self.conversation_id = None
        self.last_frame = float("-inf")
        self.interval = Settings().frame_interval
        self.observation: dict = {}
        self.observed_at = 0.
        self.metrics: dict = {}
        self.transcript = ""
        self.location: dict = {}
        self.busy = False

    async def handle(self, event: LiveEvent) -> dict:
        if event.type == "ping":
            return {"type": "pong"}
        if event.type == "configure":
            if event.mode:
                self.mode = event.mode
                self.observation = {}
            if event.conversation_id:
                self.conversation_id = event.conversation_id
            if (event.latitude is None) != (event.longitude is None):
                raise ValueError("Provide both coordinates.")
            if event.latitude is not None:
                self.location = {"latitude": event.latitude, "longitude": event.longitude}
            await self.service.cache.set("live:" + self.id, {"mode": self.mode, "user_id": self.user_id}, 1800)
            return {"type": "configured", "mode": self.mode, "frame_interval_seconds": self.interval, "session_id": self.id}
        if event.type == "metrics":
            if not event.metrics:
                raise ValueError("metrics are required")
            self.audio_metrics = event.metrics
            self.metrics = communication_metrics(self.transcript, event.metrics)
            return {"type": "metrics", "metrics": self.metrics}
        if self.busy:
            return {"type": "busy", "message": "Inference is in progress. Retry your transcript after the response."}
        if event.type == "frame":
            return await self.frame(event)
        if event.type in {"transcript", "reason"}:
            text = event.text.strip()
            if not text:
                raise ValueError("A transcript or question is required.")
            self.transcript = text
            if event.metrics:
                self.audio_metrics = event.metrics
            if hasattr(self, "audio_metrics"):
                self.metrics = communication_metrics(text, self.audio_metrics)
            self.busy = True
            try:
                observation = self.observation if self.clock() - self.observed_at <= 60 else {"unavailable": "No recent visual observation."}
                context = json.dumps({"mode": self.mode, "visual_observation": observation, "audio_metrics": self.metrics,
                                      "location": self.location})
                answer, cid = await self.service.respond(text, conversation_id=self.conversation_id, user_id=self.user_id,
                                                          context=context, mode=self.mode)
                self.conversation_id = cid
                return {"type": "response", "answer": answer, "conversation_id": cid, "user_id": self.user_id}
            finally:
                self.busy = False
        raise ValueError("Unsupported event")

    async def frame(self, event: LiveEvent) -> dict:
        if self.clock() - self.last_frame < self.interval:
            return {"type": "frame_skipped", "reason": "throttled"}
        if event.captured_at is not None and abs(time.time() - event.captured_at) > 15:
            return {"type": "frame_skipped", "reason": "stale"}
        if not event.jpeg:
            raise ValueError("A base64 JPEG is required")
        try:
            data = base64.b64decode(event.jpeg, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 JPEG") from exc
        if len(data) > MAX_FRAME_BYTES:
            raise ValueError("Frame exceeds 512 KB")
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG" or max(image.size) > 1280:
                    raise ValueError("Use a JPEG of at most 1280 pixels per side")
                image.verify()
        except Exception as exc:
            raise ValueError("Invalid JPEG frame") from exc
        # Cross-session/worker throttle is optional; gateway still serializes all inference.
        token = await self.service.cache.acquire("frame:" + self.user_id, ttl=120)
        if not token:
            return {"type": "frame_skipped", "reason": "busy"}
        self.busy = True
        self.last_frame = self.clock()
        path = None
        try:
            schema = InterviewObservation if self.mode == "interview" else VisualObservation
            instruction = ("Describe only visible evidence. Return JSON matching this schema: " + json.dumps(schema.model_json_schema()) +
                           ". Never infer mental state, anxiety, stress, confidence or deception. Never infer a product's price, brand, material, specifications or quality; copy visible text literally. No markdown.")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as file:
                path = Path(file.name)
                file.write(data)
            raw = await run_in_threadpool(self.service.gateway.generate_vision, image_path=str(path),
                                          messages=[{"role": "user", "content": instruction}], max_tokens=384, temperature=.1)
            self.observation = parse_observation(raw, self.mode)
            self.observed_at = self.clock()
            await self.service.cache.set("live:" + self.id, {"mode": self.mode, "observation": self.observation}, 120)
            return {"type": "observation", "mode": self.mode, "observation": self.observation}
        finally:
            if path:
                path.unlink(missing_ok=True)
            self.last_frame = self.clock()  # cooldown starts after inference, avoiding back-to-back VLM loads.
            self.busy = False
            await self.service.cache.release("frame:" + self.user_id, token)
