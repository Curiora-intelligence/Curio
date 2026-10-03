"""Observable communication metrics only; no psychological classification."""
import re
from statistics import pstdev, mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FORBIDDEN_INFERENCES = re.compile(r"\b(?:anxious|anxiety|depress\w*|stress\w*|confiden\w*|decept\w*|dishonest|lying|nervous|mental\s+(?:state|health)|emotion\w*)\b", re.I)


class AudioMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    duration_seconds: float = Field(gt=0, le=3600)
    speaking_seconds: float = Field(ge=0, le=3600)
    pause_count: int = Field(default=0, ge=0, le=10000)
    rms: list[float] = Field(default_factory=list, max_length=600)

    @model_validator(mode="after")
    def valid_metrics(self):
        if self.speaking_seconds > self.duration_seconds or any(not 0 <= value <= 1 for value in self.rms):
            raise ValueError("Audio metrics are outside their valid range.")
        return self


class InterviewObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gaze_direction: Literal["toward_camera", "away_from_camera", "not_visible"] = "not_visible"
    head_orientation: Literal["forward", "turned", "tilted", "not_visible"] = "not_visible"
    posture: Literal["upright", "leaning", "not_visible"] = "not_visible"
    facial_expressiveness: Literal["movement_visible", "little_movement", "not_visible"] = "not_visible"


def communication_metrics(transcript: str, metrics: AudioMetrics) -> dict:
    words = len(re.findall(r"\b[\w']+\b", transcript))
    return {"words": words, "words_per_minute": round(words / metrics.duration_seconds * 60, 1),
            "speaking_ratio": round(metrics.speaking_seconds / metrics.duration_seconds, 3),
            "pause_count": metrics.pause_count,
            "rms_mean": round(mean(metrics.rms), 4) if metrics.rms else None,
            "rms_standard_deviation": round(pstdev(metrics.rms), 4) if metrics.rms else None,
            "note": "Client-measured communication metrics; camera observations are sparse samples."}


def safe_interview_text(text: str) -> str:
    if FORBIDDEN_INFERENCES.search(text):
        return "Let's focus on your answer structure, speaking pace, pauses, and visible gaze direction. Try stating the main idea, giving one example, and summarizing."
    return text


MODE_INSTRUCTIONS = {
    "interview": "Coach only observable communication: gaze direction, head orientation, visible posture, facial movement, pace, pauses, speaking ratio, volume consistency, answer structure and clarity. Never infer anxiety, depression, stress, confidence, deception, emotions or mental state from face or voice. Do not claim temporal changes without comparative samples. Retrieve relevant career goals and skill gaps for questions.",
    "shopping": "Compare visible evidence with the user's remembered preferences and budget. Never invent price, brand, material, specifications or quality. Cite visible text or tool results supporting such claims; otherwise state that these are unknown.",
    "discovery": "Use visible observations and the user's request as context. Distinguish visible evidence from inference.",
    "general": "Ground visual claims in supplied observations and state uncertainty.",
}
