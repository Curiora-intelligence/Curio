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


# Price comparisons require observed/tool-provided prices, never a remembered ceiling alone.
BUDGET_CLAIM = re.compile(
    r"\b(?:within|under|fits?|meets?|below|in)\b.{0,45}\bbudget\b|"
    r"\bbudget[- ]friendly\b|\baffordable\b|\b(?:cheap|inexpensive)\b", re.I)
NEGATED_PRICE = re.compile(r"\b(?:can['’]?t|cannot|don['’]?t|not|unknown|whether|if|without|unconfirmed|unclear)\b", re.I)
VISIBLE_PRICE = re.compile(r"(?:₹|\$|€|£|\bINR\b|\bRs\.?|\bprice\b)\s*[:=]?\s*\d", re.I)


def shopping_price_evidence(context: str, tool_results: list[dict]) -> bool:
    import json
    try:
        observation = json.loads(context).get("visual_observation", {})
        if any(VISIBLE_PRICE.search(str(value)) for value in observation.get("visible_text", [])):
            return True
    except (ValueError, AttributeError, TypeError):
        pass

    def priced(value):
        if isinstance(value, dict):
            if isinstance(value.get("price"), (int, float)) and not isinstance(value["price"], bool):
                return True
            return any(priced(child) for child in value.values())
        return isinstance(value, list) and any(priced(child) for child in value)
    return any(priced(result) for result in tool_results)


def safe_shopping_text(text: str, price_known: bool) -> str:
    if price_known:
        return text
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    unsafe = [part for part in parts if BUDGET_CLAIM.search(part) and not NEGATED_PRICE.search(part)]
    if not unsafe:
        return text
    remaining = [part for part in parts if part not in unsafe]
    if any(NEGATED_PRICE.search(part) and re.search(r"\bprice\b", part, re.I) for part in remaining):
        return "\n".join(remaining)
    return "\n".join(remaining + ["Price is unknown, so I can’t confirm whether this item fits your budget."])
