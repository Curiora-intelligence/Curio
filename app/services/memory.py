from __future__ import annotations

from datetime import timezone
import hashlib
import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.persistence.models import Memory, now
from app.persistence.repositories import MemoryRepository
from app.services.ephemeral import EphemeralStore


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["preference", "constraint", "goal", "skill_gap", "fact"]
    key: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9_]+$")
    value: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(default=1.0, ge=0.9, le=1)
    importance: float = Field(default=0.7, ge=0, le=1)


def extract_explicit(text: str) -> list[MemoryCandidate]:
    """High precision first-person patterns. No inferred memories or quoted instructions."""
    text = text.strip()
    if len(text) > 4000 or any(marker in text for marker in ('<|', '```', '"', '“', '”')):
        return []
    explicit = bool(re.match(r"remember\b", text, re.I))
    statement = re.sub(r"^remember\s+(?:that\s+)?", "", text, flags=re.I).strip().rstrip(".!?")
    candidates = []
    def add(kind, key, value):
        candidates.append(MemoryCandidate(kind=kind, key=key, value=value.strip()))
    preference = re.match(r"I (?:prefer|like|love)\s+(.+?)(?=\s+and\s+(?:I\s+)?(?:usually|my budget|stay|keep)|[.!?]|$)", statement, re.I)
    if preference:
        value = preference.group(1)
        key = "food" if re.search(r"biryani|food|chicken|spicy|vegetarian|vegan|pizza", value, re.I) else "shopping_style" if re.search(r"products?|style|black|minimal|clothes|shopping", value, re.I) else "preference"
        add("preference", key, value)
    budget = re.search(r"(?:I\s+(?:usually\s+)?(?:stay|keep)\s+under|(?:and\s+)?usually stay under|my\s+(?:shopping\s+|food\s+)?budget\s+(?:is\s+)?|budget\s+)\s*(?:₹|Rs\.?\s*|INR\s*)?(\d+(?:\.\d+)?)", statement, re.I)
    if budget and (re.match(r"(?:I |my )", statement, re.I) or explicit):
        key = "food_budget" if re.search(r"food|biryani|chicken|meal", statement, re.I) else "shopping_budget" if re.search(r"shopping|products?|black|style", statement, re.I) else "budget"
        add("constraint", key, budget.group(1))
    gap = re.match(r"I (?:struggle with|need to improve (?:at|on))\s+(.+)", statement, re.I)
    if gap:
        add("skill_gap", "interview", gap.group(1))
    goal = re.match(r"(?:my (?:career )?goal is|I (?:want|aim) to (?:get|find|become))\s+(.+)", statement, re.I)
    if goal:
        add("goal", "career", goal.group(1))
    if explicit and not candidates and statement:
        # The explicit remember API can retain an arbitrary user-provided fact.
        add("fact", "note_" + hashlib.sha256(statement.casefold().encode()).hexdigest()[:16], statement)
    return candidates


DOMAINS = {
    "food": {"hungry", "eat", "food", "meal", "biryani", "dinner", "lunch", "breakfast", "restaurant"},
    "shopping": {"shopping", "buy", "product", "products", "style", "purchase", "like"},
    "interview": {"interview", "internship", "backend", "career", "indexing", "database", "job"},
}
STOP = {"i", "im", "m", "my", "a", "an", "the", "is", "it", "this", "that", "to", "and", "for", "me", "with", "of", "you", "would", "do", "can", "something", "find", "nearby", "today"}


def query_terms(query: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", query.lower())) - STOP
    for domain, triggers in DOMAINS.items():
        if words & triggers:
            words.add(domain)
            if domain == "interview":
                words.add("career")
    return words


def serialize(memory: Memory) -> dict:
    return {"id": memory.id, "kind": memory.kind, "key": memory.key, "value": memory.value,
            "confidence": memory.confidence, "importance": memory.importance,
            "source_message_id": memory.source_message_id, "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(), "active": memory.active}


class MemoryService:
    def __init__(self, cache: EphemeralStore | None = None, embeddings: EmbeddingProvider | None = None):
        self.cache = cache or EphemeralStore()
        self.embeddings = embeddings

    async def learn(self, repository: MemoryRepository, user_id: str, statement: str,
                    source_message_id: str | None) -> list[dict]:
        results = []
        for candidate in extract_explicit(statement):
            embedding = None
            if self.embeddings:
                try:
                    embedding = await self.embeddings.embed(candidate.value)
                except Exception:
                    pass  # Optional local embedding dependency/model may be absent.
            memory = await repository.remember(user_id, **candidate.model_dump(), source_message_id=source_message_id,
                                               embedding=embedding)
            results.append(serialize(memory))
        return results

    async def retrieve(self, repository: MemoryRepository, user_id: str, query: str, limit: int = 6) -> list[dict]:
        terms = query_terms(query)
        version = await repository.version(user_id)
        cache_key = f"memory:{user_id}:{version}:" + hashlib.sha256(query.encode()).hexdigest()
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached[:limit]
        candidates = await repository.candidates(user_id, terms)
        semantic = {}
        if self.embeddings:
            try:
                vector = await self.embeddings.embed(query)
                for memory, similarity in await repository.semantic_candidates(user_id, vector):
                    if similarity >= .65:
                        semantic[memory.id] = similarity
                        if all(candidate.id != memory.id for candidate in candidates):
                            candidates.append(memory)
            except Exception:
                pass
        scored = []
        for memory in candidates:
            words = query_terms(memory.key.replace("_", " ") + " " + memory.value)
            relevance = max(len(terms & words) / max(1, len(terms)), semantic.get(memory.id, 0))
            if relevance == 0:
                continue
            age = max(0, (now() - memory.updated_at.replace(tzinfo=timezone.utc)).total_seconds() / 86400)
            type_weight = {"constraint": 1, "goal": .9, "skill_gap": .9, "preference": .8, "fact": .5}[memory.kind]
            score = .55 * relevance + .2 * memory.importance + .15 / (1 + age / 30) + .1 * type_weight
            scored.append((score, memory.id, serialize(memory)))
        results = [entry[2] for entry in sorted(scored, key=lambda entry: (-entry[0], entry[1]))][:6]
        await self.cache.set(cache_key, results, ttl=45)
        return results[:limit]
