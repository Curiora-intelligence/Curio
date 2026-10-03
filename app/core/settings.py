"""Small environment configuration; model settings remain in the gateway."""
from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost/curio"))
    redis_url: str | None = field(default_factory=lambda: os.getenv("REDIS_URL") or None)
    browser_backend: str = field(default_factory=lambda: os.getenv("CURIO_BROWSER_BACKEND", "disabled"))
    exa_api_key: str | None = field(default_factory=lambda: os.getenv("EXA_API_KEY") or None)
    frame_interval: float = field(default_factory=lambda: max(5.0, float(os.getenv("CURIO_FRAME_INTERVAL", "5"))))
