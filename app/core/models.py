"""Resolve existing checkpoints without downloading model files."""
import os
from pathlib import Path


def cached_model(repo: str) -> str:
    root = Path(os.getenv("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    cached = root / ("models--" + repo.replace("/", "--"))
    revision = (cached / "refs/main").read_text().strip()
    snapshot = cached / "snapshots" / revision
    if not (snapshot / "config.json").is_file() or not list(snapshot.glob("*.safetensors")):
        raise FileNotFoundError(f"No usable local checkpoint: {snapshot}")
    # File presence is not model readiness. The smoke test verifies actual loading.
    return str(snapshot)
