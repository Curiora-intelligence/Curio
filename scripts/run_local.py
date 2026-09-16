"""Run Curio on this Mac using existing cached weights, with no downloads."""
import argparse
import os

from app.core.models import cached_model
from app.core.config import load_settings


def main():
    load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for key, repo in {
        "CURIO_MLX_TEXT_MODEL": "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "CURIO_MLX_VISION_MODEL": "mlx-community/Qwen3-VL-8B-Instruct-8bit",
    }.items():
        if not os.getenv(key):
            os.environ[key] = cached_model(repo)
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
