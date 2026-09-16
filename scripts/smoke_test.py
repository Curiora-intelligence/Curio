"""Real, offline API inference test. Run from Curio: python -m scripts.smoke_test."""
import argparse
import importlib.metadata
import io
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


from app.core.models import cached_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "vision", "both"], default="both")
    parser.add_argument("--output", default="test-results/inference.json")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CURIO_MAX_TEXT_TOKENS", "256")
    os.environ.setdefault("CURIO_MAX_VISION_TOKENS", "96")
    repos = {
        "text": "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "vision": "mlx-community/Qwen3-VL-8B-Instruct-8bit",
    }
    modes = ["text", "vision", "text"] if args.mode == "both" else [args.mode]
    for mode in set(modes):
        os.environ.setdefault(f"CURIO_MLX_{mode.upper()}_MODEL", cached_model(repos[mode]))

    from PIL import Image, ImageDraw
    from fastapi.testclient import TestClient
    from main import app
    from app.routers.curio import curio_service

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {x: importlib.metadata.version(x) for x in
                     ["mlx", "mlx-lm", "mlx-vlm", "transformers", "torch", "fastapi"]},
        "tests": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    with TestClient(app) as client:
        report["health_status"] = client.get("/health").status_code
        for index, mode in enumerate(modes):
            start = time.monotonic()
            entry = {"mode": mode, "model": repos[mode]}
            print(f"Starting real {mode} API inference ({index + 1}/{len(modes)})", flush=True)
            try:
                if mode == "text":
                    response = client.post("/curio/analyze", data={"message": "What is 2 + 2? Answer with only the number."})
                else:
                    image = Image.new("RGB", (256, 256), "white")
                    ImageDraw.Draw(image).rectangle((64, 64, 192, 192), fill="red")
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    response = client.post("/curio/analyze", data={"message": "What color is the square? Answer with one word."},
                                           files={"image": ("red-square.png", buffer.getvalue(), "image/png")})
                entry["http_status"] = response.status_code
                entry["response"] = response.json()
                answer = entry["response"].get("answer", "").strip().lower().rstrip(".! ")
                entry["passed"] = response.status_code == 200 and answer == ("4" if mode == "text" else "red")
                entry["runtime"] = curio_service.gateway.runtime_info.kind.value
                entry["resident_mode"] = getattr(curio_service.gateway.runtime, "_mode", None)
            except Exception as exc:
                entry.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            entry["seconds"] = round(time.monotonic() - start, 2)
            report["tests"].append(entry)
            save()
            print(json.dumps(entry), flush=True)
    report["released_on_shutdown"] = curio_service.gateway._runtime is None
    save()
    raise SystemExit(0 if all(x["passed"] for x in report["tests"]) else 1)


if __name__ == "__main__":
    main()
