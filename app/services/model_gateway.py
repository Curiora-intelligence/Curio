from __future__ import annotations

import os
from threading import Lock

from app.core.runtime import (
    RuntimeInfo,
    RuntimeKind,
    detect_runtime,
)
from app.runtimes.base import RuntimeAdapter


class ModelMode(str):
    TEXT = "text"
    VISION = "vision"


class ModelGateway:
    """
    Curio model gateway.

    Routing:

        TEXT
            -> GPT-OSS

        IMAGE / MULTIMODAL
            -> Qwen3-VL

    Only one heavy model is kept resident at a time.
    """

    TEXT_MODEL_ID = (
        "mlx-community/gpt-oss-20b-MXFP4-Q8"
    )

    VISION_MODEL_ID = (
        "mlx-community/Qwen3-VL-8B-Instruct-8bit"
    )

    TORCH_TEXT_MODEL_ID = (
        "openai/gpt-oss-20b"
    )

    TORCH_VISION_MODEL_ID = (
        "Qwen/Qwen3-VL-8B-Instruct"
    )

    def __init__(self) -> None:

        self._lock = Lock()
        self.TEXT_MODEL_ID = os.getenv("CURIO_MLX_TEXT_MODEL", self.TEXT_MODEL_ID)
        self.VISION_MODEL_ID = os.getenv("CURIO_MLX_VISION_MODEL", self.VISION_MODEL_ID)

        self._runtime_info: RuntimeInfo | None = None
        self._runtime: RuntimeAdapter | None = None

    @property
    def runtime_info(self) -> RuntimeInfo:

        if self._runtime_info is None:
            self._runtime_info = detect_runtime()

        return self._runtime_info

    @property
    def runtime(self) -> RuntimeAdapter:

        if self._runtime is None:

            info = self.runtime_info

            if info.kind == RuntimeKind.MLX:
                from app.runtimes.mlx_runtime import MLXRuntime
                self._runtime = MLXRuntime()

            elif info.kind == RuntimeKind.CUDA:
                from app.runtimes.torch_runtime import TorchRuntime
                self._runtime = TorchRuntime(
                    device="cuda"
                )

            elif info.kind == RuntimeKind.CPU:
                from app.runtimes.torch_runtime import TorchRuntime
                self._runtime = TorchRuntime(
                    device="cpu"
                )

            else:
                raise RuntimeError(
                    f"Unsupported Curio runtime: "
                    f"{info.kind}"
                )

        return self._runtime

    # =========================================================
    # TEXT
    # =========================================================

    def generate_text(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:

        with self._lock:

            if (
                self.runtime_info.kind
                == RuntimeKind.MLX
            ):
                model_id = self.TEXT_MODEL_ID
            else:
                model_id = self.TORCH_TEXT_MODEL_ID

            return self.runtime.generate_text(
                model_id=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    # =========================================================
    # VISION
    # =========================================================

    def generate_vision(
        self,
        *,
        image_path: str,
        messages: list[dict[str, str]],
        max_tokens: int = 384,
        temperature: float = 0.2,
    ) -> str:

        with self._lock:

            if (
                self.runtime_info.kind
                == RuntimeKind.MLX
            ):
                model_id = self.VISION_MODEL_ID
            else:
                model_id = self.TORCH_VISION_MODEL_ID

            return self.runtime.generate_vision(
                model_id=model_id,
                image_path=image_path,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    # =========================================================
    # LIFECYCLE
    # =========================================================

    def release(self) -> None:

        with self._lock:

            if self._runtime is not None:
                self._runtime.release()

            self._runtime = None
            self._runtime_info = None
