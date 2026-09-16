from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from enum import Enum


class RuntimeKind(str, Enum):
    MLX = "mlx"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass(frozen=True)
class RuntimeInfo:
    kind: RuntimeKind
    platform: str
    architecture: str
    device_name: str
    reason: str


def _mlx_available() -> bool:
    """
    Check whether MLX can be imported without importing it globally.
    """
    return importlib.util.find_spec("mlx") is not None


def _cuda_available() -> tuple[bool, str]:
    """
    Detect CUDA through PyTorch without making PyTorch mandatory
    for Apple/MLX installations.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)

    except Exception:
        pass

    return False, ""


def detect_runtime() -> RuntimeInfo:
    """
    Select Curio's execution runtime.

    Priority:
        1. Apple Silicon + MLX
        2. NVIDIA CUDA
        3. PyTorch CPU

    MPS is intentionally not selected by Curio.
    """

    system = platform.system()
    architecture = platform.machine().lower()

    # ---------------------------------------------------------
    # Apple Silicon → MLX
    # ---------------------------------------------------------

    is_apple_silicon = (
        system == "Darwin"
        and architecture in {"arm64", "aarch64"}
    )

    if is_apple_silicon and _mlx_available():
        processor = platform.processor() or "Apple Silicon"

        return RuntimeInfo(
            kind=RuntimeKind.MLX,
            platform=system,
            architecture=architecture,
            device_name=processor,
            reason="Apple Silicon detected and MLX is available.",
        )

    # ---------------------------------------------------------
    # NVIDIA → CUDA
    # ---------------------------------------------------------

    cuda_available, cuda_name = _cuda_available()

    if cuda_available:
        return RuntimeInfo(
            kind=RuntimeKind.CUDA,
            platform=system,
            architecture=architecture,
            device_name=cuda_name or "NVIDIA CUDA GPU",
            reason="PyTorch detected a CUDA-capable NVIDIA GPU.",
        )

    # ---------------------------------------------------------
    # Universal fallback → CPU
    # ---------------------------------------------------------

    return RuntimeInfo(
        kind=RuntimeKind.CPU,
        platform=system,
        architecture=architecture,
        device_name="CPU",
        reason="No supported accelerator runtime detected.",
    )