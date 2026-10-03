from __future__ import annotations

from abc import ABC, abstractmethod

from app.runtimes.turn import ModelTurn


class RuntimeAdapter(ABC):
    """
    Common interface for Curio inference runtimes.

    The application layer does not know whether inference
    is running through MLX, CUDA, or CPU.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        raise NotImplementedError

    def generate_turn(self, *, model_id: str, messages: list[dict], tools: list[dict],
                      max_tokens: int, temperature: float) -> ModelTurn:
        # Compatibility for simple adapters; GPT-OSS adapters override this method.
        return ModelTurn(final=self.generate_text(model_id=model_id, messages=messages,
                                                  max_tokens=max_tokens, temperature=temperature))

    @abstractmethod
    def generate_vision(
        self,
        *,
        model_id: str,
        image_path: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError