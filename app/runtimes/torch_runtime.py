from __future__ import annotations

import gc
from pathlib import Path

from app.runtimes.base import RuntimeAdapter


class TorchRuntime(RuntimeAdapter):

    def __init__(self, device: str) -> None:
        self._device = device

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._model_id: str | None = None
        self._mode: str | None = None

    # =========================================================
    # PROPERTIES
    # =========================================================

    @property
    def name(self) -> str:
        return "PyTorch"

    @property
    def device(self) -> str:
        return self._device

    # =========================================================
    # MODEL LIFECYCLE
    # =========================================================

    def _release_current_model(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._model_id = None
        self._mode = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            pass

        gc.collect()

    # =========================================================
    # TEXT / LLM
    # =========================================================

    def _ensure_text_model(
        self,
        model_id: str,
    ) -> None:

        if (
            self._model is not None
            and self._tokenizer is not None
            and self._model_id == model_id
            and self._mode == "text"
        ):
            return

        self._release_current_model()

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )

        except ImportError as exc:
            raise RuntimeError(
                "PyTorch text runtime requires torch "
                "and transformers."
            ) from exc

        print(
            f"Loading Curio LLM through PyTorch "
            f"on {self._device}: {model_id}"
        )

        dtype = (
            torch.bfloat16
            if self._device == "cuda"
            else torch.float32
        )

        self._model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
            )
        )

        self._tokenizer = (
            AutoTokenizer.from_pretrained(
                model_id
            )
        )

        self._model_id = model_id
        self._mode = "text"

        print(
            "Curio LLM loaded through PyTorch."
        )

    def generate_text(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:

        self._ensure_text_model(model_id)

        import torch

        formatted_prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self._tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        model_device = next(
            self._model.parameters()
        ).device

        inputs = {
            key: value.to(model_device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        generated = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        input_length = (
            inputs["input_ids"].shape[-1]
        )

        output_ids = generated[
            0,
            input_length:
        ]

        answer = self._tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

        if not answer:
            raise RuntimeError(
                "PyTorch LLM generated an empty response."
            )

        return answer

    # =========================================================
    # VISION / VLM
    # =========================================================

    def _ensure_vision_model(
        self,
        model_id: str,
    ) -> None:

        if (
            self._model is not None
            and self._processor is not None
            and self._model_id == model_id
            and self._mode == "vision"
        ):
            return

        self._release_current_model()

        try:
            import torch
            from transformers import (
                AutoProcessor,
                Qwen3VLForConditionalGeneration,
            )

        except ImportError as exc:
            raise RuntimeError(
                "PyTorch vision runtime requires torch and "
                "a Transformers version with Qwen3-VL support."
            ) from exc

        print(
            f"Loading Curio VLM through PyTorch "
            f"on {self._device}: {model_id}"
        )

        dtype = (
            torch.bfloat16
            if self._device == "cuda"
            else torch.float32
        )

        self._model = (
            Qwen3VLForConditionalGeneration
            .from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
            )
        )

        self._processor = (
            AutoProcessor.from_pretrained(
                model_id
            )
        )

        self._model_id = model_id
        self._mode = "vision"

        print(
            "Curio VLM loaded through PyTorch."
        )

    def generate_vision(
        self,
        *,
        model_id: str,
        image_path: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:

        image = Path(image_path)

        if not image.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image}"
            )

        self._ensure_vision_model(model_id)

        formatted_messages = []
        for i, msg in enumerate(messages):
            # For Qwen3-VL, text needs to be structured in content array
            if i == len(messages) - 1 and msg["role"] == "user":
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(image),
                        },
                        {
                            "type": "text",
                            "text": msg["content"],
                        },
                    ]
                })
            else:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"],
                        }
                    ]
                })

        inputs = (
            self._processor.apply_chat_template(
                formatted_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        model_device = next(
            self._model.parameters()
        ).device

        inputs = {
            key: value.to(model_device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        input_length = (
            inputs["input_ids"].shape[-1]
        )

        generated_ids_trimmed = [
            output_ids[input_length:]
            for output_ids in generated_ids
        ]

        output_text = (
            self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

        answer = output_text[0].strip()

        if not answer:
            raise RuntimeError(
                "PyTorch VLM generated an empty response."
            )

        return answer

    # =========================================================
    # RELEASE
    # =========================================================

    def release(self) -> None:
        self._release_current_model()