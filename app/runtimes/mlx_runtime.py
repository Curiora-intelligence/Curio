from __future__ import annotations

import gc
from pathlib import Path
from mlx_lm.sample_utils import make_sampler
from mlx_lm import generate as llm_generate
from mlx_lm import load as load_llm

from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as load_vlm
from mlx_vlm.prompt_utils import apply_chat_template as apply_vlm_chat_template
from mlx_vlm.utils import load_config as load_vlm_config

from app.runtimes.base import RuntimeAdapter


class MLXRuntime(RuntimeAdapter):

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._config = None

        self._model_id: str | None = None
        self._mode: str | None = None

    @property
    def name(self) -> str:
        return "MLX"

    @property
    def device(self) -> str:
        return "Apple Silicon GPU"

    # =========================================================
    # LIFECYCLE
    # =========================================================
    def _clean_text_output(self, text: str) -> str:
        markers = (
        "<|channel|>analysis<|message|>",
        "<|channel|>final<|message|>",
        "<|start|>assistant<|channel|>final<|message|>",
        "<|end|>",
        )

        for marker in markers:
            text = text.replace(marker, "")

        return text.strip()
        
    def _extract_final_response(self, text: str) -> str:
        """
        Extract the final user-visible response from Harmony structured output.
        The model may output internal reasoning in an 'analysis' channel before
        the 'final' channel. We must only return the final channel.
        """
        import re
        
        # Look for the final channel content
        pattern = r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|<\|channel\|>|$)"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            # We found an explicit final channel, return only its content
            answer = match.group(1).strip()
            # Clean any stray markers that regex might have grabbed
            return self._clean_text_output(answer)
            
        # Fallback: if GPT-OSS returns malformed structured output with no final channel,
        # prefer a safe fallback message rather than exposing raw structured output.
        raise RuntimeError("GPT-OSS did not produce a final channel before generation stopped.")
    
    def _debug_memory(self, label: str) -> None:
        import os
        import psutil
        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024 * 1024)
        
        try:
            import mlx.core as mx
            active_mem = mx.metal.get_active_memory() / (1024 * 1024)
            cache_mem = mx.metal.get_cache_memory() / (1024 * 1024)
            metal_str = f"Metal Active: {active_mem:.2f}MB, Metal Cache: {cache_mem:.2f}MB"
        except Exception:
            metal_str = "Metal metrics not available"
            
        print(f"[{label}] RSS: {rss_mb:.2f}MB | {metal_str} | mode: {self._mode} | model_id: {self._model_id}")

    def _clear_model(self) -> None:
        self._debug_memory("before release")
        
        # Explicitly delete old references
        if self._model is not None:
            del self._model
        if self._processor is not None:
            del self._processor
        if self._tokenizer is not None:
            del self._tokenizer
        if self._config is not None:
            del self._config
            
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._config = None
        self._model_id = None
        self._mode = None

        gc.collect()

        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass

        gc.collect()
        self._debug_memory("after release")

    def release(self) -> None:
        self._clear_model()

    # =========================================================
    # TEXT / GPT-OSS
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

        self._clear_model()

        print(
            f"Loading Curio LLM through MLX: {model_id}"
        )

        self._model, self._tokenizer = load_llm(
            model_id
        )

        self._model_id = model_id
        self._mode = "text"

        print(
            "Curio LLM loaded through MLX."
        )
        self._debug_memory("after load")

    def generate_text(
    self,
    *,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,) -> str:

        self._ensure_text_model(model_id)

        # Sanitize messages to avoid leaking harmony control tokens
        for msg in messages:
            if msg["role"] == "assistant":
                for marker in (
                    "<|channel|>analysis<|message|>",
                    "<|channel|>final<|message|>",
                    "<|end|>",
                    "<|start|>"
                ):
                    msg["content"] = msg["content"].replace(marker, "")

        formatted_prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        print("====== DEBUG HARMONY PROMPT STATE ======")
        print(f"Message count: {len(messages)}")
        print(f"Roles: {[m['role'] for m in messages]}")
        print(f"Message lengths: {[len(m['content']) for m in messages]}")
        print(f"Formatted prompt length: {len(formatted_prompt)}")
        print("========================================")

        sampler = make_sampler(
        temp=max(0.0, float(temperature)),
        top_p=1.0,
        top_k=0,
        )

        result = llm_generate(
        self._model,
        self._tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
        )

        answer = (
        result
        if isinstance(result, str)
        else str(result)
        ).strip()

        answer = self._extract_final_response(answer)

        if not answer:
            raise RuntimeError(
            "MLX LLM generated an empty response."
            )

        return answer
    # =========================================================
    # VISION / QWEN-VL
    # =========================================================

    def _ensure_vision_model(
        self,
        model_id: str,
    ) -> None:

        if (
            self._model is not None
            and self._processor is not None
            and self._config is not None
            and self._model_id == model_id
            and self._mode == "vision"
        ):
            return

        self._clear_model()

        print(
            f"Loading Curio VLM through MLX: {model_id}"
        )

        (
            self._model,
            self._processor,
        ) = load_vlm(model_id)

        self._config = load_vlm_config(
            model_id
        )

        self._model_id = model_id
        self._mode = "vision"

        print(
            "Curio VLM loaded through MLX."
        )
        self._debug_memory("after load")

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

        self._ensure_vision_model(
            model_id
        )

        formatted_prompt = (
            apply_vlm_chat_template(
                self._processor,
                self._config,
                messages,
                num_images=1,
            )
        )

        result = vlm_generate(
            self._model,
            self._processor,
            formatted_prompt,
            [str(image)],
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )

        answer = getattr(
            result,
            "text",
            None,
        )

        if answer is None:
            answer = str(result)
        
        for marker in (
        "<|channel|>analysis<|message|>",
        "<|channel|>final<|message|>",
        "<|end|>",
        "<|start|>assistant<|channel|>final<|message|>"):
            answer = answer.replace(marker, "")

        answer = answer.strip()

        answer = self._clean_text_output(answer)

        if not answer:
            raise RuntimeError(
                "MLX VLM generated an empty response."
            )

        return answer
