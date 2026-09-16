from __future__ import annotations

from pathlib import Path

from app.services.model_gateway import ModelGateway


MAX_TOKENS = 384
TEMPERATURE = 0.2


class VisionService:

    def __init__(
        self,
        gateway: ModelGateway,
    ) -> None:
        self.gateway = gateway

    def analyze(
        self,
        image_path: str | Path,
        message: str = "",
    ) -> str:

        image_path = Path(
            image_path
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        user_request = (
            message.strip()
            if message
            else "Describe what you see in this image."
        )

        prompt = (
            "You are Curio, the multimodal intelligence "
            "system developed by Curiora.\n\n"

            "Analyze only what is supported by the image.\n"
            "Distinguish observation from inference.\n"
            "Do not invent identity, ownership, location, "
            "measurements, causes, events, or actions.\n"
            "State uncertainty when visual evidence is insufficient.\n\n"

            f"USER REQUEST:\n{user_request}"
        )

        return self.gateway.generate_vision(
            image_path=str(image_path),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
