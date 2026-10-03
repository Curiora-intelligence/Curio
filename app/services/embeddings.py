"""Optional local embeddings; no model download or network calls at runtime."""
import asyncio
import os


class LocalEmbeddings:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    async def embed(self, text: str) -> list[float]:
        def encode():
            from sentence_transformers import SentenceTransformer
            if self._model is None:
                self._model = SentenceTransformer(self.model_path, device="cpu", local_files_only=True)
            return self._model.encode(text, normalize_embeddings=True).tolist()
        return await asyncio.to_thread(encode)


def configured_embeddings():
    path = os.getenv("CURIO_EMBEDDING_MODEL")
    return LocalEmbeddings(path) if path else None
