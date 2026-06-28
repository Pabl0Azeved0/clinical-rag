from __future__ import annotations


class Embedder:
    """Wraps SentenceTransformer with lazy model loading."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # loaded on first use

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
