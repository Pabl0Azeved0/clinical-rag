from __future__ import annotations

import requests

from clinical_rag.llm.provider import LLMError, LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url
        self._model = model

    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=120,
            )
        except requests.RequestException as exc:
            raise LLMError(str(exc)) from exc
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        return resp.json()["response"]
