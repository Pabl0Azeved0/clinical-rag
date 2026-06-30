from __future__ import annotations

import requests

from clinical_rag.llm.provider import LLMError, LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def generate(self, prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models"
            f"/{self._model}:generateContent?key={self._api_key}"
        )
        try:
            resp = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=120,
            )
        except requests.RequestException as exc:
            raise LLMError(str(exc)) from exc
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
