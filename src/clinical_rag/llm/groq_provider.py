from __future__ import annotations

import requests

from clinical_rag.llm.provider import LLMError, LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
        except requests.RequestException as exc:
            raise LLMError(str(exc)) from exc
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        return resp.json()["choices"][0]["message"]["content"]
