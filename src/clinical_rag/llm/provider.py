from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(Exception):
    pass


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...


def get_provider(settings) -> LLMProvider:
    name = settings.llm_provider
    if name == "ollama":
        from clinical_rag.llm.ollama_provider import OllamaProvider

        return OllamaProvider(settings.llm_base_url, settings.llm_model)
    if name == "gemini":
        from clinical_rag.llm.gemini_provider import GeminiProvider

        if not settings.llm_api_key:
            raise LLMError("llm_api_key is required for gemini")
        return GeminiProvider(settings.llm_api_key, settings.llm_model)
    if name == "groq":
        from clinical_rag.llm.groq_provider import GroqProvider

        if not settings.llm_api_key:
            raise LLMError("llm_api_key is required for groq")
        return GroqProvider(settings.llm_api_key, settings.llm_model)
    raise LLMError(f"unknown llm provider: {name}")
