"""Tests for the generation phase — no network or live LLM calls."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.generation.generator import DISCLAIMER, Answer, Generator
from clinical_rag.llm.provider import LLMError, LLMProvider

# ── FakeProvider ─────────────────────────────────────────────────────────────


class FakeProvider(LLMProvider):
    def __init__(self, response: str = "model output") -> None:
        self._response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


# ── Generator.generate_answer with chunks ────────────────────────────────────


def _make_chunks() -> list[dict]:
    return [
        {
            "text": "Chunk one text about diabetes.",
            "title": "Diabetes",
            "url": "https://medlineplus.gov/diabetes.html",
            "doc_id": "d1",
            "distance": 0.1,
        },
        {
            "text": "Chunk two text about insulin.",
            "title": "Insulin",
            "url": "https://medlineplus.gov/insulin.html",
            "doc_id": "d2",
            "distance": 0.2,
        },
    ]


def test_generate_answer_text_contains_model_output_and_disclaimer():
    provider = FakeProvider("Here is a fine answer [1].")
    gen = Generator(provider)
    answer = gen.generate_answer("What is diabetes?", _make_chunks())

    assert isinstance(answer, Answer)
    assert "Here is a fine answer [1]." in answer.text
    assert DISCLAIMER in answer.text


def test_generate_answer_sources_two_entries():
    provider = FakeProvider()
    gen = Generator(provider)
    answer = gen.generate_answer("What is diabetes?", _make_chunks())

    assert len(answer.sources) == 2
    assert answer.sources[0] == {
        "index": 1,
        "title": "Diabetes",
        "url": "https://medlineplus.gov/diabetes.html",
    }
    assert answer.sources[1] == {
        "index": 2,
        "title": "Insulin",
        "url": "https://medlineplus.gov/insulin.html",
    }


def test_generate_answer_prompt_contains_chunks_and_query():
    provider = FakeProvider()
    gen = Generator(provider)
    query = "What is diabetes?"
    gen.generate_answer(query, _make_chunks())

    prompt = provider.last_prompt
    assert prompt is not None
    assert query in prompt
    assert "Chunk one text about diabetes." in prompt
    assert "Chunk two text about insulin." in prompt
    assert "[1]" in prompt
    assert "[2]" in prompt


# ── Generator.generate_answer with empty chunks ──────────────────────────────


def test_generate_answer_empty_chunks_no_provider_call():
    provider = FakeProvider()
    gen = Generator(provider)
    answer = gen.generate_answer("What is diabetes?", [])

    assert answer.sources == []
    assert DISCLAIMER in answer.text
    assert provider.last_prompt is None  # provider was not called


# ── get_provider factory ─────────────────────────────────────────────────────


def _settings(**kwargs):
    defaults = {
        "llm_provider": "ollama",
        "llm_base_url": "http://localhost:11434",
        "llm_model": "llama3.2:3b",
        "llm_api_key": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_get_provider_ollama_returns_ollama_provider():
    from clinical_rag.llm.ollama_provider import OllamaProvider
    from clinical_rag.llm.provider import get_provider

    provider = get_provider(_settings(llm_provider="ollama"))
    assert isinstance(provider, OllamaProvider)


def test_get_provider_gemini_no_api_key_raises():
    from clinical_rag.llm.provider import get_provider

    with pytest.raises(LLMError):
        get_provider(_settings(llm_provider="gemini", llm_api_key=None))


def test_get_provider_unknown_raises():
    from clinical_rag.llm.provider import get_provider

    with pytest.raises(LLMError, match="unknown llm provider"):
        get_provider(_settings(llm_provider="bogus"))


# ── OllamaProvider.generate (monkeypatched) ──────────────────────────────────


def test_ollama_provider_generate(monkeypatch):
    import clinical_rag.llm.ollama_provider as mod
    from clinical_rag.llm.ollama_provider import OllamaProvider

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "hi"}

    posted_kwargs: list[dict] = []

    def fake_post(url, **kwargs):
        posted_kwargs.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(mod.requests, "post", fake_post)

    provider = OllamaProvider("http://localhost:11434", "llama3.2:3b")
    result = provider.generate("tell me about aspirin")

    assert result == "hi"
    assert len(posted_kwargs) == 1
    body = posted_kwargs[0]["json"]
    assert "model" in body
    assert "prompt" in body
    assert "stream" in body
