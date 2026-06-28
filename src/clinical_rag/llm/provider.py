"""LLM generation provider — Phase 2 placeholder.

TODO (Phase 2): Implement pluggable LLM generation here.
  - Providers: Ollama (local default), Google Gemini, Groq — selected via
    config.llm_provider.
  - Config fields (llm_provider, llm_model, llm_base_url, llm_api_key) already
    exist in config.py and are exposed as env vars in .env.example.
  - Generation should accept retrieved chunks + user query and return a cited
    answer string. Never generate without grounding in retrieved context.
"""
