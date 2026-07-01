"""Optional OpenTelemetry tracing via Logfire.

Off by default (ENABLE_TRACING=false) so tests, CI, and offline runs are
unaffected. When enabled it instruments the PydanticAI agent (tool-call spans and
timings) and the OpenAI-compatible client (token usage / latency). Without a
LOGFIRE_TOKEN it emits traces to the console only (send_to_logfire=False); with a
token it also ships them to Logfire.
"""

from __future__ import annotations

from clinical_rag.config import Settings

_CONFIGURED = {"done": False}


def setup_tracing(settings: Settings) -> bool:
    """Configure Logfire tracing when enabled. Returns True if tracing is active.

    Idempotent: safe to call on every Streamlit rerun / script start.
    """
    if not settings.enable_tracing:
        return False
    if _CONFIGURED["done"]:
        return True

    import logfire

    logfire.configure(
        service_name="clinical-rag",
        send_to_logfire=bool(settings.logfire_token),
        token=settings.logfire_token or None,
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_openai()
    _CONFIGURED["done"] = True
    return True
