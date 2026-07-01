"""Streamlit chat UI for the clinical-rag agent.

A chat-app layout (like Claude/ChatGPT): conversation list + config in the left
sidebar, chat bubbles in the main pane, an in-box send button, and a skeleton
loading state while the agent runs. Low-grounding / low-confidence answers pause
for a human approve / edit / reject decision (HITL). The not-medical-advice
disclaimer is appended here, outside the agent (R5).

Run with:  make ui   (or  .venv/bin/streamlit run app/streamlit_app.py)
"""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from clinical_rag.agent.clinical_agent import AgentDeps, build_agent, run_agent
from clinical_rag.agent.hitl import Decision, apply_decision, needs_approval
from clinical_rag.config import get_settings
from clinical_rag.embeddings.embedder import Embedder
from clinical_rag.generation.generator import DISCLAIMER
from clinical_rag.retrieval.retriever import Retriever
from clinical_rag.vectorstore.chroma_store import ChromaStore

# ── backend (unchanged logic; Streamlit-persistent singletons) ────────────────


@st.cache_resource
def _store():
    """Heavy embedder + vector store — built once per process, shared across reruns."""
    cfg = get_settings()
    embedder = Embedder(cfg.embedding_model_name)
    return ChromaStore(cfg.resolve(cfg.chroma_dir), cfg.collection_name, embedder)


@st.cache_resource
def _executor() -> concurrent.futures.ThreadPoolExecutor:
    """Persistent pool so a timed-out run doesn't block on executor shutdown."""
    return concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _norm(question: str) -> str:
    """Normalize a question for cache hits: trim, lowercase, drop trailing punctuation."""
    return question.strip().lower().rstrip("?.! ").strip()


def run_query(question: str, k: int) -> dict:
    """Run one agent turn, bounded by a wall-clock timeout, returning output + trajectory."""
    cfg = get_settings()
    deps = AgentDeps(retriever=Retriever(_store(), k), settings=cfg)
    agent = build_agent(cfg)
    future = _executor().submit(run_agent, agent, question, deps)
    try:
        output, tool_calls = future.result(timeout=cfg.agent_timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(
            f"The model didn't finish within {int(cfg.agent_timeout)}s — try again or "
            "rephrase. (A paid Groq tier / stronger model answers faster.)"
        )
    return {
        "question": question,
        "output": output,
        "tool_calls": tool_calls,
        "retrieved": deps.retrieved,
        "broaden_used": deps.broaden_used,
    }


def _get_cached(question: str, k: int) -> dict:
    cache = st.session_state.setdefault("query_cache", {})
    key = (_norm(question), k)
    if key not in cache:
        cache[key] = run_query(question, k)
    return cache[key]


# ── conversation state ────────────────────────────────────────────────────────


def _init_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
        st.session_state.conv_counter = 0
        _new_conversation()


def _new_conversation() -> str:
    st.session_state.conv_counter += 1
    cid = f"c{st.session_state.conv_counter}"
    st.session_state.conversations[cid] = {
        "title": None,
        "turns": [],
        "favorite": False,
    }
    st.session_state.active = cid
    return cid


def _active_conversation() -> dict:
    return st.session_state.conversations[st.session_state.active]


def _maybe_title(conv: dict, prompt: str) -> None:
    if conv["title"] is None:
        conv["title"] = prompt.strip()[:40]


def _delete_conversation(cid: str) -> None:
    st.session_state.conversations.pop(cid, None)
    if not st.session_state.conversations:
        _new_conversation()  # always keep at least one
    elif st.session_state.active == cid:
        st.session_state.active = next(iter(st.session_state.conversations))


@st.dialog("Delete conversation?")
def _confirm_delete(cid: str) -> None:
    conv = st.session_state.conversations.get(cid)
    if conv is None:
        st.session_state.pop("pending_delete", None)
        return
    st.write(f"Delete “{conv['title'] or 'New chat'}”? This can't be undone.")
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Delete", type="primary", use_container_width=True):
        _delete_conversation(cid)
        st.session_state.pop("pending_delete", None)
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        st.session_state.pop("pending_delete", None)
        st.rerun()


@st.dialog("Rename conversation")
def _rename_dialog(cid: str) -> None:
    conv = st.session_state.conversations.get(cid)
    if conv is None:
        st.session_state.pop("pending_rename", None)
        return
    new = st.text_input("Name", value=conv["title"] or "")
    c1, c2 = st.columns(2)
    if c1.button("Save", type="primary", use_container_width=True):
        if new.strip():
            conv["title"] = new.strip()[:40]
        st.session_state.pop("pending_rename", None)
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        st.session_state.pop("pending_rename", None)
        st.rerun()


@st.dialog("Logs")
def _error_dialog(msg: str) -> None:
    st.caption("Technical details (include these if you report the issue):")
    st.code(msg)


# ── rendering ─────────────────────────────────────────────────────────────────


def _render_error(idx: int, msg: str) -> None:
    """User-facing error: a reassuring message + an opt-in 'Check logs' modal."""
    st.error("⚠️ An error occurred, please try again later.")
    if st.button("Check logs", key=f"log_{idx}"):
        st.session_state.show_error = msg
        st.rerun()


_GROUNDED_HELP = (
    "Deterministic — 'yes' only when the answer cites a retrieved passage [n] that "
    "exists. Citation coverage, not the model's self-report."
)
_CONFIDENCE_HELP = (
    "From retrieval distance of the cited/retrieved passages (closer = higher), "
    "not the model's self-rating. 0–1."
)


def _skeleton() -> None:
    """Loading placeholder: metric columns + tooltips visible, values shimmering."""
    c1, c2 = st.columns(2)
    c1.metric("Grounded", "···", help=_GROUNDED_HELP)
    c2.metric("Confidence", "···", help=_CONFIDENCE_HELP)
    st.caption("🔎 Retrieving evidence and reasoning…")


def _metrics(out) -> None:
    c1, c2 = st.columns(2)
    c1.metric("Grounded", "yes" if out.grounded else "no", help=_GROUNDED_HELP)
    c2.metric("Confidence", f"{out.confidence:.2f}", help=_CONFIDENCE_HELP)


def _trajectory(res: dict) -> None:
    with st.expander("Agent trajectory"):
        seq = " → ".join(tc["tool"] for tc in res["tool_calls"]) or "(none)"
        st.write(f"**Tools:** {seq}  ·  **Broaden used:** {res['broaden_used']}")
        for tc in res["tool_calls"]:
            st.code(f"{tc['tool']}({tc['args']})", language="python")
        if res["retrieved"]:
            st.markdown("**Retrieved passages**")
            for idx in sorted(res["retrieved"]):
                r = res["retrieved"][idx]
                st.markdown(
                    f"[{idx}] {r['title']} — distance {r['distance']:.3f} — {r['url']}"
                )


def _citations(out) -> None:
    if out.citations:
        st.markdown("**Citations**")
        for c in out.citations:
            st.markdown(f"- **[{c.index}]** {c.title} — [{c.url}]({c.url})")


def _pending(conv: dict, idx: int, res: dict, threshold: float) -> None:
    out = res["output"]
    reasons = []
    if not out.grounded:
        reasons.append("it isn't grounded in a cited passage")
    if out.confidence < threshold:
        reasons.append(f"confidence {out.confidence:.2f} is below {threshold:.2f}")
    st.warning("⏸ **Held for human review** — " + " and ".join(reasons) + ".")
    _metrics(out)
    edited = st.text_area(
        "Draft (edit before approving)", value=out.answer, key=f"edit{idx}"
    )
    a, b, c = st.columns(3)
    if a.button("✅ Approve", key=f"ap{idx}", use_container_width=True):
        conv["turns"][idx]["decision"] = Decision("approve")
        st.rerun()
    if b.button("✏️ Approve edited", key=f"ed{idx}", use_container_width=True):
        conv["turns"][idx]["decision"] = Decision("edit", edited)
        st.rerun()
    if c.button("🚫 Reject", key=f"rj{idx}", use_container_width=True):
        conv["turns"][idx]["decision"] = Decision("reject")
        st.rerun()
    _trajectory(res)


def _resolved(res: dict, decision: Decision | None) -> None:
    out = res["output"]
    if decision is not None:
        final = apply_decision(out, decision)
        if final is None:
            st.error("🚫 Rejected by reviewer — not surfaced.")
            _trajectory(res)
            return
        st.caption(f"✔ Approved by reviewer ({decision.action})")
        text = final.answer
    else:
        text = out.answer
    _metrics(out)
    st.markdown(text)
    _citations(out)
    st.caption(DISCLAIMER)
    _trajectory(res)


def _answer_block(conv: dict, idx: int, res: dict) -> None:
    out = res["output"]
    cfg = get_settings()
    decision = conv["turns"][idx].get("decision")
    if decision is None and needs_approval(out, cfg.approval_confidence_threshold):
        _pending(conv, idx, res, cfg.approval_confidence_threshold)
    else:
        _resolved(res, decision)


def _render_turn(conv: dict, idx: int, turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        if turn.get("error"):
            _render_error(idx, turn["error"])
        else:
            _answer_block(conv, idx, turn["res"])


def _submit(conv: dict, prompt: str) -> None:
    k = st.session_state.get("k", get_settings().top_k)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        holder = st.empty()
        with holder.container():
            _skeleton()
        try:
            res = _get_cached(prompt, k)
        except Exception as exc:
            conv["turns"].append(
                {"q": prompt, "res": None, "decision": None, "error": str(exc)}
            )
            idx = len(conv["turns"]) - 1
            holder.empty()
            with holder.container():
                _render_error(idx, str(exc))
            _maybe_title(conv, prompt)
            return
        conv["turns"].append({"q": prompt, "res": res, "decision": None})
        idx = len(conv["turns"]) - 1
        holder.empty()
        with holder.container():
            _answer_block(conv, idx, res)
    _maybe_title(conv, prompt)


# ── sidebar + page ────────────────────────────────────────────────────────────


def _inject_css() -> None:
    # Only the conversations list scrolls (it's the one st.container(height=...) in the
    # sidebar); stretch it to fill so the header stays on top and Configuration sits at
    # the bottom. The ~15rem offset leaves room for the header + collapsed config.
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]{
            height: calc(100vh - 9rem) !important;
            overflow-y: auto !important;
          }
          /* Tighten the row so the ⋯ hugs the conversation name. */
          section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]{
            gap: 0.15rem !important; align-items: center;
          }
          /* ⋯ menu trigger: borderless icon, ALWAYS visible (subtle), full on hover.
             The menu is a popover, so it floats to the side and never vanishes on mouse-out. */
          section[data-testid="stSidebar"] [data-testid="stPopover"] > button{
            border: none !important; background: transparent !important;
            box-shadow: none !important; min-height: 0 !important;
            padding: 0.25rem !important; opacity: 0.6;
          }
          section[data-testid="stSidebar"] [data-testid="stPopover"] > button:hover{
            opacity: 1;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _conv_row(cid: str) -> None:
    """One conversation row: name (click to open) + a floating ⋯ options menu.

    The menu is a popover — it floats to the side and stays open until you click away
    or pick an option. The trigger icon is ALWAYS visible (no hover-hide), so it never
    fades out on mouse-move.
    """
    conv = st.session_state.conversations[cid]
    label = conv["title"] or "New chat"
    active = cid == st.session_state.active

    nav, menu = st.columns([0.85, 0.15])
    if nav.button(
        label,
        key=f"nav_{cid}",
        use_container_width=True,
        type="primary" if active else "secondary",
    ):
        st.session_state.active = cid
        st.rerun()
    with menu.popover("", icon=":material/more_horiz:", use_container_width=True):
        fav = conv.get("favorite", False)
        if st.button(
            "☆ Unfavorite" if fav else "⭐ Favorite",
            key=f"fav_{cid}",
            use_container_width=True,
        ):
            conv["favorite"] = not fav
            st.rerun()
        if st.button("✏️ Rename", key=f"ren_{cid}", use_container_width=True):
            st.session_state.pending_rename = cid
            st.rerun()
        if st.button("🗑️ Delete", key=f"del_{cid}", use_container_width=True):
            st.session_state.pending_delete = cid
            st.rerun()


def _sidebar() -> None:
    cfg = get_settings()
    with st.sidebar:
        st.markdown("### 🩺 clinical-rag")
        if st.button("➕  New chat", use_container_width=True):
            # Don't pile up empty chats: reuse an existing blank one if there is one.
            empty = next(
                (
                    cid
                    for cid, c in st.session_state.conversations.items()
                    if not c["turns"]
                ),
                None,
            )
            st.session_state.active = empty or _new_conversation()
            st.session_state.menu_open = None
            st.rerun()

        order = list(reversed(list(st.session_state.conversations)))
        favs = [
            cid for cid in order if st.session_state.conversations[cid].get("favorite")
        ]
        others = [
            cid
            for cid in order
            if not st.session_state.conversations[cid].get("favorite")
        ]

        # Only this container scrolls (stretched to fill via CSS); header + config stay put.
        with st.container(height=420):
            if favs:  # section shown only when something is favorited
                st.markdown("**⭐ Favorites**")
                for cid in favs:
                    _conv_row(cid)
                st.markdown("**Recents**")
            for cid in others:
                _conv_row(cid)

        with st.expander("⚙️  Configuration", expanded=False):
            st.write(f"**Model:** {cfg.llm_provider} · `{cfg.llm_model}`")
            st.write(
                f"**Index:** `{cfg.collection_name}`  ·  **Retrieval:** `{cfg.retrieval_transport}`"
            )
            st.session_state.k = st.slider("Passages to retrieve (k)", 1, 10, cfg.top_k)
            st.caption(
                f"HITL gate: pause if not grounded or confidence "
                f"< {cfg.approval_confidence_threshold:.2f}."
            )
            cache = st.session_state.get("query_cache", {})
            if cache:
                st.caption(f"🗄️ {len(cache)} cached answer(s) this session")


def main() -> None:
    cfg = get_settings()
    from clinical_rag.observability import setup_tracing

    setup_tracing(cfg)
    st.set_page_config(
        page_title="Clinical Evidence Agent", page_icon="🩺", layout="centered"
    )
    _inject_css()
    _init_state()
    _sidebar()

    if st.session_state.get("pending_delete"):
        _confirm_delete(st.session_state.pending_delete)
    if st.session_state.get("pending_rename"):
        _rename_dialog(st.session_state.pending_rename)
    err = st.session_state.pop(
        "show_error", None
    )  # pop-before-open so X-close won't reopen
    if err:
        _error_dialog(err)

    conv = _active_conversation()
    st.title("🩺 Clinical Evidence Agent")
    st.caption(
        "Retrieval-grounded answers from MedlinePlus — for information only, not medical advice."
    )

    for idx, turn in enumerate(conv["turns"]):
        _render_turn(conv, idx, turn)

    busy = st.session_state.get("busy", False)
    prompt = st.chat_input(
        "Answering… please wait" if busy else "Ask a clinical question…",
        disabled=busy,
    )
    # Two-step submit: flag busy + rerun FIRST so the input renders disabled BEFORE the
    # blocking run starts — otherwise a second question sent mid-answer hijacks the run.
    if prompt and prompt.strip() and not busy:
        st.session_state.busy = True
        st.session_state.pending_prompt = prompt.strip()
        st.rerun()
    if busy and "pending_prompt" in st.session_state:
        p = st.session_state.pop("pending_prompt")
        try:
            _submit(conv, p)
        finally:
            st.session_state.busy = False
        st.rerun()


if __name__ == "__main__":
    main()
