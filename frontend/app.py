"""
Streamlit Frontend — Editorial Intelligence System
==================================================

Run:
    streamlit run frontend/app.py

The app calls the FastAPI backend at http://localhost:8000 (configurable
in the sidebar). All four workflow types are available as tabs.
"""

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
import streamlit as st

# Ensure repo root is importable when Streamlit runs frontend/app.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ==========================================================================
# Page config — must be first Streamlit call
# ==========================================================================

st.set_page_config(
    page_title="Editorial Intelligence System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Session state initialisation
# ==========================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "results" not in st.session_state:
    st.session_state.results = {}          # keyed by tab name
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []     # raw persisted turns
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None  # opened history card
if "main_view" not in st.session_state:
    st.session_state.main_view = "create"  # create | history
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
API_BASE_URL = "http://54.218.34.106:9000"
#API_BASE_URL = "http://localhost:8000"

# Always point at the deployed API (do not let an old empty session value stick).
st.session_state.api_url = API_BASE_URL
if "brands" not in st.session_state:
    st.session_state.brands = []

# Brand font → CSS stack (licensed fonts fall back gracefully)
_BRAND_FONT_STACKS = {
    "Google Sans Flex Regular": (
        '"Google Sans Flex", "Google Sans", "Product Sans", system-ui, sans-serif'
    ),
    "Open Sans Regular": '"Open Sans", "Segoe UI", sans-serif',
    "Gotham Regular": 'Gotham, Montserrat, "Helvetica Neue", Arial, sans-serif',
    "Inter Regular": 'Inter, "Segoe UI", sans-serif',
}


# ==========================================================================
# Helper: call the API
# ==========================================================================


def call_api(endpoint: str, payload: Dict[str, Any], timeout: int = 900) -> Dict[str, Any]:
    """POST to the FastAPI backend. Returns the JSON response dict.

    Default timeout is 15 minutes — long-form content can run many serial
    LLM calls (research → strategy → write sections → review revisions).
    """
    url = f"{st.session_state.api_url}/api/{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"ok": False, "errors": [f"Cannot connect to API at {st.session_state.api_url}. Is the server running?"]}
    except requests.exceptions.Timeout:
        return {"ok": False, "errors": ["Request timed out. The workflow may still be running — try again."]}
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return {"ok": False, "errors": [f"API error {exc.response.status_code}: {detail}"]}
    except Exception as exc:
        return {"ok": False, "errors": [f"Unexpected error: {exc}"]}


def fetch_brands() -> List[Dict]:
    """Fetch brand list from /api/brands. Returns [] on failure."""
    try:
        resp = requests.get(f"{st.session_state.api_url}/api/brands", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("brands", [])
    except Exception:
        pass
    return []


def check_api_health() -> bool:
    try:
        resp = requests.get(f"{st.session_state.api_url}/api/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


LANGUAGE_OPTIONS = ["Auto-detect", "English", "Hindi", "Hinglish"]
_IST = ZoneInfo("Asia/Kolkata")


def _display_name(username: str) -> str:
    """Friendly name for greetings (rishav.patel → Rishav)."""
    raw = (username or "").strip()
    if not raw:
        return "there"
    token = raw.replace("_", " ").replace(".", " ").split()[0]
    return token[:1].upper() + token[1:] if token else "there"


def ist_greeting(username: str = "") -> str:
    """Hello {User}, Good morning/afternoon/evening — based on IST."""
    hour = datetime.now(_IST).hour
    if hour < 12:
        part = "Good morning"
    elif hour < 17:
        part = "Good afternoon"
    else:
        part = "Good evening"
    return f"Hello {_display_name(username)}, {part}"


def fetch_session_history(session_id: str, limit: int = 50) -> List[Dict]:
    """Load conversation turns from API memory (Mongo-backed)."""
    try:
        resp = requests.get(
            f"{st.session_state.api_url}/api/sessions/{session_id}/history",
            params={"limit": limit},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("turns") or []
    except Exception:
        pass
    return []


def fetch_user_history(username: str, limit: int = 100) -> List[Dict]:
    """Load persisted per-user chat history (survives logout/login)."""
    if not username:
        return []
    try:
        resp = requests.get(
            f"{st.session_state.api_url}/api/chat/users/{username}/history",
            params={"limit": limit},
            timeout=8,
        )
        if resp.status_code == 200:
            return resp.json().get("turns") or []
    except Exception:
        pass
    # Direct service fallback when API is down but Mongo/file is local
    try:
        from services.chat_history_service import get_user_history

        return get_user_history(username, limit=limit)
    except Exception:
        return []


def fetch_team_activity(limit: int = 60) -> List[Dict]:
    """Recent activity across users (for admin / team visibility)."""
    try:
        resp = requests.get(
            f"{st.session_state.api_url}/api/chat/activity",
            params={"limit": limit},
            timeout=8,
        )
        if resp.status_code == 200:
            return resp.json().get("turns") or []
    except Exception:
        pass
    try:
        from services.chat_history_service import get_all_recent_activity

        return get_all_recent_activity(limit=limit)
    except Exception:
        return []


def _persist_turn_remote(
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a turn for the logged-in user (API → Mongo/file)."""
    username = (st.session_state.get("username") or "").strip()
    if not username or not content:
        return
    payload = {
        "username": username,
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "session_id": st.session_state.get("session_id") or "",
    }
    try:
        requests.post(
            f"{st.session_state.api_url}/api/chat/turns",
            json=payload,
            timeout=8,
        )
        return
    except Exception:
        pass
    try:
        from services.chat_history_service import add_turn

        add_turn(
            username=username,
            role=role,
            content=content,
            metadata=metadata,
            session_id=st.session_state.get("session_id") or "",
        )
    except Exception:
        pass


def append_chat_turn(
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    persist: bool = True,
) -> None:
    """Append a turn to the in-browser history and optionally persist by username."""
    turn = {
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "username": st.session_state.get("username") or "",
    }
    st.session_state.chat_history.append(turn)
    if persist:
        _persist_turn_remote(role, content, metadata)


def _slim_result_for_history(result: Dict[str, Any], max_md: int = 80000) -> Dict[str, Any]:
    """Keep enough of a result to reopen Content / Review / Metadata later."""
    if not isinstance(result, dict):
        return {"ok": False, "errors": ["Invalid result"]}
    slim = {
        "ok": result.get("ok"),
        "errors": result.get("errors") or [],
        "review": result.get("review") or {},
        "metadata": result.get("metadata") or {},
        "revision_count": result.get("revision_count", 0),
        "email_meta": result.get("email_meta") or {},
        "social_meta": result.get("social_meta") or {},
        "seo_analysis": result.get("seo_analysis") or {},
    }
    final = dict(result.get("final_output") or {})
    content = final.get("content")
    if isinstance(content, dict):
        md = str(content.get("markdown") or "")
        if len(md) > max_md:
            md = md[:max_md] + "\n\n…[truncated for history storage]"
        final["content"] = {**content, "markdown": md}
    slim["final_output"] = final
    return slim


def record_generation(
    user_prompt: str,
    result: Dict[str, Any],
    workflow: str,
) -> None:
    """Save one ChatGPT-style conversation card (prompt title + full result)."""
    from services.chat_history_service import make_conversation_title

    prompt = (user_prompt or "").strip()
    title = make_conversation_title(prompt)
    review = (result or {}).get("review") or {}
    final = (result or {}).get("final_output") or {}
    social_meta = (result or {}).get("social_meta") or {}
    hashtags = final.get("hashtags") or social_meta.get("hashtags") or []
    md = _get_markdown(result or {}) if (result or {}).get("ok") else ""
    preview = (md[:500] + "…") if len(md) > 500 else md
    if not (result or {}).get("ok"):
        errs = (result or {}).get("errors") or ["Generation failed."]
        preview = "Error: " + "; ".join(str(e) for e in errs)

    conversation_id = str(uuid.uuid4())
    meta = {
        "conversation_id": conversation_id,
        "title": title,
        "prompt": prompt,
        "workflow": workflow,
        "ok": bool((result or {}).get("ok")),
        "score": review.get("score"),
        "status": review.get("status"),
        "hashtags": hashtags,
        "result": _slim_result_for_history(result or {}),
    }
    # Content field doubles as searchable preview; title lives in metadata
    append_chat_turn("conversation", preview or title, meta)

    # Open the new card immediately in the history viewer
    st.session_state.active_conversation = {
        "id": conversation_id,
        "title": title,
        "prompt": prompt,
        "workflow": workflow,
        "score": review.get("score"),
        "status": review.get("status"),
        "ok": meta["ok"],
        "result": meta["result"],
        "preview": preview,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "username": st.session_state.get("username") or "",
        "hashtags": hashtags,
    }
    st.session_state.main_view = "create"  # stay on create after generate


def _font_stack_for_result(result: Dict, brand_label: str = "") -> str:
    """Resolve CSS font-family from result payload or selected brand."""
    final = result.get("final_output") or {}
    font_name = str(final.get("font") or "").strip()
    if not font_name and brand_label and brand_label != "Auto-detect":
        for b in st.session_state.brands:
            if b.get("display_name") == brand_label or b.get("name") == brand_label:
                font_name = str(b.get("font") or "").strip()
                break
    return _BRAND_FONT_STACKS.get(font_name, '"Segoe UI", system-ui, sans-serif')


def _inject_brand_font_css(font_stack: str) -> None:
    """Apply brand font + load webfonts for Open Sans / Inter."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Open+Sans:wght@400;600&family=Montserrat:wght@400;500;600&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <style>
        .brand-content-preview, .brand-content-preview * {{
            font-family: {font_stack} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_app_chrome_css() -> None:
    """Eye-friendly UI: teal for clickable controls, orange Generate CTA."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --ei-bg: #f6f8f7;
            --ei-surface: #ffffff;
            --ei-ink: #1a2e2a;
            --ei-muted: #5f736c;
            --ei-accent: #0f766e;
            --ei-accent-soft: #d7f5ef;
            --ei-accent-border: #5eead4;
            --ei-line: #d8e3de;
            --ei-sidebar: #eef4f1;
            --ei-cta: #ea580c;
            --ei-cta-hover: #c2410c;
            --ei-control-bg: #e6f7f3;
        }

        html, body, [class*="css"] {
            font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(1200px 500px at 10% -10%, #d9f5ef 0%, transparent 55%),
                radial-gradient(900px 420px at 100% 0%, #e8f1ff 0%, transparent 50%),
                var(--ei-bg);
        }
        .main .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.5rem;
            max-width: 1040px;
        }
        h1, h2, h3 {
            font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif !important;
            color: var(--ei-ink) !important;
            letter-spacing: -0.02em;
            font-weight: 700 !important;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: var(--ei-sidebar) !important;
            border-right: 1px solid var(--ei-line);
        }
        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
            padding-top: 0.75rem;
        }
        [data-testid="stSidebar"] * {
            color: var(--ei-ink) !important;
        }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] caption,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--ei-muted) !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: var(--ei-surface);
            border: 1px solid var(--ei-line);
            color: var(--ei-ink) !important;
            text-align: left;
            justify-content: flex-start;
            border-radius: 10px;
            padding: 0.42rem 0.7rem;
            font-size: 0.86rem;
            font-weight: 500;
            line-height: 1.25;
            min-height: 0;
            box-shadow: none;
            white-space: normal;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: var(--ei-accent-soft);
            border-color: var(--ei-accent-border);
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        [data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
            background: var(--ei-accent) !important;
            border-color: var(--ei-accent) !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: var(--ei-line);
            margin: 0.6rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: var(--ei-surface);
            border: 1px solid var(--ei-line);
            border-radius: 12px;
            margin-bottom: 0.45rem;
        }

        /* ---------- Workflow tabs = clear pill buttons ---------- */
        div[data-testid="stTabs"] {
            margin-top: 0.35rem;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.45rem !important;
            background: transparent !important;
            border-bottom: none !important;
            padding: 0.15rem 0 0.65rem 0;
            flex-wrap: wrap;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            background: var(--ei-control-bg) !important;
            border: 1.5px solid var(--ei-accent-border) !important;
            border-radius: 999px !important;
            color: #0f5c56 !important;
            font-weight: 650 !important;
            padding: 0.45rem 1.05rem !important;
            margin: 0 !important;
            box-shadow: 0 1px 0 rgba(15, 118, 110, 0.08);
            transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
            background: #c8f0e8 !important;
            border-color: var(--ei-accent) !important;
            transform: translateY(-1px);
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--ei-accent) !important;
            border-color: var(--ei-accent) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(15, 118, 110, 0.28);
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* ---------- Expanders = clickable feature buttons ---------- */
        .main [data-testid="stExpander"] {
            background: var(--ei-control-bg) !important;
            border: 1.5px solid var(--ei-accent-border) !important;
            border-radius: 12px !important;
            margin: 0.35rem 0 0.75rem 0 !important;
            box-shadow: 0 1px 0 rgba(15, 118, 110, 0.08);
        }
        .main [data-testid="stExpander"] details summary {
            cursor: pointer !important;
            font-weight: 650 !important;
            color: #0f5c56 !important;
            padding: 0.15rem 0;
        }
        .main [data-testid="stExpander"] details summary:hover {
            color: var(--ei-accent) !important;
        }
        .main [data-testid="stExpander"] details summary p,
        .main [data-testid="stExpander"] details summary span {
            color: #0f5c56 !important;
            font-weight: 650 !important;
        }

        /* ---------- Selects / inputs look interactive (same family) ---------- */
        .main [data-testid="stSelectbox"] > div > div,
        .main [data-baseweb="select"] > div {
            background: var(--ei-control-bg) !important;
            border: 1.5px solid var(--ei-accent-border) !important;
            border-radius: 10px !important;
            min-height: 2.55rem;
            box-shadow: 0 1px 0 rgba(15, 118, 110, 0.06);
        }
        .main [data-testid="stSelectbox"] > div > div:hover,
        .main [data-baseweb="select"] > div:hover {
            border-color: var(--ei-accent) !important;
            background: #c8f0e8 !important;
        }
        .main [data-testid="stTextInput"] input,
        .main [data-testid="stTextArea"] textarea,
        .main [data-testid="stNumberInput"] input {
            background: #ffffff !important;
            border: 1.5px solid var(--ei-accent-border) !important;
            border-radius: 10px !important;
            color: var(--ei-ink) !important;
        }
        .main [data-testid="stTextInput"] input:focus,
        .main [data-testid="stTextArea"] textarea:focus {
            border-color: var(--ei-accent) !important;
            box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14) !important;
        }

        /* Chip / secondary buttons in main (optional extras) */
        .main .stButton > button[kind="secondary"],
        .main .stButton > button[data-testid="baseButton-secondary"] {
            background: var(--ei-control-bg) !important;
            border: 1.5px solid var(--ei-accent-border) !important;
            color: #0f5c56 !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
        }
        .main .stButton > button[kind="secondary"]:hover,
        .main .stButton > button[data-testid="baseButton-secondary"]:hover {
            background: #c8f0e8 !important;
            border-color: var(--ei-accent) !important;
        }

        /* ---------- Generate CTA = eye-catching orange (not teal) ---------- */
        .main .stButton > button[kind="primary"],
        .main .stButton > button[data-testid="baseButton-primary"],
        .main div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(180deg, #f97316 0%, var(--ei-cta) 100%) !important;
            border: 1.5px solid var(--ei-cta) !important;
            color: #ffffff !important;
            font-weight: 750 !important;
            font-size: 1.02rem !important;
            border-radius: 12px !important;
            padding: 0.7rem 1.1rem !important;
            box-shadow: 0 6px 18px rgba(234, 88, 12, 0.32) !important;
            letter-spacing: 0.01em;
        }
        .main .stButton > button[kind="primary"]:hover,
        .main .stButton > button[data-testid="baseButton-primary"]:hover,
        .main div[data-testid="stFormSubmitButton"] button:hover {
            background: linear-gradient(180deg, #fb923c 0%, var(--ei-cta-hover) 100%) !important;
            border-color: var(--ei-cta-hover) !important;
            box-shadow: 0 8px 22px rgba(194, 65, 12, 0.38) !important;
        }

        .hist-chip {
            display: inline-block;
            font-size: 0.7rem;
            padding: 0.12rem 0.45rem;
            border-radius: 999px;
            background: var(--ei-accent-soft);
            color: #115e59;
            margin-right: 0.3rem;
            font-weight: 600;
        }
        .tip-card {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0.2rem 0;
            margin: 0;
            box-shadow: none;
        }
        .tip-card h4 {
            margin: 0 0 0.3rem 0;
            color: var(--ei-accent);
            font-size: 0.95rem;
        }
        .tip-card p, .tip-card li {
            color: var(--ei-muted);
            font-size: 0.88rem;
            margin: 0.15rem 0;
        }
        .conv-hero {
            background: linear-gradient(135deg, #0f766e 0%, #0e7490 55%, #0369a1 100%);
            color: #f8fafc;
            border-radius: 16px;
            padding: 1.05rem 1.2rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 10px 28px rgba(15, 118, 110, 0.18);
        }
        .conv-hero h2 {
            color: #ffffff !important;
            margin: 0 0 0.3rem 0;
            font-size: 1.25rem;
        }
        .conv-hero p { color: #ccfbf1; margin: 0.2rem 0; font-size: 0.85rem; }
        .conv-hero .hist-chip {
            background: rgba(255,255,255,0.18);
            color: #ecfeff;
        }
        .prompt-bubble {
            background: var(--ei-surface);
            border: 1px solid var(--ei-line);
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            margin: 0.4rem 0 0.9rem 0;
        }
        .user-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: var(--ei-accent-soft);
            color: #115e59;
            border-radius: 999px;
            padding: 0.15rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .control-hint {
            display: inline-block;
            margin: 0.15rem 0 0.55rem 0;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: var(--ei-control-bg);
            border: 1px solid var(--ei-accent-border);
            color: #0f5c56;
            font-size: 0.78rem;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _normalize_turns(raw: List[Dict]) -> List[Dict]:
    return [
        {
            "role": t.get("role", ""),
            "content": t.get("content", ""),
            "metadata": t.get("metadata") or {},
            "created_at": t.get("created_at") or "",
            "username": t.get("username") or st.session_state.username,
        }
        for t in (raw or [])
    ]


def get_conversations() -> List[Dict[str, Any]]:
    from services.chat_history_service import turns_to_conversations

    return turns_to_conversations(st.session_state.chat_history or [])


def render_beginner_guide(context: str = "general") -> None:
    """Short, friendly guidance — collapsed by default so the page stays calm."""
    tips = {
        "general": (
            "Quick start — tap to open",
            [
                "<b>1.</b> Write what you want in the prompt.",
                "<b>2.</b> Leave Brand & Language on Auto-detect unless you need a override.",
                "<b>3.</b> Extra tips are optional — skip if unsure.",
                "<b>4.</b> Open past work from <b>Your chats</b> on the left.",
            ],
        ),
        "social": (
            "Social tip — tap to open",
            [
                "Use <b>comment</b> for short replies. Add <code>10 words</code> in extras for tight length.",
                "Hashtags usually come from the brand kit automatically.",
            ],
        ),
        "email": (
            "Email tip — tap to open",
            [
                "Campaign type + brand is usually enough.",
                "Emails never get hashtags.",
            ],
        ),
    }
    title, lines = tips.get(context, tips["general"])
    items = "".join(f"<li>{line}</li>" for line in lines)
    with st.expander(title, expanded=False):
        st.markdown(
            f'<div class="tip-card">'
            f'<ul style="margin:0.2rem 0 0 1.1rem;padding:0">{items}</ul></div>',
            unsafe_allow_html=True,
        )


def render_extra_instructions_help(key_prefix: str) -> str:
    """
    Beginner-friendly optional extras: explain + quick-insert chips.
    Returns the text to merge into the form field via session state default.
    """
    with st.expander("Optional extras — tap for quick tips (skip if unsure)", expanded=False):
        st.caption(
            "Brand keywords, hashtags, and language are often auto-detected. "
            "Tap a chip below to fill Extra tips — or skip if unsure."
        )
        chips = [
            ("10 words", "10 words"),
            ("~150 words", "about 150 words"),
            ("Density 1.5%", "keyword density about 1.5%"),
            ("Hinglish tone", "write in natural Hinglish"),
            ("No CTA", "do not include a hard sales CTA"),
            ("Include stats", "include 1–2 attributed statistics"),
        ]
        cols = st.columns(3)
        form_key = f"{key_prefix}_instr"
        for idx, (label, text) in enumerate(chips):
            if cols[idx % 3].button(label, key=f"{key_prefix}_chip_{idx}", use_container_width=True):
                prev = st.session_state.get(form_key, "") or st.session_state.get(
                    f"{key_prefix}_instr_buf", ""
                )
                merged = (str(prev) + " " + text).strip() if prev else text
                st.session_state[f"{key_prefix}_instr_buf"] = merged
                st.session_state[form_key] = merged
                st.rerun()
        buf = st.session_state.get(f"{key_prefix}_instr_buf", "")
        if buf:
            st.info(f"Filled into Extra tips: **{buf}**")
            if st.button("Clear suggestions", key=f"{key_prefix}_chip_clear"):
                st.session_state[f"{key_prefix}_instr_buf"] = ""
                st.session_state[form_key] = ""
                st.rerun()
    return st.session_state.get(f"{key_prefix}_instr_buf", "")


def open_conversation(conv: Dict[str, Any]) -> None:
    st.session_state.active_conversation = conv
    st.session_state.main_view = "history"
    # Also mirror into results so tab views stay in sync when user switches back
    wf = str(conv.get("workflow") or "auto")
    if conv.get("result"):
        st.session_state.results[wf] = conv["result"]
    st.rerun()


def display_conversation_view(conv: Dict[str, Any]) -> None:
    """Full ChatGPT-style reopen: title, prompt, then complete generation output."""
    title = conv.get("title") or "Conversation"
    prompt = conv.get("prompt") or ""
    workflow = str(conv.get("workflow") or "auto")
    score = conv.get("score")
    status = conv.get("status") or "—"
    created = conv.get("created_at") or ""

    def _esc(s: Any) -> str:
        return (
            str(s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    score_chip = (
        f"<span class='hist-chip'>score {_esc(score)}</span>" if score is not None else ""
    )
    st.markdown(
        f"""
        <div class="conv-hero">
          <span class="hist-chip">{_esc(workflow)}</span>
          {score_chip}
          <span class="hist-chip">{_esc(status)}</span>
          <h2>{_esc(title)}</h2>
          <p>{_esc(created)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 4])
    if c1.button("← Back to create", key="hist_back_create", use_container_width=True):
        st.session_state.main_view = "create"
        st.session_state.active_conversation = None
        st.rerun()
    c2.caption("Everything from this run is below — content, review, metadata. No extra clicks.")

    st.markdown("#### Your prompt")
    st.markdown(
        f'<div class="prompt-bubble">{prompt.replace("<", "&lt;").replace(">", "&gt;")}</div>',
        unsafe_allow_html=True,
    )

    result = conv.get("result")
    if isinstance(result, dict) and result:
        display_wf = workflow if workflow in ("content", "email", "seo", "social", "auto") else "content"
        if display_wf == "auto":
            display_wf = str((result.get("metadata") or {}).get("workflow") or "content")
        display_result(result, display_wf if display_wf != "auto" else "content")
        return

    # Legacy history without stored result payload
    assistant_text = conv.get("assistant_text") or conv.get("preview") or ""
    if assistant_text:
        st.markdown("#### Generated output")
        st.markdown(assistant_text)
        if conv.get("hashtags"):
            st.markdown("**Hashtags:** " + " ".join(str(h) for h in conv["hashtags"]))
    else:
        st.info("This older history entry has no full snapshot. Generate again to save a complete view.")


# ==========================================================================
# Result display helpers
# ==========================================================================


def _escape_markdown_currency(text: str) -> str:
    """Prevent Streamlit from treating $...$ as LaTeX (jams spaces / italicizes)."""
    if not text:
        return text
    return text.replace("$", r"\$")


def _get_markdown(result: Dict) -> str:
    """Extract the Markdown draft from a workflow result."""
    final = result.get("final_output") or {}
    content = final.get("content") or {}
    if isinstance(content, dict):
        return content.get("markdown", "")
    return str(content)


def _score_color(score: int) -> str:
    if score >= 80:
        return "green"
    if score >= 70:
        return "orange"
    return "red"


def display_review_panel(review: Dict) -> None:
    score = review.get("score", 0)
    status = review.get("status", "—")
    dim_scores = review.get("dimension_scores") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Score", f"{score} / 100")
    col2.metric("Status", status)
    col3.metric("Revisions Used", st.session_state.get("_revision_count", 0))

    if dim_scores:
        st.markdown("#### Dimension Scores")
        for dim, val in dim_scores.items():
            label = dim.replace("_", " ").title()
            st.progress(int(val) / 100, text=f"{label}: {val}/100")

    feedback = review.get("feedback", [])
    issues = review.get("issues", [])

    if feedback:
        with st.expander("Positive feedback", expanded=False):
            for fb in feedback:
                st.markdown(f"✅ {fb}")

    if issues:
        with st.expander("Issues identified", expanded=False):
            for issue in issues:
                st.markdown(f"⚠️ {issue}")


def display_metadata_panel(metadata: Dict, result: Dict, workflow_type: str = "") -> None:
    seo = result.get("final_output", {}).get("seo") or {}
    final = result.get("final_output") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Word Count", metadata.get("word_count", "—"))
    read_time = metadata.get("read_time_min")
    if read_time is None:
        read_time = metadata.get("reading_time_minutes", "—")
    col2.metric("Read Time", f"{read_time} min")
    col3.metric("Language", metadata.get("language", "—"))

    font_name = final.get("font") or ""
    if font_name:
        st.markdown(f"**Brand Font:** {font_name}")

    if seo:
        st.markdown("#### SEO Fields")
        st.markdown(f"**Meta Title:** {seo.get('meta_title', '—')}")
        st.markdown(f"**Meta Description:** {seo.get('meta_description', '—')}")
        st.markdown(f"**Slug:** `{seo.get('slug', '—')}`")
        st.markdown(f"**Search Intent:** {seo.get('search_intent', '—')}")

        primary = seo.get("primary_keywords", [])
        if primary:
            st.markdown(f"**Primary Keywords:** {', '.join(primary[:5])}")

    # Hashtags for every content type except email
    if workflow_type != "email":
        hashtags = (
            final.get("hashtags")
            or (result.get("social_meta") or {}).get("hashtags")
            or []
        )
        if hashtags:
            st.markdown("#### Hashtags")
            st.markdown(" ".join(str(h) for h in hashtags))


def display_seo_analysis_panel(seo_analysis: Dict) -> None:
    st.markdown("#### Technical SEO Checklist")
    checklist = seo_analysis.get("technical_checklist", {})
    cols = st.columns(2)
    items = list(checklist.items())
    for i, (check, passed) in enumerate(items):
        icon = "✅" if passed else "❌"
        label = check.replace("_", " ").title()
        cols[i % 2].markdown(f"{icon} {label}")

    seo_score = seo_analysis.get("seo_score", 0)
    st.markdown(f"**Technical SEO Score:** {seo_score}/100")
    st.progress(seo_score / 100)

    density = seo_analysis.get("keyword_density", {})
    if density:
        st.markdown("#### Keyword Density")
        for kw, pct in list(density.items())[:10]:
            bar_val = min(pct / 3.0, 1.0)
            indicator = "✅" if 0.5 <= pct <= 2.5 else ("⚠️" if pct < 0.5 else "🔴")
            st.progress(bar_val, text=f"{indicator} `{kw}` — {pct}%")

    heading = seo_analysis.get("heading_audit", {})
    if heading:
        st.markdown("#### Heading Audit")
        col1, col2, col3 = st.columns(3)
        col1.metric("H1 Count", heading.get("h1_count", 0))
        col2.metric("H2 Count", heading.get("h2_count", 0))
        col3.metric("H3 Count", heading.get("h3_count", 0))
        kw_h1 = heading.get("primary_keyword_in_h1", False)
        kw_h2 = heading.get("primary_keyword_in_h2", False)
        st.markdown(
            f"Primary keyword in H1: {'✅' if kw_h1 else '❌'} | "
            f"Primary keyword in H2: {'✅' if kw_h2 else '❌'}"
        )


def display_email_meta_panel(email_meta: Dict) -> None:
    subject = email_meta.get("subject_line", "—")
    preview = email_meta.get("preview_text", "—")
    tokens = email_meta.get("personalization_tokens", [])
    campaign = email_meta.get("campaign_type", "—")

    st.markdown(f"**Campaign Type:** {campaign.title()}")
    st.markdown(f"**Subject Line:** {subject}")
    st.markdown(f"**Preview Text:** {preview}")
    if tokens:
        st.markdown(f"**Personalization Tokens:** {', '.join(f'`[{t}]`' for t in tokens)}")


def display_social_meta_panel(social_meta: Dict) -> None:
    platform = social_meta.get("platform", "—")
    hook = social_meta.get("engagement_hook", "—")
    hashtags = social_meta.get("hashtags", [])
    char_count = social_meta.get("character_count", 0)
    slide_count = social_meta.get("slide_count", 0)

    st.markdown(f"**Platform:** {platform.title()}")
    st.markdown(f"**Engagement Hook:** {hook}")

    col1, col2 = st.columns(2)
    col1.metric("Character Count", char_count)
    if slide_count:
        col2.metric("Slide Count", slide_count)

    if hashtags:
        st.markdown("**Hashtags:**")
        st.markdown(" ".join(hashtags))


def display_result(result: Dict, workflow_type: str) -> None:
    """Render the full result block for any workflow type."""
    if not result.get("ok"):
        errors = result.get("errors", ["Unknown error."])
        for err in errors:
            st.error(err)
        return

    st.success(
        f"Generated successfully · "
        f"Score: {result.get('review', {}).get('score', '—')}/100 · "
        f"Status: {result.get('review', {}).get('status', '—')}"
    )

    st.session_state["_revision_count"] = result.get("revision_count", 0)

    markdown = _get_markdown(result)

    # Determine which extra tab to show
    extra_tabs = {
        "content": [],
        "email": [("✉️ Email Meta", display_email_meta_panel, result.get("email_meta", {}))],
        "seo": [("🔍 SEO Analysis", display_seo_analysis_panel, result.get("seo_analysis", {}))],
        "social": [("📱 Social Meta", display_social_meta_panel, result.get("social_meta", {}))],
    }

    tab_labels = ["📝 Content", "📊 Review", "ℹ️ Metadata"]
    for label, _, _ in extra_tabs.get(workflow_type, []):
        tab_labels.append(label)

    tabs = st.tabs(tab_labels)

    # --- Content tab ---
    with tabs[0]:
        if markdown:
            font_stack = _font_stack_for_result(result)
            _inject_brand_font_css(font_stack)
            final = result.get("final_output") or {}
            if final.get("font"):
                st.caption(f"Brand font: {final.get('font')}")
            st.markdown(
                f'<div class="brand-content-preview" style="font-family:{font_stack}">',
                unsafe_allow_html=True,
            )
            st.markdown(_escape_markdown_currency(markdown))
            hashtags = (
                final.get("hashtags")
                or (result.get("social_meta") or {}).get("hashtags")
                or []
            )
            if hashtags and workflow_type != "email":
                st.divider()
                st.markdown("**Hashtags:** " + " ".join(str(h) for h in hashtags))
            st.divider()
            col1, col2 = st.columns([1, 4])
            col1.download_button(
                label="📥 Download Markdown",
                    data=markdown,
                    file_name=f"{workflow_type}.md",
                    mime="text/markdown",
                    key=f"download_markdown_{workflow_type}_{hash(markdown)}",
                )

            col2.code(markdown[:300] + "…" if len(markdown) > 300 else markdown, language="markdown")
        else:
            st.info("No content in final_output. Check the API logs.")

    # --- Review tab ---
    with tabs[1]:
        display_review_panel(result.get("review", {}))

    # --- Metadata tab ---
    with tabs[2]:
        display_metadata_panel(result.get("metadata", {}), result, workflow_type)

    # --- Extra workflow-specific tabs ---
    for i, (_, fn, data) in enumerate(extra_tabs.get(workflow_type, [])):
        with tabs[3 + i]:
            if data:
                fn(data)
            else:
                st.info("No data available.")

    # --- Raw JSON expander (debug) ---
    with st.expander("Raw JSON response", expanded=False):
        st.json(result)


# ==========================================================================
# Login / Sign up gate
# ==========================================================================

if not st.session_state.authenticated:
    from services.user_auth import login as auth_login
    from services.user_auth import signup as auth_signup

    _inject_app_chrome_css()
    st.title("Editorial Intelligence")
    st.caption("Sign in to create content and keep your chat history")

    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            login_user = st.text_input("Username", autocomplete="username", key="login_user")
            login_pass = st.text_input(
                "Password",
                type="password",
                autocomplete="current-password",
                key="login_pass",
            )
            login_submit = st.form_submit_button(
                "Log in",
                type="primary",
                use_container_width=True,
            )
        if login_submit:
            ok, msg = auth_login(login_user, login_pass)
            if ok:
                from services.chat_history_service import stable_session_id

                from services.user_auth import is_admin as auth_is_admin

                user_key = (login_user or msg or "").strip()
                st.session_state.authenticated = True
                st.session_state.username = user_key
                st.session_state.is_admin = auth_is_admin(user_key)
                # Stable per-user session so history + ConversationMemory persist
                st.session_state.session_id = stable_session_id(user_key)
                st.session_state.chat_history = []
                st.session_state.history_loaded = False
                st.session_state.results = {}
                st.session_state.active_conversation = None
                st.session_state.main_view = "create"
                st.session_state.pop("_team_activity", None)
                st.success("Logged in. Your chat history will be restored.")
                st.rerun()
            else:
                st.error(msg)
        st.caption(
            "New here? Open the **Sign up** tab to create an account. "
            "Team admin can also use the bootstrap credentials from APP_USERNAME / APP_PASSWORD."
        )

    with tab_signup:
        with st.form("signup_form"):
            su_user = st.text_input("Choose a username", key="signup_user")
            su_pass = st.text_input(
                "Choose a password",
                type="password",
                key="signup_pass",
            )
            su_pass2 = st.text_input(
                "Confirm password",
                type="password",
                key="signup_pass2",
            )
            signup_submit = st.form_submit_button(
                "Create account",
                type="primary",
                use_container_width=True,
            )
        if signup_submit:
            ok, msg = auth_signup(su_user, su_pass, su_pass2)
            if ok:
                st.success(msg)
                st.info("Switch to the **Log in** tab and sign in with your new account.")
            else:
                st.error(msg)

    st.stop()


# ==========================================================================
# Sidebar
# ==========================================================================

_inject_app_chrome_css()

# Keep admin flag in sync (e.g. after code update without re-login)
try:
    from services.user_auth import is_admin as _auth_is_admin

    st.session_state.is_admin = _auth_is_admin(st.session_state.username or "")
except Exception:
    st.session_state.is_admin = bool(st.session_state.get("is_admin"))


def _render_conversation_buttons(
    conversations: List[Dict[str, Any]],
    key_prefix: str,
    limit: int = 30,
) -> None:
    """Compact one-line chat titles (ChatGPT-style)."""
    active_id = (st.session_state.active_conversation or {}).get("id")
    for i, conv in enumerate((conversations or [])[:limit]):
        title = str(conv.get("title") or "Untitled")
        label = title if len(title) <= 42 else title[:41] + "…"
        if st.button(
            label,
            key=f"{key_prefix}_{i}_{conv.get('id')}",
            use_container_width=True,
            type="primary" if conv.get("id") == active_id else "secondary",
        ):
            open_conversation(conv)


with st.sidebar:
    st.markdown("**Editorial Intelligence**")
    who_label = st.session_state.username or "user"
    if st.session_state.is_admin:
        st.caption(f"@{who_label} · Admin")
    else:
        st.caption(f"@{who_label}")

    if st.button("New chat", key="nav_new_gen", use_container_width=True, type="primary"):
        st.session_state.main_view = "create"
        st.session_state.active_conversation = None
        st.rerun()

    if st.button("Log out", key="nav_logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.is_admin = False
        st.session_state.chat_history = []
        st.session_state.history_loaded = False
        st.session_state.results = {}
        st.session_state.active_conversation = None
        st.session_state.main_view = "create"
        st.session_state.pop("_team_activity", None)
        st.rerun()

    st.divider()

    # ---- Own chat history (every user) ----
    st.markdown("**Your chats**")
    if not st.session_state.history_loaded:
        remote = fetch_user_history(st.session_state.username)
        if not remote:
            remote = fetch_session_history(st.session_state.session_id)
        st.session_state.chat_history = _normalize_turns(remote)
        st.session_state.history_loaded = True

    if st.button("Refresh", key="nav_refresh_hist", use_container_width=True):
        remote = fetch_user_history(st.session_state.username)
        st.session_state.chat_history = _normalize_turns(remote)
        st.session_state.history_loaded = True
        if st.session_state.is_admin:
            st.session_state["_team_activity"] = fetch_team_activity(200)
        st.rerun()

    my_conversations = get_conversations()
    if not my_conversations:
        st.caption("No chats yet — generate to start.")
    else:
        _render_conversation_buttons(my_conversations, "my_hist", limit=35)

    # ---- Admin only: all users grouped as @username ----
    if st.session_state.is_admin:
        st.divider()
        st.markdown("**Admin · all users**")
        st.caption("Team histories — only visible to admin")
        if "_team_activity" not in st.session_state:
            st.session_state["_team_activity"] = fetch_team_activity(200)

        from services.chat_history_service import (
            group_conversations_by_user,
            turns_to_conversations as _ttc,
        )

        team_convs = _ttc(st.session_state.get("_team_activity") or [])
        by_user = group_conversations_by_user(team_convs)
        if not by_user:
            st.caption("No team history yet.")
        else:
            for username, convs in by_user.items():
                display = username if username.startswith("@") else f"@{username}"
                with st.expander(f"{display} · {len(convs)}", expanded=False):
                    _render_conversation_buttons(
                        convs,
                        key_prefix=f"admin_{username}",
                        limit=25,
                    )

    st.divider()
    if not st.session_state.brands:
        st.session_state.brands = fetch_brands()

    with st.expander("Brands", expanded=False):
        if st.session_state.brands:
            for brand in st.session_state.brands:
                st.caption(f"**{brand['display_name']}**")
        else:
            st.caption("Brands unavailable")

    st.caption("Chats stay on your account.")


# ==========================================================================
# Brand selector — shared across all tabs
# ==========================================================================

brand_names = ["Auto-detect"] + [b["display_name"] for b in st.session_state.brands]
if not st.session_state.brands:
    brand_names = ["Auto-detect", "GTIB", "Kinvo", "MPM", "Futuristix", "GCB"]

# ==========================================================================
# Main canvas — history reopen OR create workflows
# ==========================================================================

if st.session_state.main_view == "history" and st.session_state.active_conversation:
    st.title(ist_greeting(st.session_state.username))
    display_conversation_view(st.session_state.active_conversation)
    st.stop()

st.title(ist_greeting(st.session_state.username))
st.caption("Describe what you need — we handle the rest. Your chats stay on the left.")
render_beginner_guide("general")

# ==========================================================================
# Workflow tabs
# ==========================================================================

tab_auto, tab_content, tab_email, tab_seo, tab_social = st.tabs(
    ["Auto", "Content", "Email", "SEO", "Social"]
)


# ---------------------------------------------------------------------------
# TAB 0 — Auto
# ---------------------------------------------------------------------------

with tab_auto:
    st.subheader("Auto Generate")
    st.caption(
        "Just describe what you need. We detect content vs email vs social vs SEO for you."
    )
    a_chip = render_extra_instructions_help("a")

    with st.form("auto_form"):
        a_user_input = st.text_area(
            "What do you want to create? *",
            placeholder="Example: Write a Hinglish LinkedIn post on AI automation for SMBs",
            height=120,
        )

        col1, col2 = st.columns(2)

        a_brand = col1.selectbox(
            "Brand (Auto-detect is fine)",
            brand_names,
            key="a_brand",
            help="Leave Auto-detect unless you must force a brand.",
        )

        a_language = col2.selectbox(
            "Language (Auto-detect is fine)",
            LANGUAGE_OPTIONS,
            key="a_lang",
            help="Auto follows your prompt (incl. Hinglish). Manual choice always wins.",
        )

        a_instructions = st.text_input(
            "Extra tips (optional — skip if unsure)",
            key="a_instr",
            placeholder="Only if needed: e.g. 10 words, density 1.5%, #AI #Agents",
            help="Brand keywords and hashtags are often applied automatically.",
        )

        a_max_rev = st.slider(
            "Max Revisions",
            1,
            5,
            1,
            key="a_rev",
            help="How many rewrite attempts if review asks for fixes. 1 is usually enough.",
        )

        a_submitted = st.form_submit_button(
            "Generate content",
            use_container_width=True,
            type="primary",
        )

    if a_submitted:
        if not a_user_input.strip():
            st.error("Please enter a prompt.")
        else:
            brand_val = None if a_brand == "Auto-detect" else a_brand
            extras = (a_instructions or a_chip or "").strip()

            payload = {
                "user_input": a_user_input,
                "brand": brand_val,
                "language": a_language,
                "additional_instructions": extras,
                "session_id": st.session_state.session_id,
                "max_revisions": a_max_rev,
            }

            with st.spinner(
                "Auto-detecting workflow and generating content… this can take a few minutes."
            ):
                result = call_api(
                    "generate",
                    payload,
                )

            st.session_state.results["auto"] = result
            record_generation(a_user_input, result, "auto")
            st.session_state.history_loaded = True

    if "auto" in st.session_state.results:
        workflow = (
            st.session_state.results["auto"]
            .get("metadata", {})
            .get("workflow")
        )

        if workflow:
            st.success(
                f"Detected workflow: {workflow.title()}"
            )

        st.divider()
        display_result(
            st.session_state.results["auto"],
            "content",
        )


# ---------------------------------------------------------------------------
# TAB 1 — Content (Article / Blog)
# ---------------------------------------------------------------------------

with tab_content:
    st.subheader("Long-Form Content")
    st.caption("Articles and blogs. Defaults handle length and SEO kit — only override when needed.")
    c_chip = render_extra_instructions_help("c")

    with st.form("content_form"):
        c_user_input = st.text_area(
            "Topic / Brief *",
            placeholder="How AI agents are transforming SMB operations",
            height=100,
        )
        col1, col2 = st.columns(2)
        c_content_type = col1.selectbox("Content Type", ["article", "blog"])
        c_brand = col2.selectbox("Brand (Auto-detect is fine)", brand_names, key="c_brand")
        col3, col4 = st.columns(2)
        c_objective = col3.selectbox("Objective", ["seo", "authority", "engagement", "leads"])
        c_language = col4.selectbox(
            "Language (Auto-detect is fine)",
            LANGUAGE_OPTIONS,
            key="c_lang",
            help="Auto-detect follows the prompt language. Manual choice always wins.",
        )
        c_instructions = st.text_input(
            "Extra tips (optional — skip if unsure)",
            key="c_instr",
            placeholder="e.g. density 1–1.5%, include state-wise cases…",
            help="Hashtags and brand keywords usually come from the brand kit automatically.",
        )
        c_max_rev = st.slider("Max Revisions", 1, 5, 1, key="c_rev")
        c_submitted = st.form_submit_button("Generate content", use_container_width=True, type="primary")

    if c_submitted:
        if not c_user_input.strip():
            st.error("Please enter a topic or brief.")
        else:
            brand_val = None if c_brand == "Auto-detect" else c_brand
            payload = {
                "user_input": c_user_input,
                "content_type": c_content_type,
                "brand": brand_val,
                "objective": c_objective,
                "language": c_language,
                "additional_instructions": (c_instructions or c_chip or "").strip(),
                "session_id": st.session_state.session_id,
                "max_revisions": c_max_rev,
            }
            with st.spinner("Running 5-agent pipeline… this can take a few minutes."):
                result = call_api("generate/content", payload)
            st.session_state.results["content"] = result
            record_generation(c_user_input, result, "content")
            st.session_state.history_loaded = True

    if "content" in st.session_state.results:
        st.divider()
        display_result(st.session_state.results["content"], "content")


# ---------------------------------------------------------------------------
# TAB 2 — Email
# ---------------------------------------------------------------------------

with tab_email:
    st.subheader("Email Campaign")
    st.caption("Newsletter, nurture, promo, or transactional — hashtags are never added.")
    render_beginner_guide("email")
    e_chip = render_extra_instructions_help("e")

    with st.form("email_form"):
        e_user_input = st.text_area(
            "Topic / Brief *",
            placeholder="Announce our new AI audit service to founders",
            height=100,
        )
        col1, col2 = st.columns(2)
        e_brand = col1.selectbox("Brand (Auto-detect is fine)", brand_names, key="e_brand")
        e_campaign = col2.selectbox(
            "Campaign Type",
            ["newsletter", "nurture", "promotional", "transactional"],
        )
        col3, col4 = st.columns(2)
        e_objective = col3.selectbox("Objective", ["leads", "engagement"], key="e_obj")
        e_language = col4.selectbox(
            "Language (Auto-detect is fine)",
            LANGUAGE_OPTIONS,
            key="e_lang",
            help="Auto-detect follows the prompt language. Manual choice always wins.",
        )
        e_instructions = st.text_input(
            "Extra tips (optional — skip if unsure)",
            key="e_instr",
            placeholder="e.g. subject under 50 chars, mention offer code…",
        )
        e_max_rev = st.slider("Max Revisions", 1, 4, 1, key="e_rev")
        e_submitted = st.form_submit_button("Generate email", use_container_width=True, type="primary")

    if e_submitted:
        if not e_user_input.strip():
            st.error("Please enter a topic or brief.")
        else:
            brand_val = None if e_brand == "Auto-detect" else e_brand
            payload = {
                "user_input": e_user_input,
                "brand": brand_val,
                "campaign_type": e_campaign,
                "objective": e_objective,
                "language": e_language,
                "additional_instructions": (e_instructions or e_chip or "").strip(),
                "session_id": st.session_state.session_id,
                "max_revisions": e_max_rev,
            }
            with st.spinner("Generating email… this can take 1–3 minutes."):
                result = call_api("generate/email", payload)
            st.session_state.results["email"] = result
            record_generation(e_user_input, result, "email")
            st.session_state.history_loaded = True

    if "email" in st.session_state.results:
        st.divider()
        display_result(st.session_state.results["email"], "email")


# ---------------------------------------------------------------------------
# TAB 3 — SEO
# ---------------------------------------------------------------------------

with tab_seo:
    st.subheader("SEO-Optimised Content")
    st.caption("Keywords and checklist come from the brand SEO kit — extras are optional.")
    s_chip = render_extra_instructions_help("s")

    with st.form("seo_form"):
        s_user_input = st.text_area(
            "Search Query / Topic *",
            placeholder="AI agents for small business automation",
            height=100,
        )
        col1, col2 = st.columns(2)
        s_content_type = col1.selectbox("Content Type", ["article", "blog"], key="s_ct")
        s_brand = col2.selectbox("Brand (Auto-detect is fine)", brand_names, key="s_brand")
        col3, col4 = st.columns(2)
        s_language = col3.selectbox(
            "Language (Auto-detect is fine)",
            LANGUAGE_OPTIONS,
            key="s_lang",
            help="Auto-detect follows the prompt language. Manual choice always wins.",
        )
        s_max_rev = col4.slider("Max Revisions", 1, 5, 1, key="s_rev")
        s_instructions = st.text_input(
            "Extra tips (optional — skip if unsure)",
            key="s_instr",
            placeholder="e.g. primary density ~1.5%, include slug-friendly H2s…",
        )
        s_submitted = st.form_submit_button("Generate SEO content", use_container_width=True, type="primary")

    if s_submitted:
        if not s_user_input.strip():
            st.error("Please enter a search query or topic.")
        else:
            brand_val = None if s_brand == "Auto-detect" else s_brand
            payload = {
                "user_input": s_user_input,
                "content_type": s_content_type,
                "brand": brand_val,
                "language": s_language,
                "additional_instructions": (s_instructions or s_chip or "").strip(),
                "session_id": st.session_state.session_id,
                "max_revisions": s_max_rev,
            }
            with st.spinner("Running SEO pipeline… this can take a few minutes."):
                result = call_api("generate/seo", payload)
            st.session_state.results["seo"] = result
            record_generation(s_user_input, result, "seo")
            st.session_state.history_loaded = True

    if "seo" in st.session_state.results:
        st.divider()
        display_result(st.session_state.results["seo"], "seo")


# ---------------------------------------------------------------------------
# TAB 4 — Social
# ---------------------------------------------------------------------------

with tab_social:
    st.subheader("Social Media Content")
    st.caption(
        "LinkedIn, X, Instagram, Facebook, Reddit, carousel, or comment replies. "
        "Comments stay short and never get hashtags."
    )
    render_beginner_guide("social")
    so_chip = render_extra_instructions_help("so")

    with st.form("social_form"):
        so_user_input = st.text_area(
            "Topic / Brief *",
            placeholder="Someone replied: your article is very good",
            height=100,
        )
        col1, col2 = st.columns(2)
        so_platform = col1.selectbox(
            "Platform",
            [
                "linkedin",
                "x",
                "instagram",
                "facebook",
                "reddit",
                "comment",
                "carousel",
            ],
            help="Use comment for thank-you / reply style messages.",
        )
        so_brand = col2.selectbox("Brand (Auto-detect is fine)", brand_names, key="so_brand")
        col3, col4 = st.columns(2)
        so_objective = col3.selectbox("Objective", ["engagement", "authority", "leads"], key="so_obj")
        so_language = col4.selectbox(
            "Language (Auto-detect is fine)",
            LANGUAGE_OPTIONS,
            key="so_lang",
            help="Auto-detect follows the prompt language. Manual choice always wins.",
        )
        so_instructions = st.text_input(
            "Extra tips (optional — skip if unsure)",
            key="so_instr",
            placeholder="e.g. 10 words, or add hashtags #KinvoCare…",
            help="For comments, try a chip like “10 words”. Hashtags are skipped automatically.",
        )
        so_max_rev = st.slider("Max Revisions", 1, 4, 1, key="so_rev")
        so_submitted = st.form_submit_button("Generate social post", use_container_width=True, type="primary")

    if so_submitted:
        if not so_user_input.strip():
            st.error("Please enter a topic or brief.")
        else:
            brand_val = None if so_brand == "Auto-detect" else so_brand
            payload = {
                "user_input": so_user_input,
                "platform": so_platform,
                "brand": brand_val,
                "objective": so_objective,
                "language": so_language,
                "additional_instructions": (so_instructions or so_chip or "").strip(),
                "session_id": st.session_state.session_id,
                "max_revisions": so_max_rev,
            }
            with st.spinner("Generating social content… this can take 1–3 minutes."):
                result = call_api("generate/social", payload)
            st.session_state.results["social"] = result
            record_generation(so_user_input, result, "social")
            st.session_state.history_loaded = True

    if "social" in st.session_state.results:
        st.divider()
        display_result(st.session_state.results["social"], "social")
