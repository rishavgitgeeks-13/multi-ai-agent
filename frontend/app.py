"""
Streamlit Frontend — Editorial Intelligence System
==================================================

Run:
    streamlit run frontend/app.py

The app calls the FastAPI backend at http://localhost:8000 (configurable
in the sidebar). All four workflow types are available as tabs.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

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
#API_BASE_URL = "http://54.218.34.106:9000"
API_BASE_URL = "http://localhost:8000"

# Always point at the deployed API (do not let an old empty session value stick).
st.session_state.api_url = API_BASE_URL
if "brands" not in st.session_state:
    st.session_state.brands = []


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


def display_metadata_panel(metadata: Dict, result: Dict) -> None:
    seo = result.get("final_output", {}).get("seo") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Word Count", metadata.get("word_count", "—"))
    read_time = metadata.get("read_time_min")
    if read_time is None:
        read_time = metadata.get("reading_time_minutes", "—")
    col2.metric("Read Time", f"{read_time} min")
    col3.metric("Language", metadata.get("language", "—"))

    if seo:
        st.markdown("#### SEO Fields")
        st.markdown(f"**Meta Title:** {seo.get('meta_title', '—')}")
        st.markdown(f"**Meta Description:** {seo.get('meta_description', '—')}")
        st.markdown(f"**Slug:** `{seo.get('slug', '—')}`")
        st.markdown(f"**Search Intent:** {seo.get('search_intent', '—')}")

        primary = seo.get("primary_keywords", [])
        if primary:
            st.markdown(f"**Primary Keywords:** {', '.join(primary[:5])}")


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
            st.markdown(_escape_markdown_currency(markdown))
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
        display_metadata_panel(result.get("metadata", {}), result)

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
# Sidebar
# ==========================================================================

with st.sidebar:
    st.title("⚙️ Settings")

    # API health indicator
    if st.button("Check API"):
        with st.spinner("Checking…"):
            alive = check_api_health()
        if alive:
            st.success("API is online")
        else:
            st.error("API is offline")

    st.divider()

    # Session management
    st.subheader("Session")
    st.caption(f"ID: `{st.session_state.session_id[:20]}…`")
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.results = {}
        st.rerun()

    st.divider()

    # Load brands
    if not st.session_state.brands:
        with st.spinner("Loading brands…"):
            st.session_state.brands = fetch_brands()

    if st.session_state.brands:
        st.subheader("Configured Brands")
        for brand in st.session_state.brands:
            with st.expander(brand["display_name"]):
                st.caption(f"Tone: {brand['tone']}")
                st.caption(f"CTA: {brand['cta']}")
    else:
        st.caption("Brands unavailable (API offline or brands.yaml unloaded)")

    st.divider()
    st.caption("Multi-Agent Editorial Intelligence System v1.0")


# ==========================================================================
# Brand selector — shared across all tabs
# ==========================================================================

brand_names = ["Auto-detect"] + [b["display_name"] for b in st.session_state.brands]
if not st.session_state.brands:
    brand_names = ["Auto-detect", "GTIB", "Kinvo", "MPM", "Futuristix", "GCB"]

# ==========================================================================
# Main header
# ==========================================================================

st.title("📝 Editorial Intelligence System")
st.caption("Multi-agent content generation · Powered by OpenAI & LangGraph")
st.divider()

# ==========================================================================
# Workflow tabs
# ==========================================================================

tab_auto, tab_content, tab_email, tab_seo, tab_social = st.tabs(
    ["🚀 Auto", "📄 Content", "✉️ Email", "🔍 SEO", "📱 Social"]
)


# ---------------------------------------------------------------------------
# TAB 0 — Auto
# ---------------------------------------------------------------------------

with tab_auto:
    st.subheader("Auto Generate")
    st.caption(
        "Automatically detects whether the request is for content, email, social, or SEO."
    )

    with st.form("auto_form"):
        a_user_input = st.text_area(
            "Prompt *",
            placeholder="Write a blog on AI workflow automation, create a LinkedIn post, generate an email campaign, or perform SEO analysis.",
            height=120,
        )

        col1, col2 = st.columns(2)

        a_brand = col1.selectbox(
            "Brand",
            brand_names,
            key="a_brand",
        )

        a_language = col2.selectbox(
            "Language",
            ["English", "Hindi"],
            key="a_lang",
        )

        a_instructions = st.text_input(
            "Additional Instructions (optional)",
            key="a_instr",
        )

        a_max_rev = st.slider(
            "Max Revisions",
            1,
            5,
            1,
            key="a_rev",
        )

        a_submitted = st.form_submit_button(
            "Generate",
            use_container_width=True,
            type="primary",
        )

    if a_submitted:
        if not a_user_input.strip():
            st.error("Please enter a prompt.")
        else:
            brand_val = None if a_brand == "Auto-detect" else a_brand

            payload = {
                "user_input": a_user_input,
                "brand": brand_val,
                "language": a_language,
                "additional_instructions": a_instructions,
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
    st.caption("Generate article (~2200 words) or blog (~1800 words) content.")

    with st.form("content_form"):
        c_user_input = st.text_area(
            "Topic / Brief *",
            placeholder="How AI agents are transforming SMB operations",
            height=100,
        )
        col1, col2 = st.columns(2)
        c_content_type = col1.selectbox("Content Type", ["article", "blog"])
        c_brand = col2.selectbox("Brand", brand_names, key="c_brand")
        col3, col4 = st.columns(2)
        c_objective = col3.selectbox("Objective", ["seo", "authority", "engagement", "leads"])
        c_language = col4.selectbox("Language", ["English", "Hindi"], key="c_lang")
        c_instructions = st.text_input("Additional Instructions (optional)", placeholder="Focus on ROI examples…")
        c_max_rev = st.slider("Max Revisions", 1, 5, 1, key="c_rev")
        c_submitted = st.form_submit_button("Generate Content", use_container_width=True, type="primary")

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
                "additional_instructions": c_instructions,
                "session_id": st.session_state.session_id,
                "max_revisions": c_max_rev,
            }
            with st.spinner("Running 5-agent pipeline… this can take a few minutes."):
                result = call_api("generate/content", payload)
            st.session_state.results["content"] = result

    if "content" in st.session_state.results:
        st.divider()
        display_result(st.session_state.results["content"], "content")


# ---------------------------------------------------------------------------
# TAB 2 — Email
# ---------------------------------------------------------------------------

with tab_email:
    st.subheader("Email Campaign")
    st.caption("Generate newsletter, nurture, promotional, or transactional emails (~400 words).")

    with st.form("email_form"):
        e_user_input = st.text_area(
            "Topic / Brief *",
            placeholder="Announce our new AI audit service to founders",
            height=100,
        )
        col1, col2 = st.columns(2)
        e_brand = col1.selectbox("Brand", brand_names, key="e_brand")
        e_campaign = col2.selectbox(
            "Campaign Type",
            ["newsletter", "nurture", "promotional", "transactional"],
        )
        col3, col4 = st.columns(2)
        e_objective = col3.selectbox("Objective", ["leads", "engagement"], key="e_obj")
        e_language = col4.selectbox("Language", ["English", "Hindi"], key="e_lang")
        e_instructions = st.text_input("Additional Instructions (optional)", key="e_instr")
        e_max_rev = st.slider("Max Revisions", 1, 4, 1, key="e_rev")
        e_submitted = st.form_submit_button("Generate Email", use_container_width=True, type="primary")

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
                "additional_instructions": e_instructions,
                "session_id": st.session_state.session_id,
                "max_revisions": e_max_rev,
            }
            with st.spinner("Generating email… this can take 1–3 minutes."):
                result = call_api("generate/email", payload)
            st.session_state.results["email"] = result

    if "email" in st.session_state.results:
        st.divider()
        display_result(st.session_state.results["email"], "email")


# ---------------------------------------------------------------------------
# TAB 3 — SEO
# ---------------------------------------------------------------------------

with tab_seo:
    st.subheader("SEO-Optimised Content")
    st.caption("Full SEO pipeline with keyword density analysis, heading audit, and technical SEO checklist.")

    with st.form("seo_form"):
        s_user_input = st.text_area(
            "Search Query / Topic *",
            placeholder="AI agents for small business automation",
            height=100,
        )
        col1, col2 = st.columns(2)
        s_content_type = col1.selectbox("Content Type", ["article", "blog"], key="s_ct")
        s_brand = col2.selectbox("Brand", brand_names, key="s_brand")
        col3, col4 = st.columns(2)
        s_language = col3.selectbox("Language", ["English", "Hindi"], key="s_lang")
        s_max_rev = col4.slider("Max Revisions", 1, 5, 1, key="s_rev")
        s_instructions = st.text_input("Additional Instructions (optional)", key="s_instr")
        s_submitted = st.form_submit_button("Generate SEO Content", use_container_width=True, type="primary")

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
                "additional_instructions": s_instructions,
                "session_id": st.session_state.session_id,
                "max_revisions": s_max_rev,
            }
            with st.spinner("Running SEO pipeline… this can take a few minutes."):
                result = call_api("generate/seo", payload)
            st.session_state.results["seo"] = result

    if "seo" in st.session_state.results:
        st.divider()
        display_result(st.session_state.results["seo"], "seo")


# ---------------------------------------------------------------------------
# TAB 4 — Social
# ---------------------------------------------------------------------------

with tab_social:
    st.subheader("Social Media Content")
    st.caption("LinkedIn posts, carousels, and X threads.")

    with st.form("social_form"):
        so_user_input = st.text_area(
            "Topic / Brief *",
            placeholder="Why AI agents are the next competitive advantage for SMBs",
            height=100,
        )
        col1, col2 = st.columns(2)
        so_platform = col1.selectbox("Platform", ["linkedin", "carousel", "x"])
        so_brand = col2.selectbox("Brand", brand_names, key="so_brand")
        col3, col4 = st.columns(2)
        so_objective = col3.selectbox("Objective", ["engagement", "authority", "leads"], key="so_obj")
        so_language = col4.selectbox("Language", ["English", "Hindi"], key="so_lang")
        so_instructions = st.text_input("Additional Instructions (optional)", key="so_instr")
        so_max_rev = st.slider("Max Revisions", 1, 4, 1, key="so_rev")
        so_submitted = st.form_submit_button("Generate Social Post", use_container_width=True, type="primary")

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
                "additional_instructions": so_instructions,
                "session_id": st.session_state.session_id,
                "max_revisions": so_max_rev,
            }
            with st.spinner("Generating social content… this can take 1–3 minutes."):
                result = call_api("generate/social", payload)
            st.session_state.results["social"] = result

    if "social" in st.session_state.results:
        st.divider()
        display_result(st.session_state.results["social"], "social")
