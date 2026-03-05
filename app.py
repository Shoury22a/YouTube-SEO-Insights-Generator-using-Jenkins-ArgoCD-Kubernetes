"""
YouTube SEO Insights Generator — Streamlit Frontend
Provides a rich UI for generating AI-powered YouTube SEO metadata.
"""

import os
import sys
import streamlit as st
from src.logger import get_logger
from src.extractor import extract_video_metadata, compute_niche_saturation, compute_contrarian_score
from src.ai_model import generate_seo_metadata, MAX_TRANSCRIPT_CHARS
from src.exception import APIException, ValidationException, SEOAppException

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TubeRank AI — YouTube SEO Insights",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom Premium CSS (Glassmorphism & Vibrant Aesthetics)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Modern Typography */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

        html, body, [class*="css"] { 
            font-family: 'Outfit', sans-serif; 
            color: #f0f0f0;
        }

        /* Animated Dark Background */
        .stApp {
            background: #0a0a12;
            background-image: 
                radial-gradient(ellipse at 10% 20%, rgba(255, 0, 80, 0.07) 0%, transparent 50%),
                radial-gradient(ellipse at 90% 80%, rgba(120, 0, 255, 0.07) 0%, transparent 50%);
            background-attachment: fixed;
        }

        /* Glassmorphism Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(15, 15, 26, 0.8) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Hero Banner */
        .hero-banner {
            position: relative;
            background: linear-gradient(135deg, #1a0a2e 0%, #0f0f1a 100%);
            border-radius: 24px;
            padding: 50px 40px;
            margin-bottom: 36px;
            border: 1px solid rgba(255, 0, 80, 0.2);
            overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(204, 0, 255, 0.15) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero-banner::after {
            content: '';
            position: absolute;
            bottom: -50%;
            left: -10%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(255, 0, 80, 0.12) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero-brand {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 16px;
        }
        .hero-brand img { 
            width: 64px; 
            height: 64px; 
            border-radius: 16px;
            box-shadow: 0 0 30px rgba(255, 0, 80, 0.4);
        }
        .hero-badge {
            display: inline-block;
            background: linear-gradient(90deg, rgba(255,0,80,0.2), rgba(204,0,255,0.2));
            border: 1px solid rgba(255, 0, 80, 0.4);
            border-radius: 30px;
            padding: 4px 14px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #ff6b9d;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .hero-banner h1 { 
            color: #fff; 
            font-size: 3.5rem; 
            margin: 0; 
            font-weight: 800; 
            letter-spacing: -2px;
            line-height: 1.1;
            background: linear-gradient(135deg, #ffffff 0%, #d4d4d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-banner .hero-sub { 
            color: rgba(255,255,255,0.55); 
            margin-top: 12px; 
            font-size: 1.15rem; 
            font-weight: 300;
            max-width: 600px;
        }
        .hero-stats {
            display: flex;
            gap: 24px;
            margin-top: 28px;
            flex-wrap: wrap;
        }
        .stat-chip {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 0.85rem;
        }
        .stat-chip span { 
            display: block; 
            font-size: 1.3rem; 
            font-weight: 700;
            background: linear-gradient(90deg, #ff0050, #cc00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Glass Cards */
        .stExpander {
            background: rgba(255, 255, 255, 0.03) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.07) !important;
            margin-bottom: 12px;
            transition: all 0.3s ease;
        }
        .stExpander:hover {
            border-color: rgba(255, 0, 80, 0.3) !important;
            box-shadow: 0 0 20px rgba(255, 0, 80, 0.08);
        }

        /* Glowing Generate Button */
        .stButton > button {
            background: linear-gradient(90deg, #ff0050 0%, #cc00ff 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            padding: 18px 0 !important;
            width: 100% !important;
            letter-spacing: 0.5px;
            box-shadow: 0 8px 30px rgba(255, 0, 80, 0.35) !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            box-shadow: 0 12px 40px rgba(255, 0, 80, 0.55) !important;
            transform: translateY(-2px);
        }
        .stButton > button:active { transform: scale(0.98); }

        /* Tag Pills */
        .tag-pill {
            display: inline-block;
            background: rgba(255, 0, 80, 0.08);
            border: 1px solid rgba(255, 0, 80, 0.25);
            color: #ff6b9d;
            border-radius: 30px;
            padding: 5px 16px;
            margin: 4px;
            font-size: 0.83rem;
            font-weight: 600;
            transition: all 0.2s ease;
            cursor: default;
        }
        .tag-pill:hover {
            background: #ff0050;
            color: #fff;
            border-color: #ff0050;
            box-shadow: 0 0 12px rgba(255, 0, 80, 0.4);
        }

        /* Section Divider */
        hr { border-color: rgba(255,255,255,0.06) !important; }

        /* Input Fields */
        .stTextArea textarea, .stTextInput input {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 12px !important;
            color: #fff !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: rgba(255, 0, 80, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(255, 0, 80, 0.1) !important;
        }

        /* Sidebar label */
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 0 20px;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            margin-bottom: 20px;
        }
        .sidebar-brand-name {
            font-weight: 700;
            font-size: 1.1rem;
            background: linear-gradient(90deg, #ff0050, #cc00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Mobile Responsive */
        @media (max-width: 768px) {
            .hero-banner h1 { font-size: 2.4rem; }
            .hero-stats { gap: 12px; }
            [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hero Banner — TubeRank AI
# ─────────────────────────────────────────────────────────────────────────────
import base64

def _img_to_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

logo_b64 = _img_to_b64("assets/logo.png")
logo_html = f'<img src="data:image/png;base64,{logo_b64}" />' if logo_b64 else "🎬"

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-brand">
            {logo_html}
            <div>
                <div class="hero-badge">✦ Powered by Gemini 2.0 Flash</div>
                <h1>TubeRank AI</h1>
            </div>
        </div>
        <p class="hero-sub">
            Generate viral titles, SEO tags, timestamps, social posts &amp; thumbnail ideas —
            in seconds. Built for creators who want to rank.
        </p>
        <div class="hero-stats">
            <div class="stat-chip"><span>10+</span>Languages</div>
            <div class="stat-chip"><span>AI</span>Competitor Intel</div>
            <div class="stat-chip"><span>∞</span>Hinglish Support</div>
            <div class="stat-chip"><span>6</span>Output Sections</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration Inputs
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Video Settings")

    content_type = st.radio(
        "Content Type",
        options=["Long-Form Video", "YouTube Short"],
        index=0,
        horizontal=True,
        help="Shorts get tighter titles (≤45 chars) and no timestamps.",
    )

    output_language = st.selectbox(
        "Output Language",
        options=[
            "English", "Hinglish", "Hindi", "Spanish", "French",
            "German", "Japanese", "Korean", "Portuguese", "Arabic",
        ],
        index=0,
        help="The AI will translate all outputs into this language.",
    )

    st.markdown("---")
    st.markdown("## 🔗 Competitor Reference (Optional)")
    competitor_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="We'll scrape metadata to give the AI competitive context.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main — Content Inputs
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("### 📝 Video Details")

    topic = st.text_input(
        "Core Topic ✱",
        placeholder="e.g. How to build a passive income stream with AI tools in 2024",
        help="Be specific. Minimum 2 words required to enable generation.",
        key="topic",
    )
    topic_word_count = len(topic.strip().split()) if topic.strip() else 0
    st.caption(f"{'✅' if topic_word_count >= 2 else '⚠️'} {topic_word_count} word(s) — minimum 2 required")

    audience = st.text_input(
        "Target Audience ✱",
        placeholder="e.g. Beginner entrepreneurs aged 20-35 interested in side hustles",
        key="audience",
    )

with col_right:
    st.markdown("### 📋 Chapter Notes")
    chapter_notes = st.text_area(
        "Rough Chapter Breakdown (for timestamps)",
        placeholder="0:00 Intro\n1:30 What is passive income?\n4:00 AI tool #1 — Notion AI\n...",
        height=158,
        help="Only for long-form. Leave blank to skip timestamp generation.",
        key="chapter_notes",
    )

# ── Transcript / Visual Description ──────────────────────────────────────────
st.markdown("### 🎙️ Transcript or Script")

use_visual_desc = st.checkbox(
    "This is a silent / gameplay / no-speech video — use visual description instead",
    key="use_visual_desc",
)

if use_visual_desc:
    visual_description = st.text_area(
        "Visual Description",
        placeholder=(
            "Describe what happens in the video. e.g.: "
            "'Timelapse of a city skyline from dawn to dusk, "
            "featuring drone shots of busy streets and skyscrapers...'"
        ),
        height=160,
        key="visual_description",
    )
    transcript = ""
else:
    transcript = st.text_area(
        "Transcript / Script (Optional)",
        placeholder=(
            "Paste your full video transcript or script here. "
            "Transcripts over 5,000 words will be summarised automatically. "
            f"Hard limit: {MAX_TRANSCRIPT_CHARS:,} characters."
        ),
        height=200,
        max_chars=MAX_TRANSCRIPT_CHARS,
        key="transcript",
    )
    visual_description = ""

    if transcript:
        char_count = len(transcript)
        pct = char_count / MAX_TRANSCRIPT_CHARS * 100
        st.caption(
            f"{'🟡' if pct > 70 else '🟢'} {char_count:,} / {MAX_TRANSCRIPT_CHARS:,} characters"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Generate Button (disabled until topic has ≥5 words)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")

can_generate = topic_word_count >= 2 and bool(audience.strip())
if not can_generate:
    st.info(
        "💡 Fill in a **Core Topic** (≥2 words) and **Target Audience** to enable generation.",
        icon="ℹ️",
    )

generate_clicked = st.button(
    "✨ Generate SEO Metadata",
    disabled=not can_generate,
    key="generate_btn",
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal: cache-wrapped AI call
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_generate(
    topic: str,
    audience: str,
    content_type: str,
    output_language: str,
    transcript: str,
    visual_description: str,
    chapter_notes: str,
    competitor_context: str,
) -> dict:
    """Cache key is the full set of inputs; avoids duplicate API calls on re-renders."""
    return generate_seo_metadata(
        topic=topic,
        audience=audience,
        content_type=content_type,
        output_language=output_language,
        transcript=transcript,
        visual_description=visual_description,
        chapter_notes=chapter_notes,
        competitor_context=competitor_context,
    )


# ─────────────────────────────────────────────────────────────────────────────
# On Generate
# ─────────────────────────────────────────────────────────────────────────────
if generate_clicked:

    # ── 1. Optional competitor scraping ──────────────────────────────────────
    competitor_context = ""
    if competitor_url.strip():
        with st.spinner("🔍 Scraping competitor video metadata..."):
            meta = extract_video_metadata(competitor_url.strip())
        if meta:
            parts = []
            if meta.get("title"):       parts.append(f"Title: {meta['title']}")
            if meta.get("description"): parts.append(f"Description: {meta['description']}")
            if meta.get("creator"):     parts.append(f"Channel: {meta['creator']}")
            competitor_context = "\n".join(parts)
            st.success(
                f"✅ Scraped competitor: **{meta.get('title', 'Unknown')}** "
                f"by **{meta.get('creator', 'Unknown')}**"
            )
        else:
            st.warning(
                "⚠️ Could not scrape the competitor URL (possible bot protection or invalid link). "
                "Generating from your inputs only.",
                icon="⚠️",
            )

    # ── 2. AI generation ─────────────────────────────────────────────────────
    try:
        with st.spinner("🤖 Gemini 2.0 Flash is crafting your SEO package... this may take 15-30 seconds"):
            result = _cached_generate(
                topic=topic.strip(),
                audience=audience.strip(),
                content_type=content_type,
                output_language=output_language,
                transcript=transcript.strip(),
                visual_description=visual_description.strip() if use_visual_desc else "",
                chapter_notes=chapter_notes.strip(),
                competitor_context=competitor_context,
            )
        st.session_state["last_result"] = result
        st.session_state["last_content_type"] = content_type
        st.session_state["last_topic"] = topic.strip()
        st.session_state["last_competitor_title"] = meta.get("title", "") if competitor_url.strip() and meta else ""
        logger.info("UI: SEO generation complete. Rendering results.")

        # ── Real-data niche saturation (replaces Gemini’s guess) ────────────
        with st.spinner("📊 Analysing real YouTube competition for your niche..."):
            real_niche = compute_niche_saturation(topic.strip())
        st.session_state["real_niche"] = real_niche
        logger.info(f"Niche saturation computed: {real_niche.get('saturation_score')}/10")

    except APIException as e:
        st.error(f"🚨 **API Error:** {e}", icon="🚨")
        logger.error(f"API error during generation: {e}")
        st.stop()

    except ValidationException as e:
        st.error(f"⚠️ **Output Validation Error:** {e}", icon="⚠️")
        logger.error(f"Validation error during generation: {e}")
        st.stop()

    except SEOAppException as e:
        st.error(f"❌ **Application Error:** {e}", icon="❌")
        logger.error(f"App error during generation: {e}")
        st.stop()

    except Exception as e:
        st.error(f"❌ **Unexpected Error:** {e}. Please try again.", icon="❌")
        logger.error(f"Unhandled exception in UI: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Results Rendering
# ─────────────────────────────────────────────────────────────────────────────
result = st.session_state.get("last_result")
last_content_type = st.session_state.get("last_content_type", "Long-Form Video")

if result:
    st.markdown("---")
    st.success("🎉 Your SEO package is ready! Use the expanders below to copy each section.", icon="✅")

    # ── Feature 5: Niche Saturation Score (REAL DATA) ──────────────────────────
    niche = st.session_state.get("real_niche") or result.get("niche_analysis", {})
    if niche and isinstance(niche, dict):
        raw_score = niche.get("saturation_score", 5)
        try:
            score = int(raw_score)
        except (ValueError, TypeError):
            score = 5
        score = max(1, min(10, score))

        level = niche.get("competition_level", "Medium")
        rec   = niche.get("recommendation", "")
        avg_views    = niche.get("avg_views", 0)
        n_analyzed   = niche.get("results_analyzed", 0)
        top_titles   = niche.get("top_video_titles", [])
        data_source  = niche.get("data_source", "gemini")
        source_badge = "🟢 Live YouTube Data" if data_source == "live_youtube" else "🟡 AI Estimate"

        if score >= 7:
            bar_color, badge_emoji, badge_label = "#ff4444", "🔴", "Very Crowded"
        elif score >= 4:
            bar_color, badge_emoji, badge_label = "#ffaa00", "🟡", "Moderate"
        else:
            bar_color, badge_emoji, badge_label = "#00cc66", "🟢", "Low Competition"

        with st.expander(f"📊 Niche Saturation Score — {badge_emoji} {badge_label}", expanded=True):
            col_score, col_detail = st.columns([1, 3])
            with col_score:
                st.markdown(
                    f"""
                    <div style='text-align:center;padding:10px'>
                        <div style='font-size:3.5rem;font-weight:800;color:{bar_color};line-height:1'>{score}</div>
                        <div style='font-size:0.85rem;opacity:0.6;margin-top:4px'>out of 10</div>
                        <div style='margin-top:8px;font-weight:600;color:{bar_color}'>{badge_emoji} {badge_label}</div>
                        <div style='margin-top:6px;font-size:0.75rem;opacity:0.55'>{source_badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_detail:
                st.markdown(f"**Competition Level:** {level}")
                st.progress(score / 10)
                if avg_views > 0 and n_analyzed > 0:
                    st.markdown(
                        f"**Data:** Analysed top **{n_analyzed}** YouTube results — "
                        f"average views = **{avg_views:,}** │ max = **{niche.get('max_views', 0):,}**"
                    )
                if rec:
                    st.info(f"💡 **Recommendation:** {rec}", icon="💡")
                if top_titles:
                    st.markdown("**🏹 Top competing videos analyzed:**")
                    for t in top_titles:
                        st.markdown(f"&nbsp;&nbsp;• {t}")

    # ── Titles ────────────────────────────────────────────────────────────────
    with st.expander("🏆 A/B Titles", expanded=True):
        st.markdown("*Click the copy icon on any title to copy it to your clipboard.*")
        for i, title in enumerate(result.get("titles", []), 1):
            st.markdown(f"**Option {i}**")
            st.code(title, language=None)

    # ── Feature 4: Contrarian Hook Generator (with real Divergence Scores) ──────
    contrarian = result.get("contrarian_titles", [])
    competitor_title_for_score = st.session_state.get("last_competitor_title", "")
    if contrarian and isinstance(contrarian, list) and any(contrarian):
        with st.expander("🎭 Contrarian Hooks — Stand-Out Titles", expanded=True):
            st.markdown(
                "*These titles deliberately challenge the competitor\u2019s angle. "
                "**Divergence Score** = how mathematically different it is (Jaccard Word Divergence).*"
            )
            for i, ct in enumerate(contrarian, 1):
                if ct and ct.strip():
                    ct = ct.strip()
                    if competitor_title_for_score:
                        score_data = compute_contrarian_score(competitor_title_for_score, ct)
                        ds = score_data["divergence_score"]
                        interp = score_data["interpretation"]
                        shared = score_data["shared_words"]
                        clr = "#00cc66" if ds >= 7 else "#ffaa00" if ds >= 5 else "#ff4444"
                        score_badge = (
                            f'<span style="color:{clr};font-weight:800;font-size:1.1rem">'
                            f"{ds}/10</span> "
                            f'<span style="opacity:0.6;font-size:0.85rem">{interp}</span>'
                        )
                        if shared:
                            overlap_note = f"*Shared words (reduced contrast): {', '.join(shared)}*"
                        else:
                            overlap_note = "*No word overlap with competitor — maximum contrast!*"
                    else:
                        score_badge = ""
                        overlap_note = ""

                    st.markdown(f"**Contrarian Title {i}**")
                    if score_badge:
                        st.markdown(score_badge, unsafe_allow_html=True)
                    st.code(ct, language=None)
                    if overlap_note:
                        st.caption(overlap_note)
            st.caption(
                "💡 Tip: Contrarian titles often get 2-3x higher CTR because they disrupt viewer expectations."
            )

    # ── Description ───────────────────────────────────────────────────────────
    with st.expander("📄 Optimized Description", expanded=True):
        st.code(result.get("description", ""), language=None)

    # ── Timestamps (long-form only) ───────────────────────────────────────────
    timestamps = result.get("timestamps", [])
    if last_content_type == "Long-Form Video":
        with st.expander("⏱️ Timestamps", expanded=bool(timestamps)):
            if timestamps:
                ts_text = "\n".join(
                    f"{ts.get('time', '')} {ts.get('label', '')}"
                    for ts in timestamps
                )
                st.code(ts_text, language=None)
            else:
                st.info(
                    "No timestamps generated — provide Chapter Notes to enable this section.",
                    icon="ℹ️",
                )

    # ── Tags ─────────────────────────────────────────────────────────────────
    with st.expander("🏷️ SEO Tags", expanded=True):
        tags = result.get("tags", [])
        if tags:
            # Visual pill display
            tag_html = "".join(f'<span class="tag-pill">{t}</span>' for t in tags)
            st.markdown(
                f'<div class="output-card">{tag_html}</div>', unsafe_allow_html=True
            )
            # Also a copyable version
            st.markdown("**Copy all tags (comma-separated):**")
            st.code(", ".join(tags), language=None)
            total_chars = len(", ".join(tags))
            st.caption(f"Tag character count: {total_chars}/500")
        else:
            st.warning("No tags returned.")

    # ── Social Media Posts ────────────────────────────────────────────────────
    with st.expander("📱 Social Media Posts", expanded=True):
        social = result.get("social_posts", {})
        tab_tw, tab_li, tab_ig = st.tabs(["Twitter / X", "LinkedIn", "Instagram"])
        with tab_tw:
            tweet = social.get("twitter", "")
            st.code(tweet, language=None)
            st.caption(f"{len(tweet)}/280 characters")
        with tab_li:
            st.code(social.get("linkedin", ""), language=None)
        with tab_ig:
            st.code(social.get("instagram", ""), language=None)

    # ── Thumbnail Ideas ───────────────────────────────────────────────────────
    with st.expander("🖼️ Thumbnail Ideas", expanded=True):
        for i, idea in enumerate(result.get("thumbnail_ideas", []), 1):
            st.markdown(f"**Concept {i}:** {idea}")
