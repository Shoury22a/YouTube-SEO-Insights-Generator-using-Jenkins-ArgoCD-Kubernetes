"""
YouTube SEO Insights Generator — Streamlit Frontend
Provides a rich UI for generating AI-powered YouTube SEO metadata.
"""

import os
import sys
import streamlit as st
from src.logger import get_logger
from src.extractor import extract_video_metadata
from src.ai_model import generate_seo_metadata, MAX_TRANSCRIPT_CHARS
from src.exception import APIException, ValidationException, SEOAppException

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube SEO Insights Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Custom Premium CSS (Glassmorphism & Vibrant Aesthetics)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Modern Typography */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] { 
            font-family: 'Outfit', sans-serif; 
            color: #f0f0f0;
        }

        /* Responsive Background */
        .stApp {
            background: radial-gradient(circle at top right, #1e1e3f 0%, #0f0f1a 100%);
            background-attachment: fixed;
        }

        /* Glassmorphism Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Premium Hero Banner Aria */
        .hero-banner {
            background: linear-gradient(135deg, hsl(340, 100%, 50%) 0%, hsl(280, 100%, 40%) 100%);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 15px 45px rgba(255, 0, 80, 0.2);
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .hero-banner h1 { 
            color: #fff; 
            font-size: 3rem; 
            margin: 0; 
            font-weight: 700; 
            letter-spacing: -1px;
            text-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        .hero-banner p { 
            color: rgba(255,255,255,0.9); 
            margin-top: 10px; 
            font-size: 1.2rem; 
            font-weight: 300;
        }

        /* Glass Cards for Sections */
        .stExpander {
            background: rgba(255, 255, 255, 0.04) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            overflow: hidden;
            margin-bottom: 12px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .stExpander:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
            border-color: rgba(255, 255, 255, 0.15) !important;
        }

        /* Glowing Generate Button */
        .stButton > button {
            background: linear-gradient(90deg, #ff0050 0%, #cc00ff 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            padding: 15px 0 !important;
            width: 100% !important;
            box-shadow: 0 8px 20px rgba(255, 0, 80, 0.3) !important;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 12px 30px rgba(255, 0, 80, 0.5) !important;
        }
        .stButton > button:active {
            transform: scale(0.98);
        }

        /* Tag Pill Modern Styling */
        .tag-pill {
            display: inline-block;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #ff0050;
            border-radius: 30px;
            padding: 5px 15px;
            margin: 5px;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .tag-pill:hover {
            background: #ff0050;
            color: white;
            border-color: #ff0050;
        }

        /* Input Overrides */
        .stTextArea textarea, .stTextInput input {
            background: rgba(0,0,0,0.2) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 10px !important;
            color: #fff !important;
            padding: 15px !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #ff0050 !important;
            box-shadow: 0 0 10px rgba(255,0,80,0.2) !important;
        }

        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .hero-banner h1 { font-size: 2.2rem; }
            .hero-banner p { font-size: 1rem; }
            [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🎬 YouTube SEO Insights Generator</h1>
        <p>Generate A/B titles, descriptions, timestamps, tags, social posts &amp;
           thumbnail ideas — powered by GPT-4.</p>
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
        with st.spinner("🤖 GPT-4 is crafting your SEO package... this may take 15-30 seconds"):
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
        logger.info("UI: SEO generation complete. Rendering results.")

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

    # ── Titles ────────────────────────────────────────────────────────────────
    with st.expander("🏆 A/B Titles", expanded=True):
        st.markdown("*Click the copy icon on any title to copy it to your clipboard.*")
        for i, title in enumerate(result.get("titles", []), 1):
            st.markdown(f"**Option {i}**")
            st.code(title, language=None)

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
