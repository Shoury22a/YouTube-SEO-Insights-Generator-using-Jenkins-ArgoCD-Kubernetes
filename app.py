"""
YouTube SEO Insights Generator — Streamlit Frontend
Provides a rich UI for generating AI-powered YouTube SEO metadata.
"""

import base64
import streamlit as st
from src.logger import get_logger
from src.extractor import extract_video_metadata, compute_niche_saturation, compute_contrarian_score
from src.ai_model import generate_seo_metadata, generate_seo_metadata_agentic, check_api_connection, MAX_TRANSCRIPT_CHARS
from src.title_scorer import score_title, compute_report_card
from src.pdf_exporter import build_pdf
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
# Theme Management
# ─────────────────────────────────────────────────────────────────────────────
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"

def inject_custom_css(theme: str):
    """Injects premium, vibrant Light/Dark mode CSS."""
    is_dark = theme == "Dark"

    if is_dark:
        bg_main         = "#07070f"
        bg_surface      = "#0e0e1c"
        text_main       = "#eeeef5"
        text_sub        = "rgba(238,238,245,0.55)"
        sidebar_bg      = "rgba(10,10,20,0.92)"
        glass_bg        = "rgba(255,255,255,0.035)"
        glass_bg_hover  = "rgba(255,255,255,0.06)"
        border_clr      = "rgba(255,255,255,0.08)"
        border_accent   = "rgba(255,30,90,0.35)"
        input_bg        = "rgba(255,255,255,0.06)"
        input_focus_shadow = "rgba(255,30,90,0.18)"
        hero_bg         = "linear-gradient(135deg, #130520 0%, #0a0a1a 60%, #150826 100%)"
        hero_txt        = "linear-gradient(135deg, #ffffff 0%, #c084fc 100%)"
        hero_sub_clr    = "rgba(230,220,255,0.65)"
        badge_bg        = "linear-gradient(90deg, rgba(255,30,90,0.25), rgba(160,0,255,0.25))"
        badge_border    = "rgba(255,30,90,0.5)"
        badge_clr       = "#ff6baa"
        chip_bg         = "rgba(255,255,255,0.04)"
        tag_bg          = "rgba(255,30,90,0.1)"
        tag_border      = "rgba(255,30,90,0.3)"
        tag_clr         = "#ff80aa"
        btn_shadow      = "rgba(255,30,90,0.45)"
        btn_hover_shadow= "rgba(255,30,90,0.65)"
        orbs            = """
            radial-gradient(ellipse at 8% 15%, rgba(255,30,90,0.12) 0%, transparent 45%),
            radial-gradient(ellipse at 92% 85%, rgba(140,0,255,0.12) 0%, transparent 45%),
            radial-gradient(ellipse at 50% 100%, rgba(0,180,255,0.06) 0%, transparent 40%);
        """
        expander_glow   = "0 0 0 1px rgba(255,30,90,0.0)"
        expander_glow_h = "0 4px 28px rgba(255,30,90,0.12)"
        code_bg         = "rgba(255,255,255,0.04)"
        success_bg      = "rgba(0,220,130,0.1)"
        metric_bg       = "rgba(255,255,255,0.04)"
    else:
        bg_main         = "#f4f3ff"
        bg_surface      = "#ffffff"
        text_main       = "#1a1830"
        text_sub        = "rgba(26,24,48,0.55)"
        sidebar_bg      = "rgba(255,255,255,0.97)"
        glass_bg        = "rgba(255,255,255,0.8)"
        glass_bg_hover  = "rgba(255,255,255,0.95)"
        border_clr      = "rgba(26,24,48,0.09)"
        border_accent   = "rgba(99,53,220,0.3)"
        input_bg        = "#ffffff"
        input_focus_shadow = "rgba(99,53,220,0.15)"
        hero_bg         = "linear-gradient(135deg, #ede9fe 0%, #fdf4ff 50%, #fce7f3 100%)"
        hero_txt        = "linear-gradient(135deg, #1a1830 0%, #6335dc 100%)"
        hero_sub_clr    = "rgba(26,24,48,0.7)"
        badge_bg        = "linear-gradient(90deg, rgba(99,53,220,0.12), rgba(219,39,119,0.12))"
        badge_border    = "rgba(99,53,220,0.4)"
        badge_clr       = "#6335dc"
        chip_bg         = "rgba(99,53,220,0.06)"
        tag_bg          = "rgba(99,53,220,0.08)"
        tag_border      = "rgba(99,53,220,0.25)"
        tag_clr         = "#6335dc"
        btn_shadow      = "rgba(99,53,220,0.35)"
        btn_hover_shadow= "rgba(99,53,220,0.55)"
        orbs            = ""
        expander_glow   = "0 2px 12px rgba(99,53,220,0.05)"
        expander_glow_h = "0 6px 32px rgba(99,53,220,0.14)"
        code_bg         = "rgba(99,53,220,0.04)"
        success_bg      = "rgba(0,180,100,0.08)"
        metric_bg       = "rgba(99,53,220,0.04)"

    # Shared gradient — adapts accent to theme
    grad = "linear-gradient(90deg, #ff1e5a 0%, #cc00ff 100%)" if is_dark else \
           "linear-gradient(90deg, #6335dc 0%, #db2777 100%)"

    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

            /* ── Keyframe Animations ── */
            @keyframes shimmer {{
                0%   {{ background-position: -200% center; }}
                100% {{ background-position: 200% center; }}
            }}
            @keyframes pulse-glow {{
                0%, 100% {{ box-shadow: 0 0 8px rgba(255,30,90,0.3); }}
                50%       {{ box-shadow: 0 0 22px rgba(255,30,90,0.65); }}
            }}
            @keyframes float {{
                0%, 100% {{ transform: translateY(0px); }}
                50%       {{ transform: translateY(-4px); }}
            }}
            @keyframes fadeSlideIn {{
                from {{ opacity: 0; transform: translateY(12px); }}
                to   {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes scoreBar {{
                from {{ width: 0; }}
            }}

            /* ── Base ── */
            html, body, [class*="css"] {{
                font-family: 'Outfit', sans-serif;
                color: {text_main};
            }}
            .stApp {{
                background: {bg_main};
                {"background-image:" + orbs if orbs.strip() else ""}
                background-attachment: fixed;
            }}

            /* ── Sidebar ── */
            section[data-testid="stSidebar"] {{
                background: {sidebar_bg} !important;
                backdrop-filter: blur(24px);
                border-right: 1px solid {border_clr};
                box-shadow: 4px 0 40px rgba(0,0,0,{"0.3" if is_dark else "0.06"});
            }}
            section[data-testid="stSidebar"] h2 {{
                font-size: 0.78rem !important;
                text-transform: uppercase;
                letter-spacing: 2px;
                opacity: 0.5;
                font-weight: 700;
                margin-bottom: 6px;
            }}

            /* ── Hero Banner ── */
            .hero-banner {{
                position: relative;
                background: {hero_bg};
                border-radius: 24px;
                padding: 36px 44px;
                margin-bottom: 28px;
                border: 1px solid {border_accent};
                overflow: hidden;
                animation: fadeSlideIn 0.5s ease;
                box-shadow: {expander_glow_h};
            }}
            .hero-banner::before {{
                content: '';
                position: absolute;
                top: -60px; right: -60px;
                width: 280px; height: 280px;
                background: {"radial-gradient(circle, rgba(255,30,90,0.15) 0%, transparent 65%)" if is_dark else "radial-gradient(circle, rgba(99,53,220,0.1) 0%, transparent 65%)"};
                border-radius: 50%;
                pointer-events: none;
            }}
            .hero-brand {{
                display: flex;
                align-items: center;
                gap: 20px;
                margin-bottom: 14px;
            }}
            .hero-brand img {{
                width: 64px !important;
                height: 64px !important;
                border-radius: 18px;
                box-shadow: 0 0 0 2px {border_accent}, 0 0 30px {"rgba(255,30,90,0.4)" if is_dark else "rgba(99,53,220,0.25)"};
                animation: float 3s ease-in-out infinite;
            }}
            .hero-badge {{
                display: inline-block;
                background: {badge_bg};
                border: 1px solid {badge_border};
                border-radius: 30px;
                padding: 4px 16px;
                font-size: 0.72rem;
                font-weight: 700;
                color: {badge_clr};
                letter-spacing: 1.5px;
                text-transform: uppercase;
                margin-bottom: 10px;
                background-size: 200% auto;
                animation: shimmer 3s linear infinite;
            }}
            .hero-banner h1 {{
                font-size: 2.8rem;
                margin: 0;
                font-weight: 900;
                letter-spacing: -2px;
                line-height: 1.05;
                background: {hero_txt};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .hero-sub {{
                color: {hero_sub_clr};
                margin-top: 10px;
                font-size: 1.05rem;
                font-weight: 400;
                max-width: 660px;
                line-height: 1.6;
            }}

            /* ── Stat Chips ── */
            .hero-stats {{
                display: flex;
                gap: 14px;
                margin-top: 28px;
                flex-wrap: wrap;
            }}
            .stat-chip {{
                background: {chip_bg};
                border: 1px solid {border_clr};
                border-radius: 14px;
                padding: 10px 20px;
                font-size: 0.82rem;
                font-weight: 600;
                color: {text_sub};
                transition: all 0.25s ease;
                cursor: default;
            }}
            .stat-chip:hover {{
                border-color: {border_accent};
                transform: translateY(-2px);
                box-shadow: {expander_glow_h};
            }}
            .stat-chip span {{
                display: block;
                font-size: 1.15rem;
                font-weight: 800;
                background: {grad};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 1px;
            }}

            /* ── Expander Cards ── */
            .stExpander {{
                background: {glass_bg} !important;
                border-radius: 18px !important;
                border: 1px solid {border_clr} !important;
                margin-bottom: 14px;
                transition: all 0.3s ease;
                box-shadow: {expander_glow};
                animation: fadeSlideIn 0.4s ease;
                backdrop-filter: blur(10px);
            }}
            .stExpander:hover {{
                border-color: {border_accent} !important;
                box-shadow: {expander_glow_h} !important;
                background: {glass_bg_hover} !important;
            }}
            .stExpander summary p {{
                color: {text_main} !important;
                font-weight: 700 !important;
                font-size: 0.95rem !important;
                letter-spacing: 0.2px;
            }}

            /* ── Generate Button ── */
            .stButton > button {{
                background: {grad} !important;
                color: white !important;
                border: none !important;
                border-radius: 16px !important;
                font-family: 'Outfit', sans-serif !important;
                font-weight: 800 !important;
                font-size: 1.05rem !important;
                padding: 18px 0 !important;
                width: 100% !important;
                letter-spacing: 0.5px;
                box-shadow: 0 8px 32px {btn_shadow} !important;
                transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important;
                position: relative;
                overflow: hidden;
            }}
            .stButton > button:hover {{
                box-shadow: 0 14px 44px {btn_hover_shadow} !important;
                transform: translateY(-3px) scale(1.01) !important;
            }}
            .stButton > button:active {{
                transform: translateY(0) scale(0.99) !important;
            }}

            /* ── Download Button ── */
            .stDownloadButton > button {{
                background: {glass_bg} !important;
                color: {text_main} !important;
                border: 1px solid {border_accent} !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
                transition: all 0.25s ease !important;
            }}
            .stDownloadButton > button:hover {{
                background: {grad} !important;
                color: white !important;
                transform: translateY(-2px) !important;
            }}

            /* ── Input Fields ── */
            .stTextArea textarea, .stTextInput input {{
                background: {input_bg} !important;
                border: 1.5px solid {border_clr} !important;
                border-radius: 14px !important;
                color: {text_main} !important;
                -webkit-text-fill-color: {text_main} !important;
                font-family: 'Outfit', sans-serif !important;
                font-size: 0.95rem !important;
                transition: all 0.2s ease !important;
                padding: 10px 14px !important;
            }}
            .stTextArea textarea:focus, .stTextInput input:focus {{
                border-color: {border_accent} !important;
                box-shadow: 0 0 0 4px {input_focus_shadow} !important;
                outline: none !important;
            }}
            div[data-baseweb="select"] > div {{
                background: {input_bg} !important;
                border: 1.5px solid {border_clr} !important;
                border-radius: 12px !important;
                color: {text_main} !important;
            }}
            ::placeholder {{
                color: {text_sub} !important;
                font-style: italic;
            }}

            /* ── Tags ── */
            .tag-pill {{
                display: inline-block;
                background: {tag_bg};
                border: 1px solid {tag_border};
                color: {tag_clr};
                border-radius: 30px;
                padding: 5px 16px;
                margin: 4px;
                font-size: 0.83rem;
                font-weight: 600;
                transition: all 0.2s ease;
                cursor: default;
            }}
            .tag-pill:hover {{
                background: {grad};
                color: #fff !important;
                border-color: transparent;
                box-shadow: 0 4px 16px {btn_shadow};
                transform: translateY(-1px);
            }}

            /* ── Metrics / Info ── */
            [data-testid="stMetric"] {{
                background: {metric_bg};
                border-radius: 14px;
                padding: 14px 18px;
                border: 1px solid {border_clr};
                transition: all 0.25s ease;
            }}
            [data-testid="stMetric"]:hover {{
                border-color: {border_accent};
                box-shadow: {expander_glow_h};
            }}
            [data-testid="stMetricLabel"] {{
                color: {text_sub} !important;
                font-size: 0.78rem !important;
                font-weight: 600 !important;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            [data-testid="stMetricValue"] {{
                color: {text_main} !important;
                font-weight: 800 !important;
                font-size: 1.5rem !important;
            }}

            /* ── Success / Info / Warning Alerts ── */
            .stSuccess {{
                background: {success_bg} !important;
                border: 1px solid rgba(0,200,120,0.3) !important;
                border-radius: 14px !important;
                animation: fadeSlideIn 0.3s ease;
            }}
            .stInfo {{
                background: {"rgba(30,100,255,0.08)" if is_dark else "rgba(99,53,220,0.06)"} !important;
                border: 1px solid {"rgba(30,100,255,0.2)" if is_dark else "rgba(99,53,220,0.2)"} !important;
                border-radius: 14px !important;
            }}

            /* ── Code blocks ── */
            .stCodeBlock pre, .stCode code {{
                background: {code_bg} !important;
                border-radius: 12px !important;
                border: 1px solid {border_clr} !important;
                font-size: 0.9rem !important;
            }}

            /* ── Progress ── */
            .stProgress > div > div > div {{
                background: {grad} !important;
                border-radius: 4px;
            }}

            /* ── Tabs ── */
            .stTabs [data-baseweb="tab"] {{
                font-family: 'Outfit', sans-serif !important;
                font-weight: 600 !important;
                color: {text_sub} !important;
            }}
            .stTabs [aria-selected="true"] {{
                color: {text_main} !important;
                border-bottom: 2px solid {"#ff1e5a" if is_dark else "#6335dc"} !important;
            }}

            /* ── Dividers ── */
            hr {{ border-color: {border_clr} !important; margin: 24px 0; }}

            /* ── Widget labels ── */
            [data-testid="stWidgetLabel"] p,
            [data-testid="stRadio"] label p,
            .stAlert p,
            .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
                color: {text_main} !important;
            }}
            section[data-testid="stSidebar"] .stMarkdown p,
            section[data-testid="stSidebar"] .stMarkdown h1,
            section[data-testid="stSidebar"] .stMarkdown h2 {{
                color: {text_main} !important;
            }}
            .sidebar-brand-name {{
                font-weight: 800;
                font-size: 1.1rem;
                background: {grad};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            /* ── Mobile ── */
            @media (max-width: 768px) {{
                .hero-banner {{ padding: 24px 20px; }}
                .hero-banner h1 {{ font-size: 2rem; }}
                .hero-stats {{ gap: 8px; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css(st.session_state.theme_mode)


# ─────────────────────────────────────────────────────────────────────────────
# Hero Banner — TubeRank AI
# ─────────────────────────────────────────────────────────────────────────────

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
                <div class="stat-chip"><span>6</span>Agent Nodes</div>
                <div class="stat-chip"><span>∞</span>Self-Correcting</div>
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
    
    # Theme Toggle
    theme_choice = st.selectbox(
        "🌓 App Theme",
        options=["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "Dark" else 1,
        help="Switch between Dark and Light mode for the interface."
    )
    if theme_choice != st.session_state.theme_mode:
        st.session_state.theme_mode = theme_choice
        st.rerun()

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
    st.markdown("## 🤖 AI Mode")
    use_agent = st.toggle(
        "⚡ Agentic AI Mode",
        value=True,
        help="Enable the full LangGraph agent pipeline with RAG memory, web search, self-critique, and auto-refinement. Disable for the faster linear generation.",
    )

    st.markdown("---")
    st.markdown("## 🔗 Competitor Reference (Optional)")
    competitor_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="We'll scrape metadata to give the AI competitive context.",
    )

    # ── Title Tester Widget ───────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔬 Title Tester")
    st.caption("Instantly score any title — no generation needed.")
    test_title_input = st.text_input(
        "Type any title to analyse:",
        placeholder="e.g. 7 Python Mistakes Nobody Warns You About",
        key="sidebar_title_tester",
        label_visibility="collapsed",
    )
    test_topic_input = st.text_input(
        "Topic keyword (for SEO check):",
        placeholder="e.g. Python mistakes",
        key="sidebar_topic_tester",
        label_visibility="collapsed",
    )
    if test_title_input.strip():
        ts = score_title(
            test_title_input.strip(),
            test_topic_input.strip() or test_title_input.strip(),
            content_type,
        )
        ctr_c = "#00e676" if ts["ctr_score"] >= 7 else "#ffaa00" if ts["ctr_score"] >= 4 else "#ff4444"
        seo_c = "#00e676" if ts["seo_score"] >= 7 else "#ffaa00" if ts["seo_score"] >= 4 else "#ff4444"
        ch_c  = {"green": "#00e676", "amber": "#ffaa00", "red": "#ff4444"}[ts["char_status"]]
        st.markdown(
            f"""
            <div style='background:rgba(255,255,255,0.04);border-radius:8px;
                        padding:10px 12px;font-size:0.8rem;margin-top:4px'>
              <div style='margin-bottom:6px'>
                <span style='color:{ctr_c};font-weight:700'>📊 CTR {ts['ctr_score']}/10</span>
                &nbsp;&nbsp;
                <span style='color:{seo_c};font-weight:700'>🔑 SEO {ts['seo_score']}/10</span>
                &nbsp;&nbsp;
                <span style='color:{ch_c};font-weight:700'>📏 {ts['char_count']} chars</span>
              </div>
              <div style='margin-bottom:4px'>{ts['hook_emoji']} <strong>{ts['hook_type']}</strong></div>
              <div style='opacity:0.6;font-size:0.73rem'>💡 {ts['feedback']}</div>
            </div>
            """,
            unsafe_allow_html=True,
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
            if meta.get("title"):
                parts.append(f"Title: {meta['title']}")
            if meta.get("description"):
                parts.append(f"Description: {meta['description']}")
            if meta.get("creator"):
                parts.append(f"Channel: {meta['creator']}")
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
        if use_agent:
            # ── Agentic Mode: LangGraph pipeline ─────────────────────────
            with st.status("🤖 **Agent is working...**", expanded=True) as status:
                st.write("🔍 **Researcher:** Searching memory & web for context...")
                result = generate_seo_metadata_agentic(
                    topic=topic.strip(),
                    audience=audience.strip(),
                    content_type=content_type,
                    output_language=output_language,
                    transcript=transcript.strip(),
                    visual_description=visual_description.strip() if use_visual_desc else "",
                    chapter_notes=chapter_notes.strip(),
                    competitor_context=competitor_context,
                )
                # Show the agent's step log live
                agent_log = result.pop("_agent_log", [])
                agent_retries = result.pop("_agent_retries", 0)
                agent_elapsed = result.pop("_agent_elapsed", 0)
                rag_count = result.pop("_rag_count", 0)

                for step in agent_log:
                    st.write(step)

                status.update(
                    label=f"✅ Agent complete in {agent_elapsed}s | "
                          f"{agent_retries} refinements | {rag_count} docs from memory",
                    state="complete",
                    expanded=False,
                )

            st.session_state["agent_log"] = agent_log
            st.session_state["agent_retries"] = agent_retries
            st.session_state["agent_elapsed"] = agent_elapsed
            st.session_state["rag_count"] = rag_count
        else:
            # ── Linear Mode: Original pipeline ───────────────────────────
            with st.spinner("🤖 Gemini is crafting your SEO package... this may take 15-30 seconds"):
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
            st.session_state["agent_log"] = []
            st.session_state["agent_retries"] = 0

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

    # ── PDF Download ─────────────────────────────────────────────────────
    try:
        _pdf_topic = st.session_state.get("last_topic", topic.strip())
        _report = compute_report_card(result, _pdf_topic, last_content_type)
        _title_scores = [
            score_title(t, _pdf_topic, last_content_type)
            for t in result.get("titles", [])
        ]
        _pdf_bytes = build_pdf(
            result=result,
            topic=_pdf_topic,
            content_type=last_content_type,
            report_card=_report,
            title_scores=_title_scores,
        )
        st.download_button(
            label="📥 Download Full SEO Package (PDF)",
            data=_pdf_bytes,
            file_name=f"TubeRank_SEO_{_pdf_topic[:30].replace(' ','_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as _pdf_err:
        logger.warning(f"PDF generation failed: {_pdf_err}")

    # ── Agent Reasoning Log ─────────────────────────────────────────────────
    agent_log = st.session_state.get("agent_log", [])
    if agent_log:
        retries = st.session_state.get("agent_retries", 0)
        elapsed = st.session_state.get("agent_elapsed", 0)
        rag_ct = st.session_state.get("rag_count", 0)
        with st.expander(
            f"🧠 Agent Reasoning — {retries} refinements, "
            f"{rag_ct} docs from memory, {elapsed}s total",
            expanded=False,
        ):
            for step in agent_log:
                st.markdown(f"  {step}")

    # ── SEO Report Card ─────────────────────────────────────────────────────
    report = compute_report_card(
        result=result,
        topic=st.session_state.get("last_topic", ""),
        content_type=last_content_type,
    )
    grade = report["grade"]
    grade_color = (
        "#00e676" if grade in ("A+", "A") else
        "#ffaa00" if grade in ("B+", "B") else
        "#ff4444"
    )
    with st.expander("📊 SEO Report Card", expanded=True):
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;gap:20px;margin-bottom:12px'>
                <div style='text-align:center;background:rgba(255,255,255,0.05);
                            border-radius:12px;padding:12px 20px;min-width:80px'>
                    <div style='font-size:2.8rem;font-weight:900;color:{grade_color};line-height:1'>{grade}</div>
                    <div style='font-size:0.7rem;opacity:0.5;margin-top:4px'>Overall Grade</div>
                </div>
                <div style='flex:1'>
                    <div style='font-size:0.8rem;opacity:0.55;margin-bottom:6px'>
                        Score: <strong>{report["overall_score"]}/10</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metrics = [
            ("🏆 Title Strength", report["title_strength"], "Average CTR power across all titles", True),
            ("🔑 Keyword Coverage", report["keyword_coverage"], "How many titles contain your core keyword", True),
            ("📄 Description Hook", report["description_hook"], "Hook-word strength in opening sentences", True),
            ("🏷️ Tag Diversity", report["tag_diversity"], "Unique tags vs total (higher = better)", True),
            ("📏 Tag Budget Used", report["tag_fill"], "YouTube's 500-char tag limit utilisation", False),
        ]
        for label, val, tip, higher_better in metrics:
            val_clamped = max(0.0, min(10.0, val))
            bar_color = (
                "#00e676" if val_clamped >= 7 else
                "#ffaa00" if val_clamped >= 4 else
                "#ff4444"
            )
            filled = int(val_clamped)
            empty = 10 - filled
            bar = (
                f'<span style="color:{bar_color}">'
                + "█" * filled
                + f'</span><span style="opacity:0.2">'
                + "█" * empty
                + "</span>"
            )
            st.markdown(
                f"<div style='margin-bottom:8px'>"
                f"<span style='font-size:0.85rem'>{label}</span>"
                f"<span style='float:right;font-size:0.8rem;color:{bar_color};font-weight:700'>{val_clamped:.1f}/10</span>"
                f"<br><span style='font-family:monospace;font-size:0.9rem'>{bar}</span>"
                f"<br><span style='font-size:0.72rem;opacity:0.5'>{tip}</span></div>",
                unsafe_allow_html=True,
            )

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

    # ── A/B Title Analytics Dashboard ─────────────────────────────────────────
    with st.expander("🏆 A/B Title Analytics Dashboard", expanded=True):
        titles = result.get("titles", [])
        topic_for_score = st.session_state.get("last_topic", "")
        if not titles:
            st.warning("No titles returned.")
        else:
            st.caption(
                "Each title is scored on CTR potential and SEO keyword alignment. "
                "Hook types identified using pattern matching on 80+ power words."
            )
            for i, title in enumerate(titles, 1):
                s = score_title(title, topic_for_score, last_content_type)
                ctr = s["ctr_score"]
                seo = s["seo_score"]
                ctr_color = "#00e676" if ctr >= 7 else "#ffaa00" if ctr >= 4 else "#ff4444"
                seo_color = "#00e676" if seo >= 7 else "#ffaa00" if seo >= 4 else "#ff4444"
                char_color = {"green": "#00e676", "amber": "#ffaa00", "red": "#ff4444"}[s["char_status"]]

                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.04);border-radius:10px;"
                    f"padding:14px 16px;margin-bottom:12px;border-left:3px solid {ctr_color}'>"
                    f"<div style='font-size:0.75rem;opacity:0.5;margin-bottom:4px'>Option {i}</div>"
                    f"<div style='font-size:1rem;font-weight:600;margin-bottom:10px'>{title}</div>"
                    f"<div style='display:flex;gap:10px;flex-wrap:wrap;font-size:0.78rem'>"
                    f"<span style='background:rgba(255,255,255,0.07);border-radius:6px;padding:3px 8px'>"
                    f"📊 CTR <strong style='color:{ctr_color}'>{ctr}/10</strong></span>"
                    f"<span style='background:rgba(255,255,255,0.07);border-radius:6px;padding:3px 8px'>"
                    f"🔑 SEO <strong style='color:{seo_color}'>{seo}/10</strong></span>"
                    f"<span style='background:rgba(255,255,255,0.07);border-radius:6px;padding:3px 8px'>"
                    f"📏 <strong style='color:{char_color}'>{s['char_count']} chars</strong></span>"
                    f"<span style='background:rgba(255,255,255,0.07);border-radius:6px;padding:3px 8px'>"
                    f"{s['hook_emoji']} {s['hook_type']}</span>"
                    + (
                        f"<span style='background:rgba(255,255,255,0.07);border-radius:6px;padding:3px 8px'>"
                        f"🔢 Has number</span>" if s["has_number"] else ""
                    )
                    + (
                        f"<span style='background:rgba(255,255,255,0.07);border-radius:6px;padding:3px 8px'>"
                        f"📌 Has brackets</span>" if s["has_brackets"] else ""
                    )
                    + f"</div>"
                    + (
                        f"<div style='margin-top:8px;font-size:0.73rem;opacity:0.5'>"
                        f"Power words: {', '.join(s['power_words'])}</div>" if s["power_words"] else ""
                    )
                    + f"<div style='margin-top:6px;font-size:0.75rem;color:#aaa'>💡 {s['feedback']}</div>"
                    + "</div>",
                    unsafe_allow_html=True,
                )
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

    # ── Thumbnail Lab ─────────────────────────────────────────────────────────
    with st.expander("🖼️ Thumbnail Lab", expanded=True):
        st.markdown(
            "*AI-generated thumbnail concepts for each psychological hook style. "
            "Click **Generate Images** to create actual visual mockups using Gemini Imagen.*"
        )
        thumbnail_concepts = result.get("thumbnail_ideas", [])
        if thumbnail_concepts:
            st.markdown("**📋 AI Thumbnail Concepts (text):**")
            for i, idea in enumerate(thumbnail_concepts, 1):
                st.markdown(f"**Concept {i}:** {idea}")

        st.markdown("---")

        if st.button("🎨 Generate Thumbnail Images (4 Styles)", use_container_width=True,
                     help="Uses Gemini Imagen 3 to generate viral-ready thumbnail mockups"):
            from src.thumbnail_gen import generate_thumbnails
            import os
            with st.spinner("🖼️ Generating 4 thumbnail mockups with Gemini Imagen..."):
                try:
                    thumbs = generate_thumbnails(
                        topic=st.session_state.get("last_topic", ""),
                        thumbnail_concepts=thumbnail_concepts,
                        api_key=os.getenv("GOOGLE_API_KEY"),
                    )
                    st.session_state["thumbnails"] = thumbs
                except Exception as _te:
                    st.warning(f"Thumbnail generation failed: {_te}")

        # Display generated thumbnails (2x2 grid)
        thumbs = st.session_state.get("thumbnails", [])
        if thumbs:
            col_a, col_b = st.columns(2)
            cols = [col_a, col_b, col_a, col_b]
            for i, thumb in enumerate(thumbs):
                with cols[i]:
                    st.markdown(
                        f"<div style='background:{thumb['color']}22;border:1px solid {thumb['color']}55;"
                        f"border-radius:10px;padding:10px;margin-bottom:10px'>"
                        f"<div style='font-weight:700;margin-bottom:4px'>"
                        f"{thumb['emoji']} {thumb['style']}</div>"
                        f"<div style='font-size:0.75rem;opacity:0.6;margin-bottom:8px'>"
                        f"{thumb['tip']}</div></div>",
                        unsafe_allow_html=True,
                    )
                    if thumb.get("image_b64"):
                        st.image(
                            f"data:image/png;base64,{thumb['image_b64']}",
                            caption=thumb["style"],
                            use_container_width=True,
                        )
                    else:
                        st.info(
                            f"Imagen API unavailable for {thumb['style']} style. "
                            "Use the text concept above in Canva or Photoshop.",
                            icon="ℹ️",
                        )

    # ── History Dashboard Link ────────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "📊 **Want to see all your past generations and analytics?** "
        "Use the sidebar navigation → **My History** page.",
        icon="📊",
    )
