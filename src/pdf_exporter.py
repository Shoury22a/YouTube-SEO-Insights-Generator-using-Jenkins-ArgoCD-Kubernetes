"""
PDF Exporter — Branded SEO Package Export for TubeRank AI.

Compiles the entire AI-generated SEO output (titles with CTR/SEO scores,
description, tags, timestamps, social posts, SEO Report Card grade)
into a professional, branded PDF using fpdf2.

Returns PDF as bytes (compatible with st.download_button — no temp files).
"""

from io import BytesIO
from datetime import datetime
from src.logger import get_logger

logger = get_logger(__name__)

# ── Colour palette matching the TubeRank AI brand ──────────────────────────
_RED   = (220, 20, 60)
_GREY  = (100, 100, 110)
_BLACK = (15, 15, 20)
_WHITE = (248, 248, 252)
_LIGHT = (240, 240, 248)


def _grade_color(grade: str) -> tuple[int, int, int]:
    if grade in ("A+", "A"):
        return (0, 200, 100)
    if grade in ("B+", "B"):
        return (255, 170, 0)
    return (220, 60, 60)


def build_pdf(
    result: dict,
    topic: str,
    content_type: str = "Long-Form Video",
    report_card: dict | None = None,
    title_scores: list[dict] | None = None,
) -> bytes:
    """
    Builds a branded PDF of the full SEO package.

    Args:
        result:       The full generation result dict from generate_seo_metadata_agentic()
        topic:        The user's topic string
        content_type: "Long-Form Video" or "YouTube Short"
        report_card:  Output of compute_report_card() (optional)
        title_scores: List of score_title() outputs, one per title (optional)

    Returns:
        PDF as bytes — pass directly to st.download_button(data=...).
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.error("fpdf2 not installed. Run: pip install fpdf2")
        raise

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Helper: section header ───────────────────────────────────────────
    def section_header(text: str):
        pdf.set_fill_color(*_RED)
        pdf.set_text_color(*_WHITE)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"  {text}", fill=True, ln=True)
        pdf.set_text_color(*_BLACK)
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(2)

    def body_text(text: str, bold: bool = False, size: int = 10):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_text_color(*_BLACK)
        try:
            safe = text.encode("latin-1", errors="replace").decode("latin-1")
        except Exception:
            safe = text
        pdf.multi_cell(0, 6, safe)
        pdf.ln(1)

    def metric_bar(label: str, val: float, total: float = 10.0):
        pct = min(1.0, val / total)
        bar_w = 100
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_GREY)
        pdf.cell(60, 5, label)
        # background bar
        pdf.set_fill_color(220, 220, 228)
        pdf.rect(pdf.get_x(), pdf.get_y() + 0.5, bar_w, 4, "F")
        # filled bar
        color = (0, 200, 100) if pct >= 0.7 else (255, 170, 0) if pct >= 0.4 else (220, 60, 60)
        pdf.set_fill_color(*color)
        pdf.rect(pdf.get_x(), pdf.get_y() + 0.5, bar_w * pct, 4, "F")
        pdf.cell(bar_w + 2, 5, "")
        pdf.set_text_color(*color)
        pdf.cell(0, 5, f"{val:.1f}/10", ln=True)
        pdf.set_text_color(*_BLACK)

    # ── Cover ─────────────────────────────────────────────────────────────
    pdf.set_fill_color(*_RED)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*_WHITE)
    pdf.cell(0, 12, "", ln=True)
    pdf.cell(0, 10, "TubeRank AI — SEO Package", align="C", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_WHITE)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}  |  {content_type}", align="C", ln=True)
    pdf.set_text_color(*_BLACK)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*_RED)
    pdf.multi_cell(0, 7, f"Topic: {topic}")
    pdf.set_text_color(*_BLACK)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(3)

    # ── SEO Report Card ───────────────────────────────────────────────────
    if report_card:
        section_header("📊 SEO Report Card")
        grade = report_card.get("grade", "B")
        gc = _grade_color(grade)
        pdf.set_font("Helvetica", "B", 36)
        pdf.set_text_color(*gc)
        pdf.cell(0, 16, grade, align="C", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_GREY)
        pdf.cell(0, 5, f"Overall Score: {report_card.get('overall_score', 0):.1f}/10", align="C", ln=True)
        pdf.ln(3)
        metrics = [
            ("Title Strength", report_card.get("title_strength", 0)),
            ("Keyword Coverage", report_card.get("keyword_coverage", 0)),
            ("Description Hook", report_card.get("description_hook", 0)),
            ("Tag Diversity", report_card.get("tag_diversity", 0)),
            ("Tag Budget Used", report_card.get("tag_fill", 0)),
        ]
        for label, val in metrics:
            metric_bar(label, val)
            pdf.ln(2)
        pdf.ln(3)

    # ── Titles ────────────────────────────────────────────────────────────
    titles = result.get("titles", [])
    if titles:
        section_header("🏆 A/B Titles with Analytics")
        for i, title in enumerate(titles, 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*_BLACK)
            pdf.multi_cell(0, 6, f"Option {i}: {title}")
            if title_scores and i <= len(title_scores):
                s = title_scores[i - 1]
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(*_GREY)
                pdf.cell(0, 5,
                    f"   CTR: {s.get('ctr_score',0)}/10  |  SEO: {s.get('seo_score',0)}/10  "
                    f"|  {s.get('char_count',0)} chars  |  Hook: {s.get('hook_type','')}  "
                    f"|  {s.get('feedback','')}",
                    ln=True,
                )
            pdf.ln(2)

    # ── Description ───────────────────────────────────────────────────────
    desc = result.get("description", "")
    if desc:
        section_header("📄 Optimized Description")
        body_text(desc)

    # ── Tags ──────────────────────────────────────────────────────────────
    tags = result.get("tags", [])
    if tags:
        section_header("🏷️ SEO Tags")
        body_text(", ".join(tags))
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*_GREY)
        pdf.cell(0, 5, f"Tag character count: {len(', '.join(tags))}/500", ln=True)
        pdf.ln(2)

    # ── Timestamps ────────────────────────────────────────────────────────
    timestamps = result.get("timestamps", [])
    if timestamps and content_type == "Long-Form Video":
        section_header("⏱️ Timestamps")
        for ts in timestamps:
            body_text(f"{ts.get('time', '')}  {ts.get('label', '')}")

    # ── Social Posts ──────────────────────────────────────────────────────
    social = result.get("social_posts", {})
    if social:
        section_header("📱 Social Media Posts")
        for platform, post in social.items():
            if post:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(*_RED)
                pdf.cell(0, 5, platform.upper(), ln=True)
                pdf.set_text_color(*_BLACK)
                body_text(post, size=9)

    # ── Footer ────────────────────────────────────────────────────────────
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*_GREY)
    pdf.cell(0, 5, "Generated by TubeRank AI — Powered by Gemini + LangGraph", align="C")

    buf = BytesIO()
    buf.write(bytes(pdf.output()))
    return buf.getvalue()
