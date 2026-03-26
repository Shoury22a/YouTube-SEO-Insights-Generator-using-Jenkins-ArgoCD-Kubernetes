"""
Title Analytics Scorer — Deterministic + AI-Aware Scoring for TubeRank AI.

Provides the following for each title:
  - CTR Score (0-10):       Curiosity, power words, numbers, brackets
  - SEO Density Score:      Keyword coverage relative to topic
  - Character Count:        With colour indicator (green/amber/red)
  - Emotional Hook Type:    Curiosity / Urgency / Authority / FOMO / How-To

Also computes the overall SEO Report Card from the full result dict.
"""

import re
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# High-CTR power words — proven to increase click-through rate
_CURIOSITY_WORDS = {
    "secret", "hidden", "truth", "nobody", "untold", "mystery",
    "shocking", "surprising", "actually", "really", "honest",
    "why", "what", "how", "mistake", "warning", "revealed",
}

_URGENCY_WORDS = {
    "now", "today", "stop", "immediately", "fast", "quick",
    "minutes", "hours", "before", "deadline", "limited", "last",
    "urgent", "alert", "breaking", "2024", "2025", "2026",
}

_AUTHORITY_WORDS = {
    "ultimate", "complete", "definitive", "expert", "professional",
    "masterclass", "guide", "tutorial", "course", "learn", "study",
    "proven", "tested", "works", "best", "top", "rank",
}

_FOMO_WORDS = {
    "everyone", "trending", "viral", "must", "need", "missing",
    "without", "before", "too late", "regret", "game-changer",
    "blowing up", "can't", "obsessed", "nobody", "most people",
}

_HOW_TO_PATTERNS = [
    r"^how (to|i|we)",
    r"^(step[- ]by[- ]step|beginners? guide)",
    r"\d+ (ways|steps|tips|tricks|hacks|secrets)",
]

# YouTube character limits
_CHAR_LIMITS = {
    "Long-Form Video": {"green": 50, "amber": 60, "max": 70},
    "YouTube Short": {"green": 35, "amber": 42, "max": 45},
}


# ---------------------------------------------------------------------------
# Per-Title Scoring
# ---------------------------------------------------------------------------

def _count_word_overlap(topic: str, title: str) -> float:
    """Returns 0.0–1.0: fraction of meaningful topic words in the title."""
    stop = {"the","a","an","in","on","to","for","of","and","or","how","what","is","are"}
    t_words = {w.lower() for w in topic.split() if w.lower() not in stop and len(w) > 2}
    if not t_words:
        return 0.0
    ti_words = set(title.lower().split())
    overlap = t_words & ti_words
    return len(overlap) / len(t_words)


def _detect_hook_type(title: str) -> tuple[str, str]:
    """
    Returns (hook_type_label, emoji).
    Checks in priority order: How-To → Curiosity → FOMO → Urgency → Authority → Informational
    """
    title_lower = title.lower()
    words = set(title_lower.split())

    for pattern in _HOW_TO_PATTERNS:
        if re.search(pattern, title_lower):
            return ("How-To", "📖")

    curiosity_hit = words & _CURIOSITY_WORDS
    fomo_hit = words & _FOMO_WORDS
    urgency_hit = words & _URGENCY_WORDS
    authority_hit = words & _AUTHORITY_WORDS

    scores = {
        "Curiosity Gap": len(curiosity_hit),
        "FOMO": len(fomo_hit),
        "Urgency":len(urgency_hit),
        "Authority": len(authority_hit),
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return ("Informational", "📋")

    emoji_map = {
        "Curiosity Gap": "🤔",
        "FOMO": "😱",
        "Urgency": "⚡",
        "Authority": "👑",
    }
    return (best, emoji_map[best])


def _has_number(title: str) -> bool:
    """True if the title contains a digit (listicle effect)."""
    return bool(re.search(r"\d", title))


def _has_brackets(title: str) -> bool:
    """True if the title has brackets/parentheses — boosts CTR ~33%."""
    return bool(re.search(r"[\[\(\{]", title))


def score_title(title: str, topic: str, content_type: str = "Long-Form Video") -> dict:
    """
    Returns full analytics for a single title.

    Structure:
        ctr_score:     int 0-10
        seo_score:     int 0-10
        char_count:    int
        char_status:   "green" | "amber" | "red"
        hook_type:     str
        hook_emoji:    str
        power_words:   list[str]  matched power words
        has_number:    bool
        has_brackets:  bool
        feedback:      str  — one sentence actionable tip
    """
    title_lower = title.lower()
    words = set(title_lower.split())
    limits = _CHAR_LIMITS.get(content_type, _CHAR_LIMITS["Long-Form Video"])

    # --- CTR Score (0-10) ------------------------------------------------
    ctr = 0

    # Power word presence (+2 each category, max 6)
    if words & _CURIOSITY_WORDS: ctr += 2
    if words & _URGENCY_WORDS:   ctr += 2
    if words & _FOMO_WORDS:      ctr += 2
    if words & _AUTHORITY_WORDS: ctr += 1

    # Structural boosts
    if _has_number(title):    ctr += 1  # Numbers increase CTR
    if _has_brackets(title):  ctr += 1  # Brackets increase CTR
    if len(title) >= 30:      ctr += 1  # Minimum descriptiveness

    ctr = min(10, ctr)

    # --- SEO Score (0-10) ------------------------------------------------
    overlap = _count_word_overlap(topic, title)
    seo = round(overlap * 10)
    seo = max(1, min(10, seo))

    # --- Character Count ------------------------------------------------
    char_count = len(title)
    if char_count <= limits["green"]:
        char_status = "green"
    elif char_count <= limits["amber"]:
        char_status = "amber"
    else:
        char_status = "red"

    # --- Hook Type -------------------------------------------------------
    hook_type, hook_emoji = _detect_hook_type(title)

    # --- Matched power words for display ---------------------------------
    all_power = _CURIOSITY_WORDS | _URGENCY_WORDS | _AUTHORITY_WORDS | _FOMO_WORDS
    power_words_found = sorted(words & all_power)

    # --- Feedback tip ----------------------------------------------------
    feedback_parts = []
    if seo < 5:
        feedback_parts.append(f"Add a keyword from your topic ('{topic[:30]}')")
    if ctr < 5:
        feedback_parts.append("Add a power word (e.g. 'Why', 'Secret', 'Actually')")
    if not _has_number(title):
        feedback_parts.append("Consider adding a number (e.g. '7 Tips')")
    if char_status == "red":
        feedback_parts.append(f"Shorten to ≤{limits['amber']} characters")
    if char_status == "green" and ctr >= 7 and seo >= 7:
        feedback_parts.append("Excellent — high CTR and SEO alignment ✅")

    feedback = " • ".join(feedback_parts) if feedback_parts else "Well-optimised title."

    return {
        "ctr_score": ctr,
        "seo_score": seo,
        "char_count": char_count,
        "char_status": char_status,
        "hook_type": hook_type,
        "hook_emoji": hook_emoji,
        "power_words": power_words_found,
        "has_number": _has_number(title),
        "has_brackets": _has_brackets(title),
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# SEO Report Card
# ---------------------------------------------------------------------------

def compute_report_card(result: dict, topic: str, content_type: str = "Long-Form Video") -> dict:
    """
    Computes the full SEO Report Card from a generation result dict.

    Metrics:
        title_strength:      avg CTR score across all titles
        tag_diversity:       ratio of unique tags (no near-duplicates)
        description_hook:    checks first 2 sentences for hook words
        tag_fill:            tag character usage vs 500-char limit
        keyword_coverage:    how many titles contain the core keyword
        overall_grade:       A+ / A / B+ / B / C
    """
    titles = result.get("titles", [])
    tags = result.get("tags", [])
    description = result.get("description", "")

    # -- Title Strength (average CTR score) --
    if titles:
        ctr_scores = [score_title(t, topic, content_type)["ctr_score"] for t in titles]
        title_strength = round(sum(ctr_scores) / len(ctr_scores), 1)
    else:
        title_strength = 0.0

    # -- Tag Diversity (unique vs total) --
    if tags:
        unique_tags = len(set(t.lower() for t in tags))
        tag_diversity = round((unique_tags / len(tags)) * 10, 1)
    else:
        tag_diversity = 0.0

    # -- Description Hook (first 2 sentences have hook words?) --
    first_two = ". ".join(description.split(".")[:2]).lower()
    hook_words_in_desc = (
        set(first_two.split()) & (_CURIOSITY_WORDS | _URGENCY_WORDS | _FOMO_WORDS)
    )
    description_hook = min(10.0, len(hook_words_in_desc) * 2.5)

    # -- Tag Fill (character budget utilisation) --
    tag_chars = len(", ".join(tags))
    if tag_chars > 500:
        tag_fill = 5.0  # Over limit — penalise
    else:
        tag_fill = round((tag_chars / 500) * 10, 1)

    # -- Keyword Coverage (% of titles containing a topic keyword) --
    topic_kws = [w.lower() for w in topic.split() if len(w) > 3]
    if titles and topic_kws:
        covered = sum(
            1 for t in titles
            if any(kw in t.lower() for kw in topic_kws)
        )
        keyword_coverage = round((covered / len(titles)) * 10, 1)
    else:
        keyword_coverage = 0.0

    # -- Overall Score & Grade --
    overall = (
        title_strength * 0.35
        + tag_diversity * 0.20
        + description_hook * 0.20
        + keyword_coverage * 0.15
        + tag_fill * 0.10
    )
    overall = round(overall, 1)

    if overall >= 9.0:
        grade = "A+"
    elif overall >= 8.0:
        grade = "A"
    elif overall >= 7.0:
        grade = "B+"
    elif overall >= 6.0:
        grade = "B"
    elif overall >= 5.0:
        grade = "C+"
    else:
        grade = "C"

    return {
        "title_strength": title_strength,
        "tag_diversity": tag_diversity,
        "description_hook": description_hook,
        "tag_fill": tag_fill,
        "keyword_coverage": keyword_coverage,
        "overall_score": overall,
        "grade": grade,
    }
