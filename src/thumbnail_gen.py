"""
Thumbnail Generator — AI-Powered Viral Thumbnail Concepts for TubeRank AI.

Generates 4 visual thumbnail mockups using Gemini's image generation,
one per psychological hook style (Curiosity, FOMO, Urgency, Authority).
Falls back to styled text-overlay cards if image generation is unavailable.

Why 4 styles? CTR research shows different audiences click on different
psychological triggers. Giving creators 4 variants lets them A/B test.
"""

import os
import base64
from src.logger import get_logger

logger = get_logger(__name__)

# ── Hook style definitions ────────────────────────────────────────────────
THUMBNAIL_STYLES = {
    "Curiosity Gap": {
        "emoji": "🤔",
        "color": "#7c3aed",
        "prompt_suffix": (
            "suspenseful, half-revealed element, dramatic shadow, question mark motif, "
            "face with shocked or curious expression, blurred background text, cinematic lighting"
        ),
        "tip": "Works best for mystery/how-to content. Viewers must click to resolve the tension.",
    },
    "FOMO": {
        "emoji": "😱",
        "color": "#dc2626",
        "prompt_suffix": (
            "bold numbers, upward trending arrow, crowd or viral visual, energetic, "
            "bright colors, social proof, trending badge, high contrast"
        ),
        "tip": "Best for listicles and trending topic videos. Creates fear of missing out.",
    },
    "Authority": {
        "emoji": "👑",
        "color": "#0ea5e9",
        "prompt_suffix": (
            "clean professional layout, expert presenter, stats bar or gauge, "
            "badge or award motif, minimal clean background, trustworthy, confident"
        ),
        "tip": "Best for tutorial and educational content. Builds instant credibility.",
    },
    "Urgency": {
        "emoji": "⚡",
        "color": "#f97316",
        "prompt_suffix": (
            "bold warning or exclamation, countdown timer, red accent, high energy, "
            "dramatic action shot, alert style, bright red and black palette"
        ),
        "tip": "Best for news, alerts, and time-sensitive content. Drives immediate clicks.",
    },
}


def generate_thumbnails(
    topic: str,
    thumbnail_concepts: list[str],
    api_key: str | None = None,
) -> list[dict]:
    """
    Generates 4 thumbnail image bytes (one per hook style) using Gemini Imagen.

    Args:
        topic:              The video topic string.
        thumbnail_concepts: List of AI-generated concept strings from the SEO output.
        api_key:            Google API key (reads from env if not provided).

    Returns:
        List of dicts with keys:
            style:      hook style name
            emoji:      display emoji
            color:      brand color hex
            tip:        why this style works
            image_b64:  base64-encoded PNG image (None if generation failed)
            concept:    the text concept used for this thumbnail
    """
    key = api_key or os.getenv("GOOGLE_API_KEY", "")
    results = []

    styles = list(THUMBNAIL_STYLES.items())
    concepts = thumbnail_concepts or []

    for i, (style_name, style_meta) in enumerate(styles):
        # Map the AI's concept to this style (cycle if fewer than 4 concepts)
        concept = concepts[i % len(concepts)] if concepts else f"YouTube thumbnail for: {topic}"

        image_b64 = _generate_single_thumbnail(
            topic=topic,
            concept=concept,
            style_name=style_name,
            style_suffix=style_meta["prompt_suffix"],
            api_key=key,
        )

        results.append({
            "style": style_name,
            "emoji": style_meta["emoji"],
            "color": style_meta["color"],
            "tip": style_meta["tip"],
            "image_b64": image_b64,
            "concept": concept,
        })

    return results


def _generate_single_thumbnail(
    topic: str,
    concept: str,
    style_name: str,
    style_suffix: str,
    api_key: str,
) -> str | None:
    """
    Calls Gemini Imagen 3 to generate a single thumbnail image.
    Returns base64-encoded PNG or None on failure.
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        prompt = (
            f"YouTube video thumbnail, 16:9 aspect ratio, "
            f"topic: '{topic}', "
            f"concept: '{concept[:200]}', "
            f"style: {style_suffix}, "
            f"professional quality, vibrant colors, no text overlays, "
            f"high resolution, attention-grabbing."
        )

        imagen = genai.ImageGenerationModel("imagen-3.0-generate-002")
        response = imagen.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            safety_filter_level="block_only_high",
        )

        if response.images:
            img_bytes = response.images[0]._image_bytes
            return base64.b64encode(img_bytes).decode("utf-8")

    except Exception as e:
        logger.warning(f"Imagen generation failed for '{style_name}': {e}")

    return None
