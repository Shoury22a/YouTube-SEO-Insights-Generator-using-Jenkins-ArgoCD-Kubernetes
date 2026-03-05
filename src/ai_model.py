"""
AI Model Module — YouTube SEO Insights Generator.
Backend: Google Gemini 2.0 Flash (FREE — 1M tokens/day)

Generates structured SEO metadata:
  - 3-5 Clickable A/B Titles
  - Optimized video Description
  - Formatted Timestamps (long-form only)
  - 15-20 SEO Tags (capped at 500 chars)
  - Social Media Posts (Twitter/X, LinkedIn, Instagram)
  - Thumbnail Concept Ideas

Edge-case handling:
  - Massive transcripts → Summarization pipeline (Flash 8B first)
  - Timestamp hallucination → Chronological validation
  - Tag limit → Auto-truncation to 500 chars
  - Shorts → Separate prompt (no timestamps, ≤45 char titles)
  - Multi-language → Output language injected into system prompt
  - Bad LLM JSON → Wrapped parsing with ValidationException
"""

import json
import os
import re
import sys
import time
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from src.logger import get_logger
from src.exception import APIException, ValidationException

load_dotenv()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TRANSCRIPT_WORDS = 5_000
MAX_TRANSCRIPT_CHARS = 25_000
MAX_TAG_CHARS        = 500
SHORT_TITLE_MAX      = 45

PRIMARY_MODEL = "gemini-flash-latest"   # Pointed to working model
FALLBACK_MODEL = "gemini-pro-latest"      # Fallback to pro if flash hits limits
SUMMARY_MODEL = "gemini-flash-lite-latest" # Faster model for summarisation

_JSON_SCHEMA = """
Respond ONLY with a valid JSON object matching this schema exactly:
{
  "titles": ["<title_1>", "<title_2>", "<title_3>", "<title_4>", "<title_5>"],
  "description": "<optimised description as a single string, use \\n for line breaks>",
  "timestamps": [{"time": "0:00", "label": "<chapter label>"}],
  "tags": ["<tag_1>", ..., "<tag_20>"],
  "social_posts": {
    "twitter": "<tweet ≤280 chars>",
    "linkedin": "<LinkedIn post>",
    "instagram": "<Instagram caption with hashtags>"
  },
  "thumbnail_ideas": ["<idea_1>", "<idea_2>", "<idea_3>"],
  "niche_analysis": {
    "saturation_score": 7,
    "competition_level": "High",
    "recommendation": "<1-2 sentence actionable advice, e.g. suggest a sub-niche angle>"
  },
  "contrarian_titles": []
}
Rules:
- niche_analysis.saturation_score: integer 1-10 (10=most crowded).
- niche_analysis.competition_level: exactly one of 'Low', 'Medium', 'High'.
- contrarian_titles: array of 2 strings that challenge the dominant angle in the competitor context. If no competitor context provided, return an empty array [].
Output ONLY the raw JSON. No markdown fences, no explanation outside the JSON.
"""

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def _configure_genai() -> None:
    """Configures the google-generativeai SDK with the API key from env."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise APIException(
            "GOOGLE_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/app/apikey "
            "and add it to the .env file.",
            sys,
        )
    genai.configure(api_key=api_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(text.split())


def _summarise_transcript(transcript: str) -> str:
    """Uses Gemini Flash Lite to compress a long transcript into key themes."""
    logger.info(
        f"Transcript too long ({_count_words(transcript)} words). "
        "Running summarisation pipeline with gemini-2.0-flash-lite."
    )
    try:
        model = genai.GenerativeModel(SUMMARY_MODEL)
        response = model.generate_content(
            "You are a concise summariser. Extract the key themes, main points, "
            "and important facts from the video transcript below. "
            "Output a structured bullet-point summary of 500 words or less.\n\n"
            + transcript[:MAX_TRANSCRIPT_CHARS],
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=700,
            ),
        )
        summary = response.text.strip()
        logger.info("Transcript summarisation complete.")
        return summary
    except Exception as e:
        logger.warning(f"Summarisation failed: {e}. Falling back to truncated transcript.")
        return " ".join(transcript.split()[:3000])


def _build_system_prompt(content_type: str, output_language: str) -> str:
    if output_language.lower() == "hinglish":
        lang_instruction = "You MUST write ALL output in Hinglish (a mix of Hindi and English using the Roman/English script). Use conversational, trendy Hinglish common on Indian YouTube."
    elif output_language.lower() != "english":
        lang_instruction = f"You MUST write ALL output in {output_language}, regardless of input language."
    else:
        lang_instruction = ""

    if content_type == "YouTube Short":
        return f"""You are an elite YouTube Shorts copywriter and SEO specialist.
Generate viral, clickable metadata for a YouTube Short.

SHORTS CONSTRAINTS:
- Titles MUST be 45 characters or fewer.
- "timestamps" array MUST be empty [].
- Description under 100 words.
- Tags highly trend-focused and specific.
- Social posts: energetic, short-form hooks.
- No keyword stuffing. Natural prose only.
- Always populate niche_analysis with saturation_score, competition_level, and recommendation.
- contrarian_titles: 2 contrarian titles if competitor context provided, else [].

{lang_instruction}
{_JSON_SCHEMA}"""

    return f"""You are a professional YouTube SEO copywriter and content strategist with deep
expertise in YouTube's search algorithm and creator growth.

Your task: generate a complete, search-optimised SEO metadata package for a long-form YouTube video.

GUIDELINES:
- Titles: Write 3-5 distinct titles with different emotional hooks (curiosity, urgency, authority, FOMO).
- Description: 200-350 words. Weave keywords naturally into flowing prose. First 2 lines must hook the viewer.
- Timestamps: Generate ONLY if chapter notes are provided, else return [].
  Times MUST be in ascending order (0:00, 1:30, 3:45…). NEVER invent timecodes.
- Tags: Generate 15-20 diverse tags mixing broad, niche, and long-tail keywords.
  The total joined character count of all tags MUST stay under 500 characters.
- Social posts: Twitter = conversational ≤280 chars. LinkedIn = professional insight.
  Instagram = visual, energetic, rich in hashtags.
- Thumbnail ideas: vivid, specific visual concepts — describe text overlay, colours, expressions.
- Niche Analysis: Based on your knowledge of the YouTube content landscape, assess how crowded this
  topic is. Give a saturation_score (1-10), a competition_level, and a concrete recommendation
  (e.g. 'Niche down to iPhone 15 battery life comparisons' or 'Low competition — publish broadly').
- Contrarian Titles: If competitor context is provided, generate exactly 2 titles that take the
  OPPOSITE or most provocative angle to the competitor's title/approach. These should be bold,
  disruptive, and designed to stand out. If no competitor context provided, return [].

Strictly no keyword stuffing. All keywords must appear naturally within sentences.
{lang_instruction}
{_JSON_SCHEMA}"""


def _build_user_prompt(
    topic: str,
    audience: str,
    transcript_or_summary: str,
    visual_description: str,
    chapter_notes: str,
    competitor_context: str,
) -> str:
    parts = [f"**Core Topic:** {topic}", f"**Target Audience:** {audience}"]
    if transcript_or_summary:
        parts.append(f"**Transcript / Script Summary:**\n{transcript_or_summary}")
    elif visual_description:
        parts.append(f"**Visual Description (no-speech content):**\n{visual_description}")
    if chapter_notes:
        parts.append(f"**Chapter Notes (for timestamps):**\n{chapter_notes}")
    if competitor_context:
        parts.append(f"**Competitor Reference (for context only):**\n{competitor_context}")
    parts.append("Generate the complete SEO metadata package now. Return ONLY valid JSON.")
    return "\n\n".join(parts)


def _extract_json(raw: str) -> str:
    """Strip markdown fences and conversational filler."""
    raw = raw.strip()
    # Find the first '{' and last '}'
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        return raw[start:end+1]
    return raw


def _validate_timestamps(timestamps: list) -> list:
    if not timestamps:
        return []

    def _to_secs(t: str) -> int:
        try:
            parts = list(map(int, t.strip().split(":")))
            return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
        except Exception:
            return -1

    valid, last = [], -1
    for ts in timestamps:
        s = _to_secs(ts.get("time", ""))
        if s > last:
            valid.append(ts)
            last = s
        else:
            logger.warning(f"Dropping out-of-order timestamp: {ts}")
    return valid


def _enforce_tag_limit(tags: list) -> list:
    result, total = [], 0
    for tag in tags:
        cost = len(tag) + (2 if result else 0)  # ", " separator
        if total + cost <= MAX_TAG_CHARS:
            result.append(tag)
            total += cost
        else:
            logger.info(f"Tag limit reached ({total} chars). Dropping: '{tag}'")
    return result


def _enforce_short_titles(titles: list) -> list:
    result = []
    for t in titles:
        if len(t) > SHORT_TITLE_MAX:
            trimmed = t[:SHORT_TITLE_MAX].rsplit(" ", 1)[0]
            logger.warning(f"Short title trimmed: '{t}' → '{trimmed}'")
            result.append(trimmed)
        else:
            result.append(t)
    return result


def _call_gemini_with_retry(
    prompt_parts: list,
    max_retries: int = 3,
) -> str:
    """Calls Gemini with exponential backoff on quota/rate errors."""
    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        model = genai.GenerativeModel(model_name)
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Calling {model_name} (attempt {attempt}/{max_retries})...")
                t0 = time.time()
                response = model.generate_content(
                    prompt_parts,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 4096,
                        "response_mime_type": "application/json",
                    },
                )
                logger.info(f"{model_name} responded in {time.time() - t0:.2f}s.")
                return response.text.strip()

            except Exception as e:
                real_error = str(e)
                err_lower = real_error.lower()
                logger.error(f"Gemini error on attempt {attempt} with {model_name}: {real_error}")

                # True rate limit — retry with backoff
                if "429" in real_error or ("quota" in err_lower and "resource_exhausted" in err_lower):
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit. Retrying in {wait}s...")
                    if attempt == max_retries:
                        break  # Try fallback model
                    time.sleep(wait)

                # Invalid key or permission error — fail immediately
                elif any(x in err_lower for x in ["api key", "api_key", "permission_denied", "403", "invalid"]):
                    raise APIException(
                        f"Invalid GOOGLE_API_KEY. Check your key at https://aistudio.google.com/app/apikey. "
                        f"Details: {real_error}", sys
                    ) from e

                # Model not found — try fallback
                elif "not found" in err_lower or "404" in real_error:
                    logger.warning(f"Model {model_name} not found. Trying fallback.")
                    break  # Move to fallback model

                # Other errors — retry then surface real error
                else:
                    if attempt == max_retries:
                        raise APIException(
                            f"Gemini API error: {real_error}", sys
                        ) from e
                    time.sleep(2)

    raise APIException(
        f"Both {PRIMARY_MODEL} and {FALLBACK_MODEL} failed. "
        "Check your API key and try again.", sys
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_seo_metadata(
    topic: str,
    audience: str,
    content_type: str = "Long-Form Video",
    output_language: str = "English",
    transcript: Optional[str] = None,
    visual_description: Optional[str] = None,
    chapter_notes: Optional[str] = None,
    competitor_context: Optional[str] = None,
) -> dict:
    """
    Generates a complete YouTube SEO metadata package using Gemini 2.0 Flash.

    Returns:
        dict with keys: titles, description, timestamps, tags, social_posts, thumbnail_ideas
    """
    logger.info(f"Generating SEO metadata | type={content_type} | lang={output_language}")
    _configure_genai()

    # Summarisation pipeline for long transcripts
    transcript_text = (transcript or "").strip()
    if transcript_text and _count_words(transcript_text) > MAX_TRANSCRIPT_WORDS:
        transcript_text = _summarise_transcript(transcript_text)
    elif transcript_text:
        transcript_text = transcript_text[:MAX_TRANSCRIPT_CHARS]

    system_prompt = _build_system_prompt(content_type, output_language)
    user_prompt = _build_user_prompt(
        topic=topic,
        audience=audience,
        transcript_or_summary=transcript_text,
        visual_description=(visual_description or "").strip(),
        chapter_notes=(chapter_notes or "").strip(),
        competitor_context=(competitor_context or "").strip(),
    )

    # Gemini takes system and user content as a list
    raw_json = _call_gemini_with_retry([system_prompt, user_prompt])
    raw_json = _extract_json(raw_json)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        # Save the full failed output for debugging
        debug_path = os.path.join("logs", f"failed_json_{int(time.time())}.json")
        try:
            os.makedirs("logs", exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(raw_json)
            logger.error(f"JSON parse error: {e}. Full output saved to {debug_path}")
        except Exception as log_err:
            logger.error(f"Failed to log debug JSON: {log_err}")

        raise ValidationException(
            f"The AI returned malformed data. This can happen with very large transcripts. "
            f"Try again or try with a slightly shorter input. Details: {e}", sys
        ) from e

    # Post-processing
    data["timestamps"] = _validate_timestamps(data.get("timestamps", []))
    data["tags"]       = _enforce_tag_limit(data.get("tags", []))

    if content_type == "YouTube Short":
        data["titles"]     = _enforce_short_titles(data.get("titles", []))
        data["timestamps"] = []

    logger.info("SEO metadata generation complete.")
    return data
