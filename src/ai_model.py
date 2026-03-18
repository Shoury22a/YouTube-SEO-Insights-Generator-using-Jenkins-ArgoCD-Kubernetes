"""
AI Model Module — YouTube SEO Insights Generator.
Backend: Google Gemini via LangChain (langchain-google-genai)

LangChain integration provides:
  - ChatGoogleGenerativeAI: unified LLM client with automatic fallbacks
  - PydanticOutputParser (langchain-core): guaranteed structured JSON output every time
  - RecursiveCharacterTextSplitter + manual map-reduce: long-transcript handling
  - ChatPromptTemplate: clean, maintainable prompt management

Generates structured SEO metadata:
  - 3-5 Clickable A/B Titles
  - Optimized video Description
  - Formatted Timestamps (long-form only)
  - 15-20 SEO Tags (capped at 500 chars)
  - Social Media Posts (Twitter/X, LinkedIn, Instagram)
  - Thumbnail Concept Ideas
  - Niche Analysis & Contrarian Titles
"""

import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

PRIMARY_MODEL       = "gemini-2.0-flash"
FALLBACK_PRO_MODEL  = "gemini-1.5-pro-latest"
FALLBACK_FLASH_MODEL = "gemini-1.5-flash"
FALLBACK_8B_MODEL    = "gemini-1.5-flash-8b"
SUMMARY_MODEL       = "gemini-2.0-flash-lite"


# ---------------------------------------------------------------------------
# Pydantic Output Schema — guarantees structured JSON every time
# ---------------------------------------------------------------------------

class NicheAnalysis(BaseModel):
    saturation_score: int = Field(description="Integer 1-10, 10 = most crowded")
    competition_level: str = Field(description="One of: 'Low', 'Medium', 'High'")
    recommendation: str = Field(description="1-2 sentence actionable advice")


class TimestampEntry(BaseModel):
    time: str = Field(description="Timestamp in MM:SS or H:MM:SS format")
    label: str = Field(description="Chapter label")


class SocialPosts(BaseModel):
    twitter: str = Field(description="Tweet ≤280 characters")
    linkedin: str = Field(description="Professional LinkedIn post")
    instagram: str = Field(description="Instagram caption with hashtags")


class SEOOutput(BaseModel):
    titles: list[str] = Field(description="3-5 A/B title options")
    description: str = Field(description="200-350 word optimised description")
    timestamps: list[TimestampEntry] = Field(description="Chapter timestamps, empty [] if no chapter notes")
    tags: list[str] = Field(description="15-20 SEO tags, total ≤500 chars when joined")
    social_posts: SocialPosts
    thumbnail_ideas: list[str] = Field(description="3 vivid thumbnail concepts")
    niche_analysis: NicheAnalysis
    contrarian_titles: list[str] = Field(
        description="2 contrarian titles if competitor context provided, else []"
    )


# ---------------------------------------------------------------------------
# LLM Client factory — with automatic fallback via LangChain
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise APIException(
            "GOOGLE_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/app/apikey "
            "and add it to the .env file.",
            sys,
        )
    return api_key


def _build_llm_with_fallback() -> ChatGoogleGenerativeAI:
    """
    Returns a robust LangChain LLM chain with multiple automatic fallbacks.
    If the primary model hits a quota/rate-limit (429), LangChain 
    transparently retries with the next model in the chain.
    """
    api_key = _get_api_key()

    def _make_llm(name: str, tokens: int = 4096) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=name,
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=tokens,
        )

    primary  = _make_llm(PRIMARY_MODEL)
    f1_flash = _make_llm(FALLBACK_FLASH_MODEL)
    f2_pro   = _make_llm(FALLBACK_PRO_MODEL)
    f3_8b    = _make_llm(FALLBACK_8B_MODEL)

    # Relay race: 2.0 Flash -> 1.5 Flash -> 1.5 Pro -> 1.5 Flash-8B
    return primary.with_fallbacks([f1_flash, f2_pro, f3_8b])


def _build_summary_llm_with_fallback() -> ChatGoogleGenerativeAI:
    """
    Returns the summary LLM with its own fallback chain.
    """
    api_key = _get_api_key()

    def _make_llm(name: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=name,
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=700,
        )

    primary  = _make_llm(SUMMARY_MODEL)
    fallback = _make_llm(FALLBACK_FLASH_MODEL)

    return primary.with_fallbacks([fallback])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(text.split())


def _summarise_transcript(transcript: str) -> str:
    """
    Map-reduce summarization using LangChain's RecursiveCharacterTextSplitter.

    Flow:
      1. Split the long transcript into overlapping chunks.
      2. MAP: summarize each chunk individually with the summary LLM.
      3. REDUCE: combine all chunk summaries into one concise bullet-point summary.

    Falls back to simple truncation on any error.
    """
    logger.info(
        f"Transcript too long ({_count_words(transcript)} words). "
        "Running LangChain map-reduce summarisation pipeline."
    )
    try:
        llm = _build_summary_llm_with_fallback()

        # Split transcript into manageable chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
        )
        chunks = splitter.split_text(transcript[:MAX_TRANSCRIPT_CHARS])

        # MAP: summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarising chunk {i + 1}/{len(chunks)}...")
            map_prompt = (
                "Extract the key points, main themes, and important facts from the text below. "
                "Be concise. Output bullet points only.\n\n"
                f"TEXT:\n{chunk}"
            )
            response = llm.invoke(map_prompt)
            chunk_summaries.append(response.content)

        # REDUCE: combine all chunk summaries
        combined = "\n\n".join(chunk_summaries)
        reduce_prompt = (
            "You have the following bullet-point summaries from sections of a long video transcript. "
            "Combine them into a single, coherent 400-word summary that captures the key themes, "
            "main points, and important facts from the entire video.\n\n"
            f"SUMMARIES:\n{combined}"
        )
        final_response = llm.invoke(reduce_prompt)
        result = final_response.content.strip()
        logger.info("Transcript map-reduce summarisation complete.")
        return result

    except Exception as e:
        logger.warning(f"Summarisation failed: {e}. Falling back to truncated transcript.")
        return " ".join(transcript.split()[:3000])


def _build_system_prompt(content_type: str, output_language: str) -> str:
    """Returns the system prompt string based on content type and language."""
    if output_language.lower() == "hinglish":
        lang_instruction = (
            "You MUST write ALL output in Hinglish (a mix of Hindi and English using "
            "the Roman/English script). Use conversational, trendy Hinglish common on Indian YouTube."
        )
    elif output_language.lower() != "english":
        lang_instruction = f"You MUST write ALL output in {output_language}, regardless of input language."
    else:
        lang_instruction = ""

    if content_type == "YouTube Short":
        return (
            "You are an elite YouTube Shorts copywriter and SEO specialist.\n"
            "Generate viral, clickable metadata for a YouTube Short.\n\n"
            "SHORTS CONSTRAINTS:\n"
            "- Titles MUST be 45 characters or fewer.\n"
            "- 'timestamps' array MUST be empty [].\n"
            "- Description under 100 words.\n"
            "- Tags highly trend-focused and specific.\n"
            "- Social posts: energetic, short-form hooks.\n"
            "- No keyword stuffing. Natural prose only.\n"
            "- Always populate niche_analysis.\n"
            "- contrarian_titles: 2 if competitor context provided, else [].\n\n"
            + lang_instruction
        )

    return (
        "You are a professional YouTube SEO copywriter and content strategist with deep "
        "expertise in YouTube's search algorithm and creator growth.\n\n"
        "Your task: generate a complete, search-optimised SEO metadata package for a long-form YouTube video.\n\n"
        "GUIDELINES:\n"
        "- Titles: Write 3-5 distinct titles with different emotional hooks (curiosity, urgency, authority, FOMO).\n"
        "- Description: 200-350 words. Weave keywords naturally into flowing prose. First 2 lines must hook the viewer.\n"
        "- Timestamps: Generate ONLY if chapter notes are provided, else return [].\n"
        "  Times MUST be in ascending order (0:00, 1:30, 3:45…). NEVER invent timecodes.\n"
        "- Tags: Generate 15-20 diverse tags mixing broad, niche, and long-tail keywords.\n"
        "  The total joined character count of all tags MUST stay under 500 characters.\n"
        "- Social posts: Twitter = conversational ≤280 chars. LinkedIn = professional insight.\n"
        "  Instagram = visual, energetic, rich in hashtags.\n"
        "- Thumbnail ideas: vivid, specific visual concepts — describe text overlay, colours, expressions.\n"
        "- Niche Analysis: Assess how crowded this topic is. Give saturation_score (1-10), "
        "competition_level, and a concrete recommendation.\n"
        "- Contrarian Titles: If competitor context is provided, generate exactly 2 titles that take "
        "the OPPOSITE or most provocative angle. If no competitor context, return [].\n\n"
        "Strictly no keyword stuffing. All keywords must appear naturally within sentences.\n"
        + lang_instruction
    )


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
        # ts may be a TimestampEntry Pydantic object or a plain dict
        t_time  = ts.time  if hasattr(ts, "time")  else ts.get("time", "")
        t_label = ts.label if hasattr(ts, "label") else ts.get("label", "")
        s = _to_secs(t_time)
        if s > last:
            valid.append({"time": t_time, "label": t_label})
            last = s
        else:
            logger.warning(f"Dropping out-of-order timestamp: {t_time} {t_label}")
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
    Generates a complete YouTube SEO metadata package using LangChain + Gemini.

    LangChain powers:
      - Structured output via PydanticOutputParser (SEOOutput Pydantic model)
      - Automatic model fallback via .with_fallbacks()
      - Long-transcript map-reduce summarization via RecursiveCharacterTextSplitter
      - Clean prompt management via ChatPromptTemplate

    Returns:
        dict with keys: titles, description, timestamps, tags,
                        social_posts, thumbnail_ideas, niche_analysis, contrarian_titles
    """
    logger.info(f"Generating SEO metadata | type={content_type} | lang={output_language}")

    # ── 1. Transcript handling ────────────────────────────────────────────────
    transcript_text = (transcript or "").strip()
    if transcript_text and _count_words(transcript_text) > MAX_TRANSCRIPT_WORDS:
        transcript_text = _summarise_transcript(transcript_text)
    elif transcript_text:
        transcript_text = transcript_text[:MAX_TRANSCRIPT_CHARS]

    # ── 2. Build LLM with automatic fallback ─────────────────────────────────
    try:
        llm = _build_llm_with_fallback()
    except APIException:
        raise
    except Exception as e:
        raise APIException(f"Failed to initialise LangChain LLM: {e}", sys) from e

    # ── 3. Output parser — guarantees structured JSON via Pydantic ───────────
    parser = PydanticOutputParser(pydantic_object=SEOOutput)
    format_instructions = parser.get_format_instructions()

    # ── 4. Prompt template ────────────────────────────────────────────────────
    system_prompt_text = _build_system_prompt(content_type, output_language)

    human_parts = [
        f"**Core Topic:** {topic}",
        f"**Target Audience:** {audience}",
    ]
    if transcript_text:
        human_parts.append(f"**Transcript / Script Summary:**\n{transcript_text}")
    elif (visual_description or "").strip():
        human_parts.append(f"**Visual Description (no-speech content):**\n{visual_description.strip()}")
    if (chapter_notes or "").strip():
        human_parts.append(f"**Chapter Notes (for timestamps):**\n{chapter_notes.strip()}")
    if (competitor_context or "").strip():
        human_parts.append(f"**Competitor Reference (for context only):**\n{competitor_context.strip()}")
    human_parts.append(
        "Generate the complete SEO metadata package now.\n\n"
        f"{format_instructions}"
    )
    human_message = "\n\n".join(human_parts)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system}"),
        HumanMessagePromptTemplate.from_template("{human}"),
    ])

    # ── 5. Chain: Prompt → LLM (with fallback) → PydanticOutputParser ────────
    chain = prompt | llm | parser

    try:
        logger.info("Invoking LangChain chain: Prompt → Gemini (with fallback) → PydanticOutputParser")
        t0 = time.time()
        seo: SEOOutput = chain.invoke({
            "system": system_prompt_text,
            "human": human_message,
        })
        logger.info(f"LangChain chain completed in {time.time() - t0:.2f}s.")

    except Exception as e:
        err_str = str(e)
        err_lower = err_str.lower()

        if any(x in err_lower for x in ["quota", "resource_exhausted", "429", "rate limit"]):
            raise APIException(
                "All AI models are currently at their free-tier quota limits. "
                "The LangChain fallback chain (4 models deep) has been exhausted. "
                "Please wait 60 seconds and try again. "
                f"Details: {err_str}", sys
            ) from e

        raise APIException(f"LangChain chain error: {err_str}", sys) from e

    # ── 6. Convert SEOOutput Pydantic model → plain dict ─────────────────────
    # Normalize timestamps
    normalized_timestamps = []
    for ts in (seo.timestamps or []):
        if hasattr(ts, "time"):
            normalized_timestamps.append({"time": ts.time, "label": ts.label})
        elif isinstance(ts, dict):
            normalized_timestamps.append(ts)

    # Normalize social_posts
    sp = seo.social_posts
    if hasattr(sp, "twitter"):
        social_dict = {"twitter": sp.twitter, "linkedin": sp.linkedin, "instagram": sp.instagram}
    else:
        social_dict = dict(sp)

    # Normalize niche_analysis
    na = seo.niche_analysis
    if hasattr(na, "saturation_score"):
        niche_dict = {
            "saturation_score": na.saturation_score,
            "competition_level": na.competition_level,
            "recommendation": na.recommendation,
        }
    else:
        niche_dict = dict(na)

    data = {
        "titles": list(seo.titles),
        "description": seo.description,
        "timestamps": normalized_timestamps,
        "tags": list(seo.tags),
        "social_posts": social_dict,
        "thumbnail_ideas": list(seo.thumbnail_ideas),
        "niche_analysis": niche_dict,
        "contrarian_titles": list(seo.contrarian_titles),
    }

    # ── 7. Post-processing (same as before) ───────────────────────────────────
    data["timestamps"] = _validate_timestamps(data["timestamps"])
    data["tags"]       = _enforce_tag_limit(data["tags"])

    if content_type == "YouTube Short":
        data["titles"]     = _enforce_short_titles(data["titles"])
        data["timestamps"] = []

    logger.info("SEO metadata generation complete.")
    return data
