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
import requests
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import get_logger
from src.exception import APIException

from langchain_groq import ChatGroq
import random

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
FALLBACK_FLASH_MODEL = "gemini-flash-latest"  # Confirmed Working Fallback
FALLBACK_PRO_MODEL  = "gemini-pro-latest"
FALLBACK_8B_MODEL    = "gemini-flash-lite-latest"
SUMMARY_MODEL       = "gemini-flash-latest"
GROQ_MODEL           = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Pydantic Output Schema — guarantees structured JSON every time
# ---------------------------------------------------------------------------

class NicheAnalysis(BaseModel):
    saturation_score: int = Field(default=5, description="Integer 1-10, 10 = most crowded")
    competition_level: str = Field(default="Medium", description="One of: 'Low', 'Medium', 'High'")
    recommendation: str = Field(default="Focus on unique hook angles to stand out.", description="1-2 sentence actionable advice")


class TimestampEntry(BaseModel):
    time: str = Field(description="Timestamp in MM:SS or H:MM:SS format")
    label: str = Field(description="Chapter label")


class SocialPosts(BaseModel):
    twitter: str = Field(default="", description="Tweet ≤280 characters")
    linkedin: str = Field(default="", description="Professional LinkedIn post")
    instagram: str = Field(default="", description="Instagram caption with hashtags")


class SEOOutput(BaseModel):
    titles: list[str] = Field(default_factory=list, description="3-5 A/B title options")
    description: str = Field(default="", description="200-350 word optimised description")
    timestamps: list[TimestampEntry] = Field(default_factory=list, description="Chapter timestamps, empty [] if no chapter notes")
    tags: list[str] = Field(default_factory=list, description="15-20 SEO tags, total ≤500 chars when joined")
    social_posts: SocialPosts = Field(default_factory=lambda: SocialPosts(twitter="", linkedin="", instagram=""))
    thumbnail_ideas: list[str] = Field(default_factory=list, description="3 vivid thumbnail concepts")
    niche_analysis: NicheAnalysis = Field(default_factory=lambda: NicheAnalysis(saturation_score=5, competition_level="Medium", recommendation=""))
    contrarian_titles: list[str] = Field(
        default_factory=list,
        description="2 contrarian titles if competitor context provided, else []"
    )


# ---------------------------------------------------------------------------
# LLM Client factory — with automatic fallback via LangChain
# ---------------------------------------------------------------------------

def _get_api_keys() -> list[str]:
    """Returns a list of all available Gemini API keys from the environment."""
    keys = []
    # Primary
    p = os.getenv("GOOGLE_API_KEY")
    if p: keys.append(p)
    # Additional keys (GOOGLE_API_KEY_2, 3...)
    for i in range(2, 6):
        k = os.getenv(f"GOOGLE_API_KEY_{i}")
        if k: keys.append(k)
        
    if not keys:
        raise APIException(
            "No GOOGLE_API_KEY found. "
            "Add at least one to your .env file.",
            sys,
        )
    return keys


def _build_llm_with_fallback() -> ChatGoogleGenerativeAI:
    """
    Returns a robust LangChain LLM chain with multiple automatic fallbacks.
    Rotates through multiple API keys and falls back to Groq if provided.
    """
    keys = _get_api_keys()
    groq_key = os.getenv("GROQ_API_KEY")
    
    all_llms = []

    # 1. Build Gemini models for each key (Key 1 -> Key 2 -> Key 3...)
    for key in keys:
        def _make_gemini(name: str, k=key) -> ChatGoogleGenerativeAI:
            return ChatGoogleGenerativeAI(
                model=name,
                google_api_key=k,
                temperature=0.7,
                max_output_tokens=4096,
            )
        
        all_llms.append(_make_gemini(PRIMARY_MODEL))
        all_llms.append(_make_gemini(FALLBACK_FLASH_MODEL))
        all_llms.append(_make_gemini(FALLBACK_PRO_MODEL))

    # 2. Add Groq as the ultimate fallback if key exists
    if groq_key:
        logger.info("Adding Groq as the final fallback provider.")
        groq_llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=groq_key,
            temperature=0.7,
        )
        all_llms.append(groq_llm)

    if not all_llms:
        raise APIException("No models could be initialized.", sys)

    primary = all_llms[0]
    fallbacks = all_llms[1:]
    
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def _build_summary_llm_with_fallback() -> ChatGoogleGenerativeAI:
    """
    Summary LLM with a single key rotation.
    """
    keys = _get_api_keys()
    key = random.choice(keys)
    
    primary = ChatGoogleGenerativeAI(
        model=SUMMARY_MODEL,
        google_api_key=key,
        temperature=0.3,
        max_output_tokens=700,
    )
    
    # Add a fallback to 1.5 flash-lite just in case
    fallback = ChatGoogleGenerativeAI(
        model=FALLBACK_8B_MODEL,
        google_api_key=key,
        temperature=0.3,
        max_output_tokens=700,
    )
    
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
# API Connection Diagnostic (Privacy-Safe)
# ---------------------------------------------------------------------------

def check_api_connection() -> dict:
    """
    Performs a minimal, zero-cost metadata call to verify if the API key
    is valid and has an active quota.
    
    Returns:
        dict with keys: 'status' (bool), 'message' (str), 'details' (str)
    """
    try:
        api_keys = _get_api_keys()
        api_key = api_keys[0] # Just check the first one
        # Minimalist call: just list models (doesn't use quota/cost)
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return {
                "status": True,
                "message": "Connected",
                "details": f"API is ready ({len(api_keys)} keys found)."
            }
        
        err_msg = response.json().get("error", {}).get("message", "Unknown Error")
        return {
            "status": False,
            "message": "Error",
            "details": f"API Rejected: {err_msg}"
        }
    except APIException as e:
        return {"status": False, "message": "Error", "details": str(e)}
    except Exception as e:
        return {"status": False, "message": "Error", "details": f"Connection failed: {e}"}


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

    # ── 1b. Adaptive RAG Retrieval (New for Linear) ─────────────────────────
    if not competitor_context:
        try:
            from src.rag_store import retrieve_similar
            similar = retrieve_similar(topic=topic, k=2, content_type=content_type, language=output_language)
            if similar:
                rag_data = "\n\n--- PAST SUCCESSES ---\n" + "\n".join([s['content'] for s in similar])
                competitor_context = (competitor_context or "") + rag_data
                logger.info("Linear generation enriched with RAG context.")
        except Exception as e:
            logger.warning(f"Linear RAG retrieval failed: {e}")

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
        logger.info("Invoking LangChain chain: Prompt → Gemini/Groq (with fallback) → PydanticOutputParser")
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
                "All AI models (including fallbacks) are currently at their quota limits. "
                "TIP: Add a second API key (GOOGLE_API_KEY_2) or a GROQ_API_KEY to your .env "
                "to increase your daily capacity. "
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

    # Final metadata dictionary
    metadata = {
        "titles": seo.titles,
        "description": seo.description,
        "timestamps": normalized_timestamps,
        "tags": seo.tags,
        "social_posts": {
            "twitter": seo.social_posts.twitter,
            "linkedin": seo.social_posts.linkedin,
            "instagram": seo.social_posts.instagram,
        },
        "thumbnail_ideas": seo.thumbnail_ideas,
        "niche_analysis": {
            "saturation_score": seo.niche_analysis.saturation_score,
            "competition_level": seo.niche_analysis.competition_level,
            "recommendation": seo.niche_analysis.recommendation,
        },
        "contrarian_titles": seo.contrarian_titles,
    }

    # ── 7. Persist to ChromaDB (Memory/History) ─────────────────────────────
    try:
        from src.rag_store import persist_generation
        persist_generation(
            topic=topic,
            seo_bundle=metadata,
            content_type=content_type,
            language=output_language,
        )
    except Exception as e:
        logger.warning(f"Metadata persistence failed in linear path: {e}")

    logger.info("Linear SEO generation complete.")
    return metadata


# ---------------------------------------------------------------------------
# Agentic API — Full LangGraph Pipeline (Researcher → Critic → Refiner)
# ---------------------------------------------------------------------------

def generate_seo_metadata_agentic(
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
    Enhanced SEO generation using the full LangGraph Agent pipeline.

    This wraps the existing generate_seo_metadata() inside a multi-step
    agentic workflow that adds:
      - RAG retrieval (past successes from ChromaDB)
      - Web search for trending topics
      - Automated quality critique (5 benchmarks)
      - Targeted refinement loop (max 1 retry)
      - Persistence to vector store for future use

    Returns:
        dict with keys:
          - All standard SEO fields (titles, description, tags, etc.)
          - _agent_log: list of step descriptions
          - _agent_retries: number of critique-refine loops
          - _agent_elapsed: total time in seconds
          - _rag_count: number of documents retrieved from memory
    """
    # Pre-process transcript (same as the linear version)
    transcript_text = (transcript or "").strip()
    if transcript_text and _count_words(transcript_text) > MAX_TRANSCRIPT_WORDS:
        transcript_text = _summarise_transcript(transcript_text)
    elif transcript_text:
        transcript_text = transcript_text[:MAX_TRANSCRIPT_CHARS]

    try:
        from src.agent import run_seo_agent
        from src.metrics import record_generation

        result = run_seo_agent(
            topic=topic,
            audience=audience,
            content_type=content_type,
            output_language=output_language,
            transcript=transcript_text,
            visual_description=visual_description or "",
            chapter_notes=chapter_notes or "",
            competitor_context=competitor_context or "",
        )

        metadata = result.get("metadata", {})

        # Record Prometheus metrics
        try:
            record_generation(
                elapsed_seconds=result.get("elapsed_seconds", 0),
                retry_count=result.get("retry_count", 0),
                retrieved_count=result.get("retrieved_count", 0),
                rag_eval=result.get("rag_eval", None),
            )
        except Exception as e:
            logger.warning(f"Metrics recording failed: {e}")

        # Attach agent telemetry to the output
        metadata["_agent_log"] = result.get("step_log", [])
        metadata["_agent_retries"] = result.get("retry_count", 0)
        metadata["_agent_elapsed"] = result.get("elapsed_seconds", 0)
        metadata["_rag_count"] = result.get("retrieved_count", 0)
        metadata["_rag_eval"] = result.get("rag_eval", {})

        logger.info("Agentic SEO generation complete.")
        return metadata

    except Exception as e:
        logger.warning(f"Agent pipeline failed: {e}. Falling back to linear generation.")
        # 1. Retrieve RAG context for the linear fallback if possible
        competitor_context = competitor_context or ""
        try:
            from src.rag_store import retrieve_similar
            similar = retrieve_similar(topic=topic, k=2, content_type=content_type, language=output_language)
            if similar:
                rag_data = "\n\n--- PAST SUCCESSES ---\n" + "\n".join([s['content'] for s in similar])
                competitor_context += rag_data
                logger.info("Linear fallback enriched with RAG context.")
        except Exception as rag_err:
            logger.warning(f"Linear RAG retrieval failed: {rag_err}")

        # 2. Run the original linear pipeline
        result = generate_seo_metadata(
            topic=topic,
            audience=audience,
            content_type=content_type,
            output_language=output_language,
            transcript=transcript_text,
            visual_description=visual_description,
            chapter_notes=chapter_notes,
            competitor_context=competitor_context,
        )

        # 3. Persist this linear result to RAG so we 'learn' even from fallbacks
        try:
            from src.rag_store import persist_generation
            persist_generation(topic=topic, seo_bundle=result, content_type=content_type, language=output_language)
        except Exception as persist_err:
            logger.warning(f"Linear RAG persistence failed: {persist_err}")

        result["_agent_log"] = ["⚠️ Agent offline (Quota). Used linear + RAG memory as fallback."]
        result["_agent_retries"] = 0
        result["_agent_elapsed"] = 0
        result["_rag_count"] = 0
        return result
