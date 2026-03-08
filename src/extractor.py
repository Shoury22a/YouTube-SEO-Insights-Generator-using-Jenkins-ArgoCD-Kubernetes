"""
YouTube Data Extractor Module.
Uses yt-dlp for robust metadata extraction (title, description, views, creator, thumbnail).
Implements graceful degradation: if scraping fails, returns None.

Additional utilities:
  - compute_niche_saturation(): Real YouTube search-based competition analysis.
  - compute_contrarian_score(): Mathematical Jaccard divergence scoring.
"""

import re
import yt_dlp
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Stop words for Jaccard scoring — these carry no semantic meaning
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "but", "is", "are", "was", "were", "how", "why", "what", "when",
    "where", "who", "i", "you", "my", "your", "this", "that", "it",
    "be", "do", "can", "will", "just", "not", "with", "from", "by",
    "as", "so", "up", "if", "no", "out", "its", "he", "she", "we",
    "they", "them", "their", "our", "about", "into", "than", "then",
}


def _validate_youtube_url(url: str) -> bool:
    """Validates that the given URL is a legitimate YouTube video link."""
    # Simplified pattern that covers watch?, short/, and youtu.be/
    pattern = r"^(https?://)?(www\.)?(youtube\.com/|youtu\.be/)[\w\-\?=/]{5,}"
    return bool(re.match(pattern, url.strip()))

def extract_video_metadata(url: str) -> dict | None:
    """
    Main entry point for extracting YouTube video metadata using yt-dlp.
    Implements graceful degradation — returns None on failure instead of crashing.
    """
    if not url or not url.strip():
        logger.warning("Empty URL provided to extractor. Skipping.")
        return None

    if not _validate_youtube_url(url):
        logger.warning(f"Invalid YouTube URL format: {url}")
        return None

    logger.info(f"Attempting to extract metadata with yt-dlp: {url}")

    # yt-dlp configuration
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'extract_flat': True, # Don't extract playlist, just the video
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if not info:
                logger.error("yt-dlp returned no info for URL.")
                return None

            metadata = {
                "title": info.get("title", "Unknown Title"),
                "description": info.get("description", ""),
                "creator": info.get("uploader", "Unknown Creator"),
                "views": info.get("view_count"),
                "thumbnail": info.get("thumbnail"),
                "source_url": url,
            }

            logger.info(f"Successfully extracted metadata for: {metadata['title']}")
            return metadata

    except Exception as e:
        logger.error(f"yt-dlp extraction error: {e}")
        return None


# ---------------------------------------------------------------------------
# Feature 5: Niche Saturation — Real YouTube Data Analysis
# ---------------------------------------------------------------------------

def compute_niche_saturation(topic: str, sample_size: int = 15) -> dict:
    """
    Computes a REAL, data-backed niche saturation score by searching YouTube
    for the topic and analysing the view counts of the top results.

    Method:
      1. Uses yt-dlp to fetch the top `sample_size` YouTube search results.
      2. Collects actual view counts from each result.
      3. Derives a 1-10 saturation score from the average views — because if
         existing competitors already have millions of views, the niche is
         crowded and hard to break into.

    Returns:
        dict with saturation_score (int 1-10), competition_level (str),
        avg_views (int), results_analyzed (int), recommendation (str).
    """
    logger.info(f"Running real niche saturation analysis for topic: '{topic}'")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "playlist_items": f"1-{sample_size}",
    }

    view_counts = []
    video_titles = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch{sample_size}:{topic}"
            info = ydl.extract_info(search_query, download=False)
            entries = info.get("entries", []) if info else []

            for entry in entries:
                views = entry.get("view_count") or 0
                title = entry.get("title", "")
                if views > 0:
                    view_counts.append(views)
                if title:
                    video_titles.append(title)

        logger.info(
            f"Niche analysis: {len(view_counts)} results with view data "
            f"out of {len(entries)} total results."
        )
    except Exception as e:
        logger.warning(f"Niche saturation yt-dlp search failed: {e}. Returning fallback.")
        return {
            "saturation_score": 5,
            "competition_level": "Medium",
            "avg_views": 0,
            "results_analyzed": 0,
            "recommendation": "Could not fetch live data. Score is an estimate.",
            "data_source": "fallback",
            "top_video_titles": [],
        }

    if not view_counts:
        return {
            "saturation_score": 3,
            "competition_level": "Low",
            "avg_views": 0,
            "results_analyzed": 0,
            "recommendation": "Very few results found — this niche has very low competition. Great opportunity!",
            "data_source": "live_youtube",
            "top_video_titles": video_titles[:5],
        }

    avg_views = int(sum(view_counts) / len(view_counts))
    max_views = max(view_counts)

    # Score mapping based on average competitor views
    # Rationale: if top videos average 1M+ views, the niche is very established
    # and hard to break into organically.
    if avg_views >= 1_000_000:
        score, level = 9, "Very High"
        rec = (
            f"Extremely competitive — top videos average {avg_views:,} views. "
            "Niche down to a very specific sub-topic (e.g. add location, audience, or time-frame)."
        )
    elif avg_views >= 500_000:
        score, level = 7, "High"
        rec = (
            f"Highly competitive — top videos average {avg_views:,} views. "
            "Consider a unique angle or micro-niche to stand out."
        )
    elif avg_views >= 100_000:
        score, level = 5, "Medium"
        rec = (
            f"Moderate competition — top videos average {avg_views:,} views. "
            "A strong title and thumbnail can still break through."
        )
    elif avg_views >= 10_000:
        score, level = 3, "Low"
        rec = (
            f"Low competition — top videos average {avg_views:,} views. "
            "Great opportunity. Focus on quality and consistency."
        )
    else:
        score, level = 1, "Very Low"
        rec = (
            f"Virtually no competition — top videos average {avg_views:,} views. "
            "First-mover advantage available. Act now."
        )

    return {
        "saturation_score": score,
        "competition_level": level,
        "avg_views": avg_views,
        "max_views": max_views,
        "results_analyzed": len(view_counts),
        "recommendation": rec,
        "data_source": "live_youtube",
        "top_video_titles": video_titles[:5],
    }


# ---------------------------------------------------------------------------
# Feature 4: Contrarian Score — Mathematical Jaccard Divergence
# ---------------------------------------------------------------------------

def compute_contrarian_score(competitor_title: str, contrarian_title: str) -> dict:
    """
    Measures how semantically different a contrarian title is from the
    competitor's title using Jaccard word divergence.

    Method:
      1. Tokenize both titles into word sets, stripping stop words and
         punctuation (these carry no meaning and inflate similarity).
      2. Jaccard Similarity = |intersection| / |union|
         (how many words they share vs how many total unique words)
      3. Divergence Score = (1 - Jaccard Similarity) × 10
         → Score 10: no shared words = maximally contrarian
         → Score 0:  identical word sets = same angle, not contrarian at all

    Returns:
        dict with divergence_score (int 1-10), shared_words (list),
        unique_to_contrarian (list), interpretation (str).
    """
    def _tokenize(text: str) -> set:
        # Lowercase, strip punctuation, remove stop words
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        return {w for w in words if w not in _STOP_WORDS and len(w) > 1}

    comp_words = _tokenize(competitor_title)
    cont_words = _tokenize(contrarian_title)

    if not comp_words or not cont_words:
        return {
            "divergence_score": 5,
            "shared_words": [],
            "unique_to_contrarian": list(cont_words),
            "interpretation": "Could not compute — one of the titles is empty.",
        }

    intersection = comp_words & cont_words
    union = comp_words | cont_words

    jaccard_similarity = len(intersection) / len(union)
    divergence_score = max(1, min(10, round((1 - jaccard_similarity) * 10)))

    # Human-readable interpretation
    if divergence_score >= 9:
        interp = "Excellent ✅ — Maximally contrarian. Zero conceptual overlap with competitor."
    elif divergence_score >= 7:
        interp = "Strong ✅ — Very different angle. Will stand out in search results."
    elif divergence_score >= 5:
        interp = "Moderate ⚠️ — Some overlap. Could be more distinct."
    elif divergence_score >= 3:
        interp = "Weak ⚠️ — Too similar to competitor. Consider a bolder pivot."
    else:
        interp = "Poor ❌ — Almost identical to competitor. Not truly contrarian."

    return {
        "divergence_score": divergence_score,
        "shared_words": sorted(intersection),
        "unique_to_contrarian": sorted(cont_words - comp_words),
        "interpretation": interp,
    }
