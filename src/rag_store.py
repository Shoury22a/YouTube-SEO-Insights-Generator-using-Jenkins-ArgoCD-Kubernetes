"""
RAG Store Module — Semantic Memory for TubeRank AI.

Uses ChromaDB as a local persistent vector database and Google's
text-embedding-004 model to store and retrieve past SEO generations.

How it works:
  1. After every successful generation, the SEO metadata bundle (topic,
     titles, tags, description) is embedded as a vector and stored.
  2. On the next request, the system retrieves the top-k most
     semantically similar past generations to use as context.

ChromaDB runs locally (no K8s needed for development).
"""

import os
import json
from typing import Optional

from dotenv import load_dotenv

from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — these are heavy, only load when needed
# ---------------------------------------------------------------------------

_vectorstore = None
_embeddings = None


def _get_embeddings():
    """Lazily initialize Google Generative AI Embeddings."""
    global _embeddings
    if _embeddings is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set. Cannot initialize embeddings.")

        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        logger.info("Initialized GoogleGenerativeAIEmbeddings (text-embedding-004).")
    return _embeddings


def _get_vectorstore():
    """
    Lazily initialize the ChromaDB vector store.

    Uses a local persistent directory so data survives app restarts.
    In production (K8s), this would connect to a ChromaDB StatefulSet instead.
    """
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma

        persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        persist_dir = os.path.abspath(persist_dir)

        _vectorstore = Chroma(
            collection_name="seo_generations",
            embedding_function=_get_embeddings(),
            persist_directory=persist_dir,
        )
        logger.info(f"ChromaDB initialized at: {persist_dir}")
    return _vectorstore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def persist_generation(
    topic: str,
    seo_bundle: dict,
    content_type: str = "Long-Form Video",
    language: str = "English",
) -> None:
    """
    Store a successful SEO generation in the vector database.

    Each generation is stored as a single atomic document (NOT chunked)
    because SEO bundles are small and splitting would break the
    relationship between titles, tags, and descriptions.

    Args:
        topic: The video topic used for generation.
        seo_bundle: The full SEO output dict (titles, description, tags, etc).
        content_type: "Long-Form Video" or "YouTube Short".
        language: Output language used.
    """
    try:
        store = _get_vectorstore()

        # Build a rich text document from the SEO bundle
        titles_text = " | ".join(seo_bundle.get("titles", []))
        tags_text = ", ".join(seo_bundle.get("tags", []))
        description = seo_bundle.get("description", "")

        document_text = (
            f"Topic: {topic}\n"
            f"Titles: {titles_text}\n"
            f"Tags: {tags_text}\n"
            f"Description: {description[:500]}"
        )

        # Metadata for filtered retrieval
        metadata = {
            "topic": topic,
            "content_type": content_type,
            "language": language,
            "titles": titles_text[:500],
            "bundle_json": json.dumps(seo_bundle, default=str)[:2000],
        }

        store.add_texts(
            texts=[document_text],
            metadatas=[metadata],
        )

        logger.info(
            f"Persisted SEO generation to ChromaDB | topic='{topic}' | "
            f"type={content_type} | lang={language}"
        )

    except Exception as e:
        # RAG persistence should NEVER crash the main generation flow
        logger.warning(f"Failed to persist generation to ChromaDB: {e}")


def retrieve_similar(
    topic: str,
    k: int = 5,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve the top-k most semantically similar past SEO generations.

    Uses cosine similarity on the embedding vectors to find related content.
    Supports optional metadata filtering (e.g., only Shorts, only Hinglish).

    Args:
        topic: The current video topic to search for.
        k: Number of similar results to retrieve (default: 5).
        content_type: Optional filter ("Long-Form Video" or "YouTube Short").
        language: Optional filter (e.g., "English", "Hinglish").

    Returns:
        List of dicts with keys: 'topic', 'titles', 'content_type', 'score'.
        Empty list if no matches or ChromaDB is not available.
    """
    try:
        store = _get_vectorstore()

        # Build metadata filter
        where_filter = {}
        if content_type:
            where_filter["content_type"] = content_type
        if language:
            where_filter["language"] = language

        # Similarity search with scores
        if where_filter:
            results = store.similarity_search_with_relevance_scores(
                query=topic,
                k=k,
                filter=where_filter,
            )
        else:
            results = store.similarity_search_with_relevance_scores(
                query=topic,
                k=k,
            )

        similar = []
        for doc, score in results:
            similar.append({
                "content": doc.page_content,
                "topic": doc.metadata.get("topic", "Unknown"),
                "titles": doc.metadata.get("titles", ""),
                "content_type": doc.metadata.get("content_type", "Unknown"),
                "language": doc.metadata.get("language", "Unknown"),
                "score": round(score, 3),
            })

        logger.info(
            f"RAG retrieval for '{topic}': {len(similar)} results "
            f"(filter: type={content_type}, lang={language})"
        )
        return similar

    except Exception as e:
        # RAG retrieval should NEVER crash the main generation flow
        logger.warning(f"RAG retrieval failed: {e}. Returning empty context.")
        return []


def get_store_stats() -> dict:
    """
    Returns basic stats about the vector store (for UI display).
    """
    try:
        store = _get_vectorstore()
        collection = store._collection
        count = collection.count()
        return {"total_documents": count, "status": "connected"}
    except Exception as e:
        logger.warning(f"Could not get store stats: {e}")
        return {"total_documents": 0, "status": "unavailable"}
