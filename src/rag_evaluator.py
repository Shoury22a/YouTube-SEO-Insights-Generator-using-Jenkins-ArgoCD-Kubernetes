"""
RAG Evaluator Module — RAGAS-based quality scoring for TubeRank AI.

Evaluates the quality of the RAG pipeline after each agent execution
using 4 industry-standard metrics from the RAGAS library:

  1. Faithfulness       — Is the output grounded in retrieved context?
  2. AnswerRelevancy    — Is the output relevant to the user's topic?
  3. ContextPrecision   — Were the retrieved docs actually useful?
  4. ContextRecall      — Did we retrieve enough relevant information?

Toggle: Set RAGAS_EVALUATION_ENABLED=true in .env to activate.
When disabled (default), all functions are zero-cost no-ops.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def is_eval_enabled() -> bool:
    """Check if RAGAS evaluation is enabled via environment variable."""
    return os.getenv("RAGAS_EVALUATION_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Result Data Class
# ---------------------------------------------------------------------------

@dataclass
class RAGEvalResult:
    """Container for RAG evaluation scores."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    verdict: str = "Not Evaluated"
    enabled: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "verdict": self.verdict,
            "enabled": self.enabled,
            "error": self.error,
        }

    @property
    def average_score(self) -> float:
        """Weighted average of all 4 metrics."""
        scores = [self.faithfulness, self.answer_relevancy,
                  self.context_precision, self.context_recall]
        valid = [s for s in scores if s > 0]
        return round(sum(valid) / len(valid), 3) if valid else 0.0


def _compute_verdict(avg: float) -> str:
    """Human-readable verdict from the average score."""
    if avg >= 0.8:
        return "Excellent"
    elif avg >= 0.6:
        return "Good"
    elif avg >= 0.4:
        return "Needs Improvement"
    else:
        return "Poor"


# ---------------------------------------------------------------------------
# RAGAS LLM Factory — Uses Gemini via google-genai SDK
# ---------------------------------------------------------------------------

_ragas_llm = None


def _get_ragas_llm():
    """
    Lazily create and cache the RAGAS-compatible LLM wrapper.
    Uses the google-genai SDK (recommended by RAGAS for Gemini).
    Falls back to LangchainLLMWrapper if the native client isn't available.
    """
    global _ragas_llm
    if _ragas_llm is not None:
        return _ragas_llm

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Cannot initialize RAGAS evaluator.")

    # Strategy 1: Use google-genai native client (recommended)
    try:
        from google import genai
        from ragas.llms import llm_factory

        client = genai.Client(api_key=api_key)
        _ragas_llm = llm_factory(
            model="gemini-1.5-flash-8b",
            provider="google",
            client=client,
        )
        logger.info("RAGAS evaluator initialized (google-genai native client).")
        return _ragas_llm
    except ImportError:
        logger.info("google-genai not available. Trying LangChain wrapper fallback.")
    except Exception as e:
        logger.warning(f"RAGAS native client failed: {e}. Trying LangChain fallback.")

    # Strategy 2: Fall back to LangchainLLMWrapper
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper

        langchain_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-8b",
            google_api_key=api_key,
            temperature=0,
        )
        _ragas_llm = LangchainLLMWrapper(langchain_llm)
        logger.info("RAGAS evaluator initialized (LangChain wrapper fallback).")
        return _ragas_llm
    except Exception as e:
        raise RuntimeError(f"Could not initialize RAGAS LLM: {e}") from e


# ---------------------------------------------------------------------------
# Core Evaluation Function
# ---------------------------------------------------------------------------

def evaluate_rag_quality(
    topic: str,
    audience: str,
    retrieved_contexts: list[str],
    final_output: dict,
    content_type: str = "Long-Form Video",
) -> RAGEvalResult:
    """
    Evaluate the RAG pipeline quality using RAGAS metrics.

    This function is safe to call regardless of the toggle state —
    it returns a no-op result when disabled and never crashes.

    Args:
        topic: The video topic (user's question).
        audience: The target audience.
        retrieved_contexts: List of documents retrieved from ChromaDB.
        final_output: The final SEO metadata dict from the agent.
        content_type: "Long-Form Video" or "YouTube Short".

    Returns:
        RAGEvalResult with all 4 metric scores.
    """
    # ── Gate: Skip if disabled ──────────────────────────────────────────
    if not is_eval_enabled():
        return RAGEvalResult(verdict="Disabled", enabled=False)

    # ── Gate: Skip if no context was retrieved (nothing to evaluate) ────
    if not retrieved_contexts:
        logger.info("RAGAS: No retrieved contexts — skipping evaluation.")
        return RAGEvalResult(
            verdict="Skipped (no retrieved context)",
            enabled=True,
        )

    logger.info(f"RAGAS: Evaluating RAG quality for topic='{topic[:50]}' "
                f"with {len(retrieved_contexts)} contexts...")

    try:
        from ragas import evaluate
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
            ContextRecall,
        )
        from ragas import SingleTurnSample, EvaluationDataset

        # ── Build the user input (the "question") ──────────────────────
        user_input = (
            f"Generate YouTube SEO metadata for a {content_type} about: {topic}. "
            f"Target audience: {audience}."
        )

        # ── Build the response (the LLM's answer) ─────────────────────
        titles = final_output.get("titles", [])
        description = final_output.get("description", "")
        tags = final_output.get("tags", [])
        response_text = (
            f"Titles: {' | '.join(titles)}\n"
            f"Description: {description[:500]}\n"
            f"Tags: {', '.join(tags[:15])}"
        )

        # ── Build the reference (ground truth intent) ──────────────────
        reference = (
            f"A comprehensive SEO metadata package for a {content_type} "
            f"about '{topic}' targeting '{audience}'. Should include "
            f"relevant titles with keywords, an optimized description, "
            f"and targeted tags."
        )

        # ── Create the RAGAS sample ────────────────────────────────────
        sample = SingleTurnSample(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
            response=response_text,
            reference=reference,
        )
        eval_dataset = EvaluationDataset(samples=[sample])

        # ── Get the evaluator LLM ─────────────────────────────────────
        evaluator_llm = _get_ragas_llm()

        # ── Run RAGAS evaluation ──────────────────────────────────────
        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecisionWithoutReference(),
            ContextRecall(),
        ]

        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
        )

        # ── Extract scores ────────────────────────────────────────────
        scores_df = results.to_pandas()
        row = scores_df.iloc[0] if len(scores_df) > 0 else {}

        faithfulness = float(row.get("faithfulness", 0.0))
        answer_rel = float(row.get("answer_relevancy", 0.0))
        ctx_precision = float(row.get("context_precision", 0.0))
        ctx_recall = float(row.get("context_recall", 0.0))

        avg = (faithfulness + answer_rel + ctx_precision + ctx_recall) / 4
        verdict = _compute_verdict(avg)

        result = RAGEvalResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_rel,
            context_precision=ctx_precision,
            context_recall=ctx_recall,
            verdict=verdict,
            enabled=True,
        )

        logger.info(
            f"RAGAS evaluation complete | "
            f"faithfulness={faithfulness:.2f} | "
            f"answer_relevancy={answer_rel:.2f} | "
            f"context_precision={ctx_precision:.2f} | "
            f"context_recall={ctx_recall:.2f} | "
            f"verdict={verdict}"
        )

        return result

    except Exception as e:
        # RAGAS evaluation should NEVER crash the main pipeline
        logger.warning(f"RAGAS evaluation failed: {e}. Returning empty scores.")
        return RAGEvalResult(
            verdict="Error",
            enabled=True,
            error=str(e),
        )
