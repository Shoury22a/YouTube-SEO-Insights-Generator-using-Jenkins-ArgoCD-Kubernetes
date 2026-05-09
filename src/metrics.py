"""
Prometheus Metrics Module — Observability for TubeRank AI Agent.

Exposes custom metrics for tracking:
  - Number of refinement loops per request
  - End-to-end generation latency
  - RAG retrieval counts
  - Critic pass/fail rates

Metrics are served on a background HTTP server (port 8502)
and can be scraped by Prometheus via K8s annotations.
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

GENERATION_DURATION = Histogram(
    "tuberank_generation_duration_seconds",
    "End-to-end generation latency in seconds",
    buckets=[5, 10, 15, 20, 30, 45, 60, 90, 120],
)

REFINEMENT_LOOPS = Counter(
    "tuberank_agent_refinement_loops_total",
    "Total number of critique-refine cycles",
)

CRITIC_PASS = Counter(
    "tuberank_critic_first_pass_total",
    "Number of times the critic passed on the first attempt",
)

CRITIC_FAIL = Counter(
    "tuberank_critic_fail_total",
    "Number of times the critic found issues",
)

RAG_RETRIEVED_DOCS = Gauge(
    "tuberank_rag_retrieved_docs",
    "Number of documents retrieved from ChromaDB in the last request",
)

AGENT_REQUESTS = Counter(
    "tuberank_agent_requests_total",
    "Total number of agent invocations",
)

# ── RAGAS RAG Evaluation Metrics ──────────────────────────────────────────────

RAG_FAITHFULNESS = Gauge(
    "tuberank_rag_faithfulness",
    "RAGAS faithfulness score (0.0-1.0) — is output grounded in context?",
)

RAG_ANSWER_RELEVANCY = Gauge(
    "tuberank_rag_answer_relevancy",
    "RAGAS answer relevancy score (0.0-1.0) — is output relevant to topic?",
)

RAG_CONTEXT_PRECISION = Gauge(
    "tuberank_rag_context_precision",
    "RAGAS context precision score (0.0-1.0) — were retrieved docs useful?",
)

RAG_CONTEXT_RECALL = Gauge(
    "tuberank_rag_context_recall",
    "RAGAS context recall score (0.0-1.0) — did we retrieve enough?",
)


# ---------------------------------------------------------------------------
# Metrics Server
# ---------------------------------------------------------------------------

_metrics_started = False


def start_metrics_server(port: int = 8502) -> None:
    """
    Start the Prometheus metrics HTTP server on a background thread.
    Safe to call multiple times — only starts once.
    """
    global _metrics_started
    if not _metrics_started:
        try:
            start_http_server(port)
            _metrics_started = True
            logger.info(f"Prometheus metrics server started on port {port}.")
        except Exception as e:
            logger.warning(f"Could not start metrics server: {e}")


# ---------------------------------------------------------------------------
# Convenience functions for recording metrics
# ---------------------------------------------------------------------------

def record_generation(
    elapsed_seconds: float,
    retry_count: int,
    retrieved_count: int,
    rag_eval: dict | None = None,
) -> None:
    """Record metrics for a completed generation."""
    GENERATION_DURATION.observe(elapsed_seconds)
    AGENT_REQUESTS.inc()
    RAG_RETRIEVED_DOCS.set(retrieved_count)

    if retry_count == 0:
        CRITIC_PASS.inc()
    else:
        CRITIC_FAIL.inc()
        for _ in range(retry_count):
            REFINEMENT_LOOPS.inc()

    # Record RAGAS evaluation scores (if available)
    if rag_eval and rag_eval.get("enabled"):
        RAG_FAITHFULNESS.set(rag_eval.get("faithfulness", 0.0))
        RAG_ANSWER_RELEVANCY.set(rag_eval.get("answer_relevancy", 0.0))
        RAG_CONTEXT_PRECISION.set(rag_eval.get("context_precision", 0.0))
        RAG_CONTEXT_RECALL.set(rag_eval.get("context_recall", 0.0))
        logger.info(
            f"RAGAS metrics recorded: faith={rag_eval.get('faithfulness', 0):.2f} | "
            f"relevancy={rag_eval.get('answer_relevancy', 0):.2f} | "
            f"precision={rag_eval.get('context_precision', 0):.2f} | "
            f"recall={rag_eval.get('context_recall', 0):.2f}"
        )
