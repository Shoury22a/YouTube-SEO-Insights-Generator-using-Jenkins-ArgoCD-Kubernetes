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

def record_generation(elapsed_seconds: float, retry_count: int, retrieved_count: int) -> None:
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
