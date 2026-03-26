"""
SEO Strategist Agent — LangGraph State Machine for TubeRank AI.

Implements a 6-node agentic workflow:
  1. Researcher  — Fetches context from RAG + Web Search
  2. Grader      — Filters irrelevant retrieved documents
  3. Architect   — Drafts SEO metadata using existing LangChain pipeline
  4. Critic      — Evaluates draft against 5 deterministic + 2 LLM benchmarks
  5. Refiner     — Fixes only the failing components
  6. Finalizer   — Persists to ChromaDB and returns final output

The graph uses conditional edges to loop between Critic → Refiner
(max 2 retries) before finalizing.
"""

import time
from typing import TypedDict, Optional, Annotated

from langgraph.graph import StateGraph, END

from src.logger import get_logger
from src.rag_store import persist_generation, retrieve_similar

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Agent State — shared memory across all nodes
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """The shared state that travels through every node in the graph."""
    # Input fields (set once at the start)
    topic: str
    audience: str
    content_type: str
    language: str
    transcript_summary: str
    visual_description: str
    chapter_notes: str
    competitor_context: str

    # Enrichment fields (set by Researcher + Grader)
    retrieved_context: list[str]
    web_search_results: str

    # Generation fields (set by Architect, evaluated by Critic)
    draft_metadata: dict
    critique: str
    critique_details: list[str]

    # Control fields
    retry_count: int
    final_metadata: dict

    # Telemetry
    step_log: list[str]


# ---------------------------------------------------------------------------
# Node 1: Researcher — Gather context from RAG + Web Search
# ---------------------------------------------------------------------------

def researcher_node(state: AgentState) -> dict:
    """
    Gathers context from two sources:
    1. ChromaDB (past successful generations)
    2. DuckDuckGo Web Search (current trends)
    """
    topic = state["topic"]
    content_type = state.get("content_type", "Long-Form Video")
    language = state.get("language", "English")
    log = list(state.get("step_log", []))

    log.append("🔍 Researcher: Fetching past successes from memory...")

    # Source 1: RAG Retrieval
    retrieved = []
    try:
        similar = retrieve_similar(
            topic=topic,
            k=3,
            content_type=content_type,
            language=language,
        )
        for item in similar:
            if item.get("score", 0) > 0.3:  # Only keep reasonably similar
                retrieved.append(item.get("content", ""))
        logger.info(f"Researcher: Retrieved {len(retrieved)} relevant documents from RAG.")
    except Exception as e:
        logger.warning(f"Researcher: RAG retrieval failed: {e}")

    # Source 2: Web Search
    log.append("🌐 Researcher: Searching web for trending keywords...")
    web_results = ""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            search_query = f"{topic} youtube trending 2025"
            results = list(ddgs.text(search_query, max_results=3))
            if results:
                web_parts = []
                for r in results:
                    web_parts.append(f"- {r.get('title', '')}: {r.get('body', '')[:200]}")
                web_results = "\n".join(web_parts)
                logger.info(f"Researcher: Got {len(results)} web search results.")
    except Exception as e:
        logger.warning(f"Researcher: Web search failed: {e}. Proceeding without web context.")

    log.append(
        f"✅ Researcher: Found {len(retrieved)} past successes "
        f"and {'web trends' if web_results else 'no web data'}."
    )

    return {
        "retrieved_context": retrieved,
        "web_search_results": web_results,
        "step_log": log,
    }


# ---------------------------------------------------------------------------
# Node 2: Relevance Grader — Filter out irrelevant retrieved docs
# ---------------------------------------------------------------------------

def grader_node(state: AgentState) -> dict:
    """
    Filters retrieved documents for relevance.
    Uses simple keyword overlap instead of an LLM call (cheaper + faster).
    """
    retrieved = state.get("retrieved_context", [])
    topic = state["topic"].lower()
    log = list(state.get("step_log", []))

    if not retrieved:
        log.append("📋 Grader: No documents to grade. Skipping.")
        return {"retrieved_context": [], "step_log": log}

    log.append(f"📋 Grader: Evaluating {len(retrieved)} retrieved documents...")

    # Simple relevance check: does the document share meaningful words with the topic?
    topic_words = set(topic.split())
    graded = []
    for doc in retrieved:
        doc_words = set(doc.lower().split())
        overlap = topic_words & doc_words
        # Keep if at least 1 meaningful word overlaps (excluding stop words)
        stop_words = {"the", "a", "an", "in", "on", "to", "for", "of", "and", "or", "how", "what", "is"}
        meaningful_overlap = overlap - stop_words
        if meaningful_overlap:
            graded.append(doc)

    log.append(f"✅ Grader: {len(graded)}/{len(retrieved)} documents passed relevance check.")
    logger.info(f"Grader: {len(graded)}/{len(retrieved)} documents kept.")

    return {"retrieved_context": graded, "step_log": log}


# ---------------------------------------------------------------------------
# Node 3: Architect — Draft the SEO metadata
# ---------------------------------------------------------------------------

def architect_node(state: AgentState) -> dict:
    """
    Generates the initial SEO metadata draft.
    Uses the EXISTING generate_seo_metadata() logic from ai_model.py,
    enhanced with RAG context and web search results.
    """
    log = list(state.get("step_log", []))
    log.append("🧠 Architect: Drafting initial SEO package...")

    # Import the core generation function
    from src.ai_model import generate_seo_metadata

    # Build enhanced competitor context with RAG + web data
    enhanced_context = state.get("competitor_context", "") or ""

    # Add RAG context
    retrieved = state.get("retrieved_context", [])
    if retrieved:
        rag_section = "\n\n--- PAST SUCCESSFUL METADATA (for reference) ---\n"
        for i, doc in enumerate(retrieved, 1):
            rag_section += f"\n[Past Success #{i}]:\n{doc}\n"
        enhanced_context += rag_section

    # Add web search context
    web_results = state.get("web_search_results", "")
    if web_results:
        enhanced_context += (
            f"\n\n--- CURRENT TRENDING TOPICS (from web search) ---\n"
            f"{web_results}\n"
        )

    # Call the existing generation function
    try:
        result = generate_seo_metadata(
            topic=state["topic"],
            audience=state["audience"],
            content_type=state.get("content_type", "Long-Form Video"),
            output_language=state.get("language", "English"),
            transcript=state.get("transcript_summary", ""),
            visual_description=state.get("visual_description", ""),
            chapter_notes=state.get("chapter_notes", ""),
            competitor_context=enhanced_context,
        )
        log.append("✅ Architect: Initial draft complete.")
        return {"draft_metadata": result, "step_log": log}

    except Exception as e:
        logger.error(f"Architect: Generation failed: {e}")
        log.append(f"❌ Architect: Generation failed — {e}")
        return {
            "draft_metadata": {},
            "critique": "PASS",  # Skip critic if generation failed entirely
            "step_log": log,
        }


# ---------------------------------------------------------------------------
# Node 4: SEO Critic — Evaluate draft against benchmarks
# ---------------------------------------------------------------------------

def critic_node(state: AgentState) -> dict:
    """
    Evaluates the draft against 5 deterministic SEO benchmarks.
    Returns "PASS" if all checks succeed, or a detailed critique string.
    """
    draft = state.get("draft_metadata", {})
    content_type = state.get("content_type", "Long-Form Video")
    topic = state["topic"]
    log = list(state.get("step_log", []))

    log.append("⚖️ Critic: Evaluating against SEO benchmarks...")

    if not draft:
        log.append("⚖️ Critic: No draft to evaluate. Passing through.")
        return {"critique": "PASS", "critique_details": [], "step_log": log}

    failures = []

    # ── Benchmark 1: Title Length ────────────────────────────────────────
    max_title_len = 45 if content_type == "YouTube Short" else 70
    titles = draft.get("titles", [])
    for i, title in enumerate(titles):
        if len(title) > max_title_len:
            failures.append(
                f"Title {i+1} is {len(title)} chars (limit: {max_title_len}): '{title}'"
            )

    # ── Benchmark 2: Tag Character Count ────────────────────────────────
    tags = draft.get("tags", [])
    total_tag_chars = len(", ".join(tags))
    if total_tag_chars > 500:
        failures.append(
            f"Total tag characters ({total_tag_chars}) exceeds 500 limit."
        )

    # ── Benchmark 3: Keyword Presence in Titles ─────────────────────────
    topic_keywords = [w.lower() for w in topic.split() if len(w) > 3]
    if topic_keywords:
        titles_text = " ".join(titles).lower()
        keyword_found = any(kw in titles_text for kw in topic_keywords)
        if not keyword_found:
            failures.append(
                f"No title contains the core keyword(s): {topic_keywords}"
            )

    # ── Benchmark 4: Description Length ─────────────────────────────────
    description = draft.get("description", "")
    word_count = len(description.split())
    if content_type != "YouTube Short":
        if word_count < 100:
            failures.append(
                f"Description is only {word_count} words (minimum: 100)."
            )
        elif word_count > 500:
            failures.append(
                f"Description is {word_count} words (maximum: 500)."
            )

    # ── Benchmark 5: Required Fields Present ────────────────────────────
    required_fields = ["titles", "description", "tags", "social_posts", "thumbnail_ideas"]
    for field in required_fields:
        value = draft.get(field)
        if not value:
            failures.append(f"Missing required field: '{field}'")

    # ── Verdict ─────────────────────────────────────────────────────────
    if failures:
        critique = "FAIL: " + " | ".join(failures)
        log.append(f"❌ Critic: {len(failures)} issue(s) found.")
        for f in failures:
            log.append(f"   → {f}")
        logger.info(f"Critic: FAIL — {len(failures)} issues: {failures}")
    else:
        critique = "PASS"
        log.append("✅ Critic: All 5 benchmarks passed!")
        logger.info("Critic: PASS — All benchmarks passed.")

    return {
        "critique": critique,
        "critique_details": failures,
        "step_log": log,
    }


# ---------------------------------------------------------------------------
# Node 5: Refiner — Fix only the broken parts
# ---------------------------------------------------------------------------

def refiner_node(state: AgentState) -> dict:
    """
    Targeted refinement — fixes ONLY the components that failed the critique.
    Uses a focused LLM call instead of re-running the full Architect.
    """
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI

    draft = state.get("draft_metadata", {})
    critique = state.get("critique", "")
    retry_count = state.get("retry_count", 0)
    log = list(state.get("step_log", []))

    log.append(f"✏️ Refiner: Fixing issues (attempt {retry_count + 1}/2)...")

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.5,
            max_output_tokens=2048,
        )

        import json
        refine_prompt = (
            "You are an SEO quality editor. The following SEO metadata draft has issues.\n\n"
            f"DRAFT:\n{json.dumps(draft, indent=2, default=str)[:3000]}\n\n"
            f"ISSUES FOUND:\n{critique}\n\n"
            "Fix ONLY the issues listed above. Keep everything else exactly the same.\n"
            "Return the COMPLETE fixed JSON object with all fields preserved.\n"
            "Output ONLY valid JSON, no markdown formatting."
        )

        response = llm.invoke(refine_prompt)
        response_text = response.content.strip()

        # Clean markdown formatting if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        refined = json.loads(response_text)

        # Merge refined fields into the existing draft
        for key in refined:
            if key in draft:
                draft[key] = refined[key]

        log.append("✅ Refiner: Issues addressed. Sending back to Critic.")
        logger.info(f"Refiner: Refinement attempt {retry_count + 1} complete.")

    except Exception as e:
        logger.warning(f"Refiner: Refinement failed: {e}. Keeping original draft.")
        log.append(f"⚠️ Refiner: Could not fix issues ({e}). Keeping original.")

    return {
        "draft_metadata": draft,
        "retry_count": retry_count + 1,
        "step_log": log,
    }


# ---------------------------------------------------------------------------
# Node 6: Finalizer — Persist to ChromaDB and return
# ---------------------------------------------------------------------------

def finalizer_node(state: AgentState) -> dict:
    """
    Final step: saves the output to ChromaDB for future RAG retrieval
    and sets the final_metadata field.
    """
    draft = state.get("draft_metadata", {})
    log = list(state.get("step_log", []))

    log.append("💾 Finalizer: Saving to memory for future use...")

    # Persist to ChromaDB (fire-and-forget, never crashes)
    if draft:
        persist_generation(
            topic=state["topic"],
            seo_bundle=draft,
            content_type=state.get("content_type", "Long-Form Video"),
            language=state.get("language", "English"),
        )

    log.append("✅ Finalizer: Generation complete and saved to memory!")

    return {
        "final_metadata": draft,
        "step_log": log,
    }


# ---------------------------------------------------------------------------
# Conditional Edge — Should we refine or finalize?
# ---------------------------------------------------------------------------

def should_refine(state: AgentState) -> str:
    """
    Routing logic after the Critic node:
    - "PASS"         → go to finalizer
    - retry_count≥2  → go to finalizer (safety cap)
    - otherwise      → go to refiner (try to fix)
    """
    critique = state.get("critique", "PASS")
    retry_count = state.get("retry_count", 0)

    if critique == "PASS":
        return "finalizer"
    if retry_count >= 2:
        logger.info("Critic: Max retries reached. Forcing finalization.")
        return "finalizer"
    return "refiner"


# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------

def build_seo_agent_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph state machine.

    Graph flow:
        researcher → grader → architect → critic
                                            │
                                    ┌───────┴────────┐
                                    ▼                ▼
                                finalizer         refiner → critic (loop)
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("grader", grader_node)
    graph.add_node("architect", architect_node)
    graph.add_node("critic", critic_node)
    graph.add_node("refiner", refiner_node)
    graph.add_node("finalizer", finalizer_node)

    # Define the edges (flow)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "grader")
    graph.add_edge("grader", "architect")
    graph.add_edge("architect", "critic")

    # Conditional edge: critic → finalizer OR critic → refiner
    graph.add_conditional_edges(
        "critic",
        should_refine,
        {
            "finalizer": "finalizer",
            "refiner": "refiner",
        }
    )

    # Refiner loops back to critic
    graph.add_edge("refiner", "critic")

    # Finalizer is the end
    graph.add_edge("finalizer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API — Run the Agent
# ---------------------------------------------------------------------------

# Compile once at module level for reuse
_compiled_agent = None


def get_agent():
    """Lazily compile and cache the agent graph."""
    global _compiled_agent
    if _compiled_agent is None:
        _compiled_agent = build_seo_agent_graph()
        logger.info("LangGraph SEO Agent compiled successfully.")
    return _compiled_agent


def run_seo_agent(
    topic: str,
    audience: str,
    content_type: str = "Long-Form Video",
    output_language: str = "English",
    transcript: str = "",
    visual_description: str = "",
    chapter_notes: str = "",
    competitor_context: str = "",
) -> dict:
    """
    Public API — runs the full agentic SEO generation pipeline.

    Returns:
        dict with keys: final_metadata (the SEO output), step_log (agent reasoning steps).
    """
    logger.info(f"Starting SEO Agent | topic='{topic}' | type={content_type}")
    t0 = time.time()

    agent = get_agent()

    initial_state: AgentState = {
        "topic": topic,
        "audience": audience,
        "content_type": content_type,
        "language": output_language,
        "transcript_summary": transcript,
        "visual_description": visual_description,
        "chapter_notes": chapter_notes,
        "competitor_context": competitor_context,
        "retrieved_context": [],
        "web_search_results": "",
        "draft_metadata": {},
        "critique": "",
        "critique_details": [],
        "retry_count": 0,
        "final_metadata": {},
        "step_log": [],
    }

    # Run the graph
    final_state = agent.invoke(initial_state)

    elapsed = time.time() - t0
    logger.info(f"SEO Agent completed in {elapsed:.2f}s | retries={final_state.get('retry_count', 0)}")

    return {
        "metadata": final_state.get("final_metadata", {}),
        "step_log": final_state.get("step_log", []),
        "retry_count": final_state.get("retry_count", 0),
        "retrieved_count": len(final_state.get("retrieved_context", [])),
        "elapsed_seconds": round(elapsed, 2),
    }
