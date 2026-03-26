"""
History Dashboard — My Past Generations page for TubeRank AI.

Reads all stored SEO generations from ChromaDB and displays:
  - Total generation count
  - Recent topics timeline
  - Best and most common hook types
  - Topic keyword frequency analysis

This page surfaces the RAG store data visually — proving to the user
that the "memory" system is working and growing over time.
"""

import streamlit as st
from src.rag_store import retrieve_similar, get_store_stats
from src.title_scorer import score_title
from src.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="My History — TubeRank AI",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: #0a0a12; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 My Generation History")
st.markdown("All past SEO packages generated and stored in your ChromaDB memory.")
st.markdown("---")

# ── Stats banner ──────────────────────────────────────────────────────────
stats = get_store_stats()
total = stats.get("total_documents", 0)
status = stats.get("status", "unavailable")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📚 Total Generations", total, help="Total SEO packages stored in memory")
with col2:
    st.metric("🧠 Memory Status", status.title())
with col3:
    st.metric("🎯 RAG Active", "✅ Yes" if total > 0 else "⏳ Building...")

st.markdown("---")

if total == 0:
    st.info(
        "🚀 **No generations yet!** Go back to the main page and generate your first SEO package. "
        "It will appear here automatically.",
        icon="ℹ️",
    )
    st.stop()

# ── Retrieve all stored topics ────────────────────────────────────────────
st.markdown("## 🕐 Recent Generations")
st.caption("Most semantically recent topics retrieved from ChromaDB.")

# Use a broad query to retrieve all stored docs
all_docs = retrieve_similar("youtube video seo metadata", k=50)

if not all_docs:
    st.warning("Could not retrieve history from ChromaDB.")
    st.stop()

# ── Recent topics timeline ────────────────────────────────────────────────
for i, doc in enumerate(all_docs[:20], 1):
    topic = doc.get("topic", "Unknown Topic")
    content_type = doc.get("content_type", "Long-Form Video")
    language = doc.get("language", "English")
    titles_preview = doc.get("titles", "")
    score_val = doc.get("score", 0)

    # Parse first title for scoring
    first_title = titles_preview.split(" | ")[0] if titles_preview else ""
    score_data = score_title(first_title, topic) if first_title else {}
    ctr = score_data.get("ctr_score", 0) if score_data else 0
    hook = score_data.get("hook_type", "—") if score_data else "—"
    hook_emoji = score_data.get("hook_emoji", "") if score_data else ""

    type_badge = "📱 Short" if content_type == "YouTube Short" else "🎬 Long-Form"
    ctr_color = "#00e676" if ctr >= 7 else "#ffaa00" if ctr >= 4 else "#ff4444"

    st.markdown(
        f"""
        <div style='background:rgba(255,255,255,0.03);border-radius:10px;
                    padding:12px 16px;margin-bottom:8px;
                    border-left:3px solid {ctr_color}'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start'>
            <div>
              <div style='font-size:0.7rem;opacity:0.4;margin-bottom:3px'>#{i} · {type_badge} · {language}</div>
              <div style='font-weight:600;font-size:0.95rem;margin-bottom:6px'>{topic}</div>
              <div style='font-size:0.78rem;opacity:0.55'>{first_title}</div>
            </div>
            <div style='text-align:right;min-width:80px'>
              <div style='color:{ctr_color};font-weight:700;font-size:0.9rem'>CTR {ctr}/10</div>
              <div style='font-size:0.72rem;opacity:0.5'>{hook_emoji} {hook}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Hook Type Distribution ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🎯 Hook Type Distribution")
st.caption("Which psychological hooks appear most in your content?")

hook_counts: dict[str, int] = {}
for doc in all_docs:
    t = doc.get("titles", "").split(" | ")[0]
    tp = doc.get("topic", "")
    if t:
        s = score_title(t, tp)
        ht = s.get("hook_type", "Informational")
        hook_counts[ht] = hook_counts.get(ht, 0) + 1

if hook_counts:
    sorted_hooks = sorted(hook_counts.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_hooks[0][1]

    for hook_name, count in sorted_hooks:
        pct = count / max_count
        bar_w = int(pct * 200)
        st.markdown(
            f"<div style='margin-bottom:8px;font-size:0.85rem'>"
            f"<span style='display:inline-block;width:140px'>{hook_name}</span>"
            f"<span style='background:#7c3aed;display:inline-block;"
            f"height:12px;width:{bar_w}px;border-radius:3px;vertical-align:middle'></span>"
            f"<span style='margin-left:8px;opacity:0.6'>{count}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Keyword Frequency ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔑 Your Most Common Topics")

word_freq: dict[str, int] = {}
stop = {"the","a","an","in","on","to","for","of","and","or","how","what","is","are","with","my","your"}
for doc in all_docs:
    topic = doc.get("topic", "")
    for w in topic.lower().split():
        w = w.strip(".,!?")
        if w not in stop and len(w) > 3:
            word_freq[w] = word_freq.get(w, 0) + 1

top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
if top_words:
    pills = "".join(
        f"<span style='background:rgba(124,58,237,0.2);border:1px solid rgba(124,58,237,0.4);"
        f"border-radius:20px;padding:4px 12px;margin:3px;display:inline-block;"
        f"font-size:0.8rem'>{w} ({c})</span>"
        for w, c in top_words
    )
    st.markdown(f"<div style='line-height:2.5'>{pills}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("💡 This data is stored locally in `chroma_db/` — it grows smarter with every generation.")
