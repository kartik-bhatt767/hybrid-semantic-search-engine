import os
import sys
import json
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine import HybridSearchEngine

st.set_page_config(
    page_title="Hybrid Semantic Search Workbench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern FAANG dark-mode styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .result-card {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 14px;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
    }
    .badge {
        background-color: #0284c7;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_search_engine():
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    engine = HybridSearchEngine(embedding_dim=32)
    engine.build_index(corpus)
    return engine

engine = load_search_engine()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("⚡ Engine Controls")
st.sidebar.markdown("**Two-Stage Hybrid Search & RRF**")

st.sidebar.markdown("---")
sample_queries = [
    "Select a pre-filled query...",
    "Mitigating hallucinations in LLMs via Hybrid RAG",
    "Approximate Nearest Neighbor Search with HNSW",
    "Scaling Distributed Key-Value Stores with Raft Consensus",
    "LSM Trees vs B-Trees in Modern Storage Engines",
    "Quantization and Pruning for Edge ML Inference"
]
selected_sample = st.sidebar.selectbox("💡 Sample Evaluation Queries:", sample_queries)

top_k = st.sidebar.slider("Top-K Final Results", min_value=1, max_value=10, value=5)
candidate_pool = st.sidebar.slider("Candidate Retrieval Pool (Stage 1)", min_value=5, max_value=25, value=15)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🏗️ Architecture
- **Stage 1A**: Dense Vector (Cosine Inner Product)
- **Stage 1B**: Sparse Okapi BM25
- **Stage 2A**: Reciprocal Rank Fusion ($k=60$)
- **Stage 2B**: Cross-Encoder Semantic Re-Ranker
""")
st.sidebar.info("Built by **@kartik-bhatt767** | FAANG Placement Portfolio")

# --- MAIN CONTENT ---
st.title("⚡ Hybrid Semantic Search Engine")
st.markdown("Combines **Dense Semantic Embeddings** and **Sparse Okapi BM25** via **Reciprocal Rank Fusion (RRF)** to eliminate vocabulary mismatch and semantic blindness.")

# Query input
default_text = selected_sample if selected_sample != "Select a pre-filled query..." else ""
query = st.text_input("🔍 Enter your search query:", value=default_text, placeholder="e.g. How to prevent hallucination in LLMs using vector retrieval?")

col1, col2 = st.columns([1, 5])
with col1:
    search_clicked = st.button("🚀 Search", use_container_width=True, type="primary")

if query and (search_clicked or default_text):
    response = engine.search_hybrid(query, top_k=top_k, candidate_pool_size=candidate_pool)
    latencies = response["latency_breakdown_ms"]
    results = response["results"]

    st.markdown("### ⏱️ Latency & Execution Breakdown (P95 Sub-Millisecond)")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total E2E", f"{latencies['total_e2e']:.2f} ms")
    with m2:
        st.metric("Dense Vector", f"{latencies['dense_retrieval']:.2f} ms")
    with m3:
        st.metric("Sparse BM25", f"{latencies['sparse_bm25']:.2f} ms")
    with m4:
        st.metric("RRF Fusion", f"{latencies['rrf_fusion']:.2f} ms")
    with m5:
        st.metric("Re-Ranking", f"{latencies['cross_encoder_rerank']:.2f} ms")

    st.markdown("---")
    st.subheader(f"🎯 Top {len(results)} Ranked Results (Fused & Re-Ranked)")

    for rank, item in enumerate(results, start=1):
        score_pct = min(100, int(item["relevance_score"] * 30))
        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #f8fafc;">#{rank} {item['title']}</h4>
                <span class="badge">{item['category']}</span>
            </div>
            <p style="color: #cbd5e1; margin-top: 8px; font-size: 14px; line-height: 1.5;">{item['content']}</p>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">
                <b>Document ID:</b> <code>{item['id']}</code> | 
                <b>Fusion Score:</b> <code>{item['fusion_rank_score']:.5f}</code> | 
                <b>Relevance Score:</b> <code>{item['relevance_score']:.2f}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🛠️ View Raw JSON Payload"):
        st.json(response)
