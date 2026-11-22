import os
import sys
import json
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine import HybridSearchEngine

st.set_page_config(
    page_title="HYBRID // Semantic AI Search",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Silicon Valley Dark-Mode Design System (Perplexity / Linear style)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #06090e;
        color: #f1f5f9;
    }

    .main .block-container {
        max-width: 1080px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Glow Hero Header */
    .hero-container {
        text-align: center;
        padding: 30px 10px 20px 10px;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38bdf8;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -1.2px;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        font-size: 16px;
        color: #94a3b8;
        max-width: 650px;
        margin: 0 auto 25px auto;
        line-height: 1.5;
    }

    /* Stat Glass Cards */
    .stat-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }
    .stat-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        margin-bottom: 4px;
    }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Result Cards */
    .result-item {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 16px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .result-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, #38bdf8, #818cf8);
    }
    .result-item:hover {
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5);
        transform: translateY(-1px);
    }
    .result-title {
        font-size: 18px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .result-category {
        font-size: 11px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        background: rgba(129, 140, 248, 0.12);
        color: #a5b4fc;
        border: 1px solid rgba(129, 140, 248, 0.25);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-body {
        font-size: 14px;
        line-height: 1.6;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    .meta-footer {
        display: flex;
        gap: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #64748b;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 10px;
    }
    .meta-highlight {
        color: #38bdf8;
        font-weight: 600;
    }

    /* Streamlit input custom overrides */
    .stTextInput > div > div > input {
        background-color: #0d131f !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-size: 16px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
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

# --- HERO SECTION ---
st.markdown("""
<div class="hero-container">
    <div class="hero-pill">⚡ PRODUCTION RETRIEVAL ENGINE • v1.2</div>
    <div class="hero-title">Hybrid Semantic Search</div>
    <div class="hero-subtitle">
        Sub-millisecond multi-index information retrieval combining <b>Dense Neural Embeddings</b> + <b>Okapi BM25 Lexical Matching</b> fused via <b>Reciprocal Rank Fusion (RRF)</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# Pre-filled Chips for instant tryout
chip_queries = [
    "Mitigating hallucinations in LLMs via Hybrid RAG",
    "Approximate Nearest Neighbor Search with HNSW",
    "Scaling Distributed Key-Value Stores with Raft Consensus",
    "Quantization and Pruning for Edge ML Inference"
]

st.markdown("<p style='text-align: center; color: #64748b; font-size: 13px; margin-bottom: 8px;'>TRY SUGGESTED EVALUATION QUERIES:</p>", unsafe_allow_html=True)
chip_cols = st.columns(len(chip_queries))

selected_query = None
for i, cq in enumerate(chip_queries):
    with chip_cols[i]:
        # Truncate for button label
        short_label = cq.split(" with ")[0].split(" via ")[0]
        if st.button(f"🔍 {short_label}", key=f"chip_{i}", use_container_width=True):
            selected_query = cq

# Search Bar
search_query = st.text_input(
    label="Search query",
    value=selected_query or "",
    placeholder="Ask anything or search engineering docs (e.g., 'How does Raft elect leaders?')...",
    label_visibility="collapsed"
)

# Controls in subtle columns
c1, c2, c3 = st.columns([2, 2, 4])
with c1:
    top_k = st.slider("Top Results (k)", min_value=1, max_value=8, value=4)
with c2:
    pool_size = st.slider("Retrieval Pool", min_value=5, max_value=20, value=12)

# --- EXECUTE SEARCH ---
if search_query:
    response = engine.search_hybrid(search_query, top_k=top_k, candidate_pool_size=pool_size)
    latencies = response["latency_breakdown_ms"]
    results = response["results"]

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 4 Sleek Glassmorphism Stat Cards
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total E2E</div>
            <div class="stat-value">{latencies['total_e2e']:.2f} ms</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Dense Vector</div>
            <div class="stat-value">{latencies['dense_retrieval']:.2f} ms</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Sparse BM25</div>
            <div class="stat-value">{latencies['sparse_bm25']:.2f} ms</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">RRF Fusion</div>
            <div class="stat-value">{latencies['rrf_fusion']:.2f} ms</div>
        </div>
        """, unsafe_allow_html=True)
    with sc5:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Re-Ranker</div>
            <div class="stat-value">{latencies['cross_encoder_rerank']:.2f} ms</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #64748b; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Retrieved & Reranked ({len(results)} of {response['total_candidates_retrieved']} candidates)</p>", unsafe_allow_html=True)

    # Display Rich Result Cards
    for rank, item in enumerate(results, start=1):
        st.markdown(f"""
        <div class="result-item">
            <div class="result-title">
                <span><span style="color: #38bdf8; margin-right: 8px;">0{rank}.</span> {item['title']}</span>
                <span class="result-category">{item['category']}</span>
            </div>
            <div class="result-body">{item['content']}</div>
            <div class="meta-footer">
                <span>DOC_ID: <span class="meta-highlight">{item['id']}</span></span>
                <span>FUSION_RANK_SCORE: <span class="meta-highlight">{item['fusion_rank_score']:.5f}</span></span>
                <span>RELEVANCE: <span class="meta-highlight">{item['relevance_score']:.2f}</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 12px;'>Engineered with Python, FastAPI, NumPy, and PyTorch by <b>Kartik Bhatt</b> (@kartik-bhatt767)</p>", unsafe_allow_html=True)
