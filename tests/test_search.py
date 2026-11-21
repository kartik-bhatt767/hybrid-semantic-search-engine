# tests/test_search.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import numpy as np
from src.embeddings import BM25Encoder, DenseVectorEncoder
from src.reranker import reciprocal_rank_fusion, CrossEncoderReranker
from src.engine import HybridSearchEngine

SAMPLE_CORPUS = [
    {"id": "1", "title": "Transformer Attention", "content": "Self-attention enables massive context in deep learning.", "category": "AI"},
    {"id": "2", "title": "Raft Consensus", "content": "Raft is a leader-based distributed consensus protocol.", "category": "Systems"},
    {"id": "3", "title": "Vector Databases", "content": "Approximate nearest neighbor indexes like HNSW scale vector search.", "category": "Search"}
]

def test_bm25_scoring():
    bm25 = BM25Encoder()
    texts = [f"{d['title']} {d['content']}" for d in SAMPLE_CORPUS]
    bm25.fit(texts)

    results = bm25.search("consensus leader", top_k=2)
    assert len(results) > 0
    # Doc 1 (index 1) should be top result for consensus
    assert results[0][0] == 1

def test_dense_vector_normalization():
    encoder = DenseVectorEncoder(embedding_dim=8)
    texts = [f"{d['title']} {d['content']}" for d in SAMPLE_CORPUS]
    embeddings = encoder.fit_transform(texts)

    # Unit norm check
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert np.isclose(norm, 1.0, atol=1e-4)

def test_reciprocal_rank_fusion():
    # Dense ranks: doc 2 (first), doc 0 (second)
    list_a = [(2, 0.95), (0, 0.80)]
    # Sparse ranks: doc 2 (first), doc 1 (second)
    list_b = [(2, 4.5), (1, 3.2)]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    # Doc 2 is #1 in both lists, must be top
    assert fused[0][0] == 2
    assert fused[0][1] > fused[1][1]

def test_hybrid_search_e2e():
    engine = HybridSearchEngine(embedding_dim=8)
    engine.build_index(SAMPLE_CORPUS)

    query = "nearest neighbor vector search"
    res = engine.search_hybrid(query, top_k=2)

    assert "results" in res
    assert len(res["results"]) > 0
    assert "latency_breakdown_ms" in res
    assert res["results"][0]["id"] == "3"
