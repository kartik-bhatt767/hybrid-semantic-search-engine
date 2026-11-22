# tests/test_search.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.embeddings import BM25Encoder, DenseVectorEncoder
from src.reranker import reciprocal_rank_fusion, CrossEncoderReranker
from src.engine import HybridSearchEngine
from api.main import app

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
    # Doc 2 (index 1) should be top result for consensus
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

def test_search_modes():
    engine = HybridSearchEngine(embedding_dim=8)
    engine.build_index(SAMPLE_CORPUS)

    query = "transformer attention"
    # Test Dense Only
    res_dense = engine.search_hybrid(query, top_k=2, mode="dense")
    assert res_dense["mode"] == "dense"
    assert len(res_dense["results"]) > 0

    # Test Sparse Only
    res_sparse = engine.search_hybrid(query, top_k=2, mode="sparse")
    assert res_sparse["mode"] == "sparse"
    assert len(res_sparse["results"]) > 0

    # Test Hybrid
    res_hybrid = engine.search_hybrid(query, top_k=2, mode="hybrid")
    assert res_hybrid["mode"] == "hybrid"
    assert len(res_hybrid["results"]) > 0

def test_2d_projection_engine():
    engine = HybridSearchEngine(embedding_dim=8)
    engine.build_index(SAMPLE_CORPUS)

    proj = engine.get_2d_projection(query="raft consensus")
    assert "documents" in proj
    assert len(proj["documents"]) == 3
    for doc in proj["documents"]:
        assert "x" in doc and "y" in doc
        assert -1.0 <= doc["x"] <= 1.0
        assert -1.0 <= doc["y"] <= 1.0

    assert proj["query"] is not None
    assert "x" in proj["query"] and "y" in proj["query"]

def test_dynamic_document_ingestion():
    engine = HybridSearchEngine(embedding_dim=8)
    engine.build_index(SAMPLE_CORPUS)
    assert len(engine.corpus) == 3

    new_doc = {
        "id": "4",
        "title": "Quantum Computing",
        "content": "Qubits and superposition revolutionize cryptography.",
        "category": "Quantum"
    }
    engine.build_index(SAMPLE_CORPUS + [new_doc])
    assert len(engine.corpus) == 4

    res = engine.search_hybrid("quantum superposition qubits", top_k=1)
    assert res["results"][0]["id"] == "4"

def test_fastapi_endpoints_e2e():
    with TestClient(app) as client:
        # 1. Health endpoint
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "HEALTHY"

        # 2. Search endpoint
        search_resp = client.post("/api/v1/search", json={
            "query": "Raft consensus protocol",
            "top_k": 3,
            "mode": "hybrid"
        })
        assert search_resp.status_code == 200
        data = search_resp.json()
        assert "results" in data
        assert "latency_breakdown_ms" in data

        # 3. Vector map endpoint
        vmap_resp = client.get("/api/v1/vector-map?query=transformer")
        assert vmap_resp.status_code == 200
        assert "documents" in vmap_resp.json()

        # 4. Metrics endpoint
        metrics_resp = client.get("/api/v1/metrics")
        assert metrics_resp.status_code == 200
        assert "NDCG@10" in metrics_resp.json()["metrics"]
