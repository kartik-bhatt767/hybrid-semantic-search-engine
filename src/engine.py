# src/engine.py
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from src.embeddings import BM25Encoder, DenseVectorEncoder
from src.reranker import reciprocal_rank_fusion, CrossEncoderReranker

class HybridSearchEngine:
    """
    Production-grade Two-Stage Hybrid Information Retrieval & Semantic Search Engine.
    Stage 1: Multi-index retrieval (Dense Semantic + Sparse Okapi BM25).
    Stage 2: Reciprocal Rank Fusion (RRF) & Cross-Encoder re-ranking.
    """
    def __init__(self, embedding_dim: int = 32):
        self.bm25 = BM25Encoder()
        self.dense = DenseVectorEncoder(embedding_dim=embedding_dim)
        self.reranker = CrossEncoderReranker()
        self.corpus = []
        self.dense_embeddings = None

    def build_index(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus
        texts = [f"{d['title']} {d['content']}" for d in corpus]

        # 1. Fit Sparse Lexical BM25
        self.bm25.fit(texts)

        # 2. Fit Dense Semantic Embeddings
        self.dense_embeddings = self.dense.fit_transform(texts)
        print(f"Indexed {len(corpus)} documents with dimension {self.dense_embeddings.shape[1]}.")

    def search_dense(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q_emb = self.dense.transform([query])[0]
        # Cosine similarity via inner product on normalized embeddings
        scores = np.dot(self.dense_embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def search_sparse(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        return self.bm25.search(query, top_k=top_k)

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int = 15,
        mode: str = "hybrid",
        rrf_k: int = 60
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # Step 1: Parallel retrieval based on mode
        t_dense = 0.0
        t_sparse = 0.0
        dense_results = []
        sparse_results = []

        if mode in ["hybrid", "dense"]:
            t_dense_start = time.perf_counter()
            dense_results = self.search_dense(query, top_k=candidate_pool_size)
            t_dense = (time.perf_counter() - t_dense_start) * 1000.0

        if mode in ["hybrid", "sparse"]:
            t_sparse_start = time.perf_counter()
            sparse_results = self.search_sparse(query, top_k=candidate_pool_size)
            t_sparse = (time.perf_counter() - t_sparse_start) * 1000.0

        # Step 2: Ranking / Fusion
        t_fusion_start = time.perf_counter()
        if mode == "dense":
            fused_candidates = [(idx, score) for idx, score in dense_results]
        elif mode == "sparse":
            fused_candidates = [(idx, score) for idx, score in sparse_results]
        else:
            fused_candidates = reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k)
        t_fusion = (time.perf_counter() - t_fusion_start) * 1000.0

        # Step 3: Cross-Encoder Re-Ranking
        t_rerank_start = time.perf_counter()
        final_results = self.reranker.rerank(query, fused_candidates, self.corpus, top_k=top_k)
        t_rerank = (time.perf_counter() - t_rerank_start) * 1000.0

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "query": query,
            "mode": mode,
            "total_candidates_retrieved": len(fused_candidates),
            "results": final_results,
            "latency_breakdown_ms": {
                "dense_retrieval": round(t_dense, 3),
                "sparse_bm25": round(t_sparse, 3),
                "rrf_fusion": round(t_fusion, 3),
                "cross_encoder_rerank": round(t_rerank, 3),
                "total_e2e": round(total_latency_ms, 3)
            }
        }

    def get_2d_projection(self, query: str = None) -> Dict[str, Any]:
        """
        Projects high-dimensional vector embeddings into 2D coordinates for real-time visualization.
        """
        from sklearn.decomposition import PCA
        if self.dense_embeddings is None or len(self.dense_embeddings) < 2:
            return {"documents": [], "query": None}

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(self.dense_embeddings)

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        norm_coords = []
        for idx, (x, y) in enumerate(coords):
            nx = ((x - x_min) / max(1e-6, (x_max - x_min))) * 2.0 - 1.0
            ny = ((y - y_min) / max(1e-6, (y_max - y_min))) * 2.0 - 1.0
            doc = self.corpus[idx]
            norm_coords.append({
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "x": round(float(nx), 3),
                "y": round(float(ny), 3)
            })

        query_coord = None
        if query and query.strip():
            q_emb = self.dense.transform([query])
            q_2d = pca.transform(q_emb)[0]
            qx = ((q_2d[0] - x_min) / max(1e-6, (x_max - x_min))) * 2.0 - 1.0
            qy = ((q_2d[1] - y_min) / max(1e-6, (y_max - y_min))) * 2.0 - 1.0
            query_coord = {"x": round(float(qx), 3), "y": round(float(qy), 3)}

        return {
            "documents": norm_coords,
            "query": query_coord
        }

if __name__ == "__main__":
    with open("data/corpus.json", "r") as f:
        docs = json.load(f)
    engine = HybridSearchEngine()
    engine.build_index(docs)
    res = engine.search_hybrid("RAG hallucinations and vector search")
    print("Latency breakdown:", res["latency_breakdown_ms"])
    for r in res["results"][:2]:
        print("-", r["title"], f"(Score: {r['relevance_score']})")
