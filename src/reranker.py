# src/reranker.py
from typing import List, Dict, Any, Tuple
from collections import defaultdict

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Combines disparate ranking lists (e.g. Dense Vector and BM25 Sparse)
    using the Reciprocal Rank Fusion (RRF) algorithm:
        RRF_Score(d) = sum( 1.0 / (k + rank_m(d)) )
    Parameters:
        k: Smoothing constant (default 60, optimal according to Cormack et al.)
    """
    rrf_scores = defaultdict(float)

    for r_list in ranked_lists:
        for rank, (doc_idx, _) in enumerate(r_list, start=1):
            rrf_scores[doc_idx] += 1.0 / (k + rank)

    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs

class CrossEncoderReranker:
    """
    Simulates cross-encoder second-stage re-ranking.
    Computes bidirectional cross-token interaction scores between query and document candidate pairs.
    """
    def __init__(self, boost_factor: float = 1.25):
        self.boost_factor = boost_factor

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float]],
        corpus: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        reranked = []

        for doc_idx, fusion_score in candidates:
            doc = corpus[doc_idx]
            text = f"{doc['title']} {doc['content']}".lower()

            # Cross-token exact sequence match & title alignment boost
            exact_matches = sum(1 for w in query_words if w in text)
            title_boost = 1.3 if any(w in doc["title"].lower() for w in query_words) else 1.0

            # Composite calibrated relevance score
            relevance = (fusion_score * 100.0) * title_boost * (1.0 + (exact_matches * 0.15))

            reranked.append({
                "doc_idx": doc_idx,
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "content": doc["content"],
                "relevance_score": round(float(relevance), 4),
                "fusion_rank_score": round(float(fusion_score), 5)
            })

        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return reranked[:top_k]
