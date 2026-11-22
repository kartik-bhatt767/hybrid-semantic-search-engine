# benchmarks/run_benchmark.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.engine import HybridSearchEngine

EVALUATION_QUERIES = [
    ("Raft consensus and leader election in distributed systems", "doc_001"),
    ("Self-attention mechanisms in transformer neural networks", "doc_002"),
    ("Approximate nearest neighbor graph search algorithms", "doc_003"),
    ("Building fast async microservices in python", "doc_004"),
    ("Mitigating hallucinations with retrieval augmented generation", "doc_005"),
    ("Cross-encoder second stage re-ranking models", "doc_006"),
    ("LSM trees versus B-Trees write throughput", "doc_007"),
    ("Adam optimizer learning rate schedules", "doc_008"),
    ("Kubernetes service mesh Envoy sidecar architecture", "doc_009"),
    ("Neural network pruning and INT8 quantization", "doc_010"),
    ("Distributed atomic transactions with two-phase commit protocol", "doc_011"),
    ("Redis cache-aside and cache invalidation strategies", "doc_012"),
    ("Inverted index BM25 term frequency saturation", "doc_013"),
    ("Zero knowledge proofs zk-SNARKs privacy", "doc_014"),
    ("Kafka partition commit log exactly once processing", "doc_015"),
    ("Parameter efficient fine tuning LoRA QLoRA", "doc_016"),
    ("B+ Tree database indexing and query explain plans", "doc_017"),
    ("QUIC protocol UDP HTTP/3 flow control", "doc_018"),
    ("Product quantization vector compression memory reduction", "doc_019"),
    ("OAuth2 JWT stateless tokens RS256 authorization", "doc_020"),
    ("CAP theorem PACELC trade-offs in distributed databases", "doc_021"),
    ("Graph neural networks neighborhood message passing GCN", "doc_022"),
    ("CI CD pipeline automation GitHub Actions multi-stage Docker", "doc_023"),
    ("Virtual memory page tables and translation lookaside buffer TLB", "doc_024"),
    ("Reciprocal rank fusion multi-engine ranking", "doc_025"),
]

def compute_mrr_and_ndcg(retrieved_ids, target_id, k=10):
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id == target_id:
            mrr = 1.0 / rank
            break

    # NDCG@k (Binary relevance: 1 if target else 0)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id == target_id else 0.0
        dcg += rel / np.log2(rank + 1)
    idcg = 1.0 / np.log2(1 + 1) # Optimal rank 1
    ndcg = dcg / idcg

    return mrr, ndcg

def run_benchmarks():
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    engine = HybridSearchEngine(embedding_dim=32)
    engine.build_index(corpus)

    strategies = ["Sparse BM25", "Dense Vector", "Hybrid + RRF"]
    metrics = {s: {"mrr": [], "ndcg": [], "latencies": []} for s in strategies}

    print("Running Information Retrieval & Latency Benchmarks...")

    for query, target_id in EVALUATION_QUERIES:
        # 1. BM25
        t0 = time.perf_counter()
        bm25_res = engine.search_sparse(query, top_k=10)
        t_bm25 = (time.perf_counter() - t0) * 1000.0
        bm25_ids = [corpus[idx]["id"] for idx, _ in bm25_res]
        mrr, ndcg = compute_mrr_and_ndcg(bm25_ids, target_id)
        metrics["Sparse BM25"]["mrr"].append(mrr)
        metrics["Sparse BM25"]["ndcg"].append(ndcg)
        metrics["Sparse BM25"]["latencies"].append(t_bm25)

        # 2. Dense
        t0 = time.perf_counter()
        dense_res = engine.search_dense(query, top_k=10)
        t_dense = (time.perf_counter() - t0) * 1000.0
        dense_ids = [corpus[idx]["id"] for idx, _ in dense_res]
        mrr, ndcg = compute_mrr_and_ndcg(dense_ids, target_id)
        metrics["Dense Vector"]["mrr"].append(mrr)
        metrics["Dense Vector"]["ndcg"].append(ndcg)
        metrics["Dense Vector"]["latencies"].append(t_dense)

        # 3. Hybrid
        t0 = time.perf_counter()
        hybrid_res = engine.search_hybrid(query, top_k=10)
        t_hybrid = (time.perf_counter() - t0) * 1000.0
        hybrid_ids = [r["id"] for r in hybrid_res["results"]]
        mrr, ndcg = compute_mrr_and_ndcg(hybrid_ids, target_id)
        metrics["Hybrid + RRF"]["mrr"].append(mrr)
        metrics["Hybrid + RRF"]["ndcg"].append(ndcg)
        metrics["Hybrid + RRF"]["latencies"].append(t_hybrid)

    # Summarize
    summary = {}
    print("\n" + "=" * 65)
    print(f"{'Strategy':<18} | {'MRR@10':<10} | {'NDCG@10':<10} | {'p95 Latency (ms)':<15}")
    print("=" * 65)
    for s in strategies:
        avg_mrr = np.mean(metrics[s]["mrr"])
        avg_ndcg = np.mean(metrics[s]["ndcg"])
        p95_lat = np.percentile(metrics[s]["latencies"], 95)
        summary[s] = {"MRR": avg_mrr, "NDCG": avg_ndcg, "p95": p95_lat}
        print(f"{s:<18} | {avg_mrr:.4f}     | {avg_ndcg:.4f}     | {p95_lat:.3f} ms")
    print("=" * 65)

    # Plot Benchmarks
    _render_chart(summary)
    return summary

def _render_chart(summary):
    strategies = list(summary.keys())
    mrr_vals = [summary[s]["MRR"] for s in strategies]
    ndcg_vals = [summary[s]["NDCG"] for s in strategies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    colors = ["#38bdf8", "#818cf8", "#34d399"]

    # Plot 1: Information Retrieval Quality (NDCG & MRR)
    ax1.set_facecolor("#1e293b")
    x = np.arange(len(strategies))
    w = 0.35
    b1 = ax1.bar(x - w/2, mrr_vals, w, label="MRR@10", color="#38bdf8", edgecolor="white", alpha=0.9)
    b2 = ax1.bar(x + w/2, ndcg_vals, w, label="NDCG@10", color="#a78bfa", edgecolor="white", alpha=0.9)
    ax1.set_title("Information Retrieval Quality (NDCG@10 & MRR@10)", color="white", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, color="#cbd5e1", fontweight="bold")
    ax1.tick_params(colors="#cbd5e1")
    ax1.set_ylim(0, 1.15)
    ax1.legend(facecolor="#334155", edgecolor="none", labelcolor="white")
    ax1.grid(color="#334155", linestyle="--", alpha=0.5)

    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{bar.get_height():.2f}",
                 ha="center", color="white", fontsize=9, fontweight="bold")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{bar.get_height():.2f}",
                 ha="center", color="white", fontsize=9, fontweight="bold")

    # Plot 2: Latency Profiles
    ax2.set_facecolor("#1e293b")
    p95_vals = [summary[s]["p95"] for s in strategies]
    b3 = ax2.bar(strategies, p95_vals, color=["#60a5fa", "#c084fc", "#4ade80"], edgecolor="white", alpha=0.9, width=0.5)
    ax2.set_title("P95 Latency Profile (Sub-Millisecond Engine)", color="white", fontsize=12, fontweight="bold", pad=10)
    ax2.tick_params(colors="#cbd5e1")
    ax2.set_ylabel("Latency (milliseconds)", color="#94a3b8")
    ax2.grid(color="#334155", linestyle="--", alpha=0.5)

    for bar in b3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04, f"{bar.get_height():.2f} ms",
                 ha="center", color="white", fontsize=9, fontweight="bold")

    plt.suptitle("Hybrid Semantic Search — Benchmark Evaluation", color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "..", "assets", "benchmark_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"✅ Benchmark chart saved to {out_path}")

if __name__ == "__main__":
    run_benchmarks()
