# src/embeddings.py
import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class BM25Encoder:
    """
    Production-grade Okapi BM25 sparse lexical scoring engine.
    Formula:
        Score(D, Q) = sum( IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl))) )
    Parameters:
        k1 = 1.5 (term frequency saturation)
        b  = 0.75 (document length penalization)
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.corpus_size = 0

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())

    def fit(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        total_length = 0
        df = Counter()

        self.doc_freqs = []
        self.doc_len = []

        for doc in corpus:
            tokens = self.tokenize(doc)
            self.doc_len.append(len(tokens))
            total_length += len(tokens)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for token in freqs.keys():
                df[token] += 1

        self.avgdl = total_length / max(1, self.corpus_size)

        # Compute Robertson-Spärck Jones IDF
        for word, count in df.items():
            self.idf[word] = math.log((self.corpus_size - count + 0.5) / (count + 0.5) + 1.0)

    def score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        doc_freq = self.doc_freqs[doc_idx]
        dl = self.doc_len[doc_idx]

        for token in query_tokens:
            if token not in doc_freq:
                continue
            tf = doc_freq[token]
            idf = self.idf.get(token, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokens = self.tokenize(query)
        scores = [(idx, self.score_document(tokens, idx)) for idx in range(self.corpus_size)]
        scores = [s for s in scores if s[1] > 0.0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class DenseVectorEncoder:
    """
    Generates normalized continuous dense semantic embeddings using Latent Semantic Projection.
    Produces low-dimensional unit vectors enabling fast Cosine Similarity Inner Product search.
    """
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=1000)
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=42)

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        tfidf_mat = self.tfidf.fit_transform(corpus)
        dim = min(self.embedding_dim, tfidf_mat.shape[1] - 1)
        if dim < 2:
            dim = 2
        self.svd.n_components = dim
        embeddings = self.svd.fit_transform(tfidf_mat)
        # L2 Normalization for Cosine Similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def transform(self, texts: List[str]) -> np.ndarray:
        tfidf_mat = self.tfidf.transform(texts)
        embeddings = self.svd.transform(tfidf_mat)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
