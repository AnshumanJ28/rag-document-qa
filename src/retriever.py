"""
src/retriever.py
Hybrid retrieval: dense FAISS vector similarity + sparse BM25 keyword
search, combined with Reciprocal Rank Fusion (RRF).

Why hybrid: vector search is great at "meaning" matches (paraphrases,
synonyms) but can miss exact terms like product codes, names, or
numbers. BM25 is the opposite. Fusing both rankings (rather than
trying to normalize and average two incomparable score scales) gives
a single robust ranking.
"""
import json
import os
from typing import Dict, List, Tuple

import faiss
from rank_bm25 import BM25Okapi

from src.embeddings import embed_texts


def load_index(doc_id: str, index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
    index_path = os.path.join(index_dir, f"{doc_id}.faiss")
    meta_path = os.path.join(index_dir, f"{doc_id}.meta.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"No index found for doc_id='{doc_id}' in '{index_dir}'")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def vector_search(query: str, index: faiss.Index, chunks: List[Dict], k: int) -> List[Tuple[int, float]]:
    query_emb = embed_texts([query])
    scores, indices = index.search(query_emb, min(k, len(chunks)))
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]


def keyword_search(query: str, chunks: List[Dict], k: int) -> List[Tuple[int, float]]:
    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(idx, float(scores[idx])) for idx in ranked[:k]]


def reciprocal_rank_fusion(rank_lists: List[List[int]], k_rrf: float = 60.0) -> List[int]:
    """
    Combine several ranked lists of chunk indices into one consensus
    ranking. Using rank position (not raw score) avoids having to
    normalize across incompatible scales (cosine similarity vs. BM25).
    """
    fused_scores: Dict[int, float] = {}
    for ranked_indices in rank_lists:
        for rank, idx in enumerate(ranked_indices):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)


def hybrid_search(
    query: str,
    doc_id: str,
    index_dir: str,
    k: int = 5,
    candidate_pool: int = 20,
) -> List[Dict]:
    """Retrieve the top-k chunks for a query using vector + BM25 fusion."""
    index, chunks = load_index(doc_id, index_dir)

    vec_results = vector_search(query, index, chunks, candidate_pool)
    kw_results = keyword_search(query, chunks, candidate_pool)

    vec_rank = [idx for idx, _ in vec_results]
    kw_rank = [idx for idx, _ in kw_results]
    fused_order = reciprocal_rank_fusion([vec_rank, kw_rank])

    results = []
    for idx in fused_order[:k]:
        chunk = dict(chunks[idx])
        results.append(chunk)
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.retriever <doc_id> <query> [index_dir] [k]")
        sys.exit(1)

    doc_id, query = sys.argv[1], sys.argv[2]
    index_dir = sys.argv[3] if len(sys.argv) > 3 else "index_store"
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    for r in hybrid_search(query, doc_id, index_dir, k):
        print(f"[p.{r['page']}] {r['text'][:120]}...")
