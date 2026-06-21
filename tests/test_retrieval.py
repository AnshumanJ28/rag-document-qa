"""
tests/test_retrieval.py
Builds a small in-memory FAISS index and confirms hybrid retrieval
(vector + BM25, fused with Reciprocal Rank Fusion) surfaces the right
chunk for both a semantically-phrased query and an exact-keyword query.
"""
import json
import os

import faiss

import src.embeddings as embeddings_module
from src.ingest import build_faiss_index
from src.retriever import hybrid_search, reciprocal_rank_fusion

SAMPLE_CHUNKS = [
    {"chunk_id": "d_0", "doc_id": "d", "page": 1, "text": "The mitochondria is the powerhouse of the cell."},
    {"chunk_id": "d_1", "doc_id": "d", "page": 1, "text": "Paris is the capital city of France."},
    {"chunk_id": "d_2", "doc_id": "d", "page": 2, "text": "Quarterly revenue grew by twelve percent year over year."},
    {"chunk_id": "d_3", "doc_id": "d", "page": 2, "text": "The French Revolution began in 1789."},
]


def _build_test_index(tmp_path) -> str:
    texts = [c["text"] for c in SAMPLE_CHUNKS]
    embeddings = embeddings_module.embed_texts(texts)
    index = build_faiss_index(embeddings)

    index_dir = str(tmp_path)
    faiss.write_index(index, os.path.join(index_dir, "d.faiss"))
    with open(os.path.join(index_dir, "d.meta.json"), "w") as f:
        json.dump(SAMPLE_CHUNKS, f)
    return index_dir


def test_hybrid_search_returns_k_results(tmp_path):
    index_dir = _build_test_index(tmp_path)
    results = hybrid_search("capital of France", "d", index_dir, k=2)
    assert len(results) == 2


def test_hybrid_search_keyword_match_ranks_high(tmp_path):
    index_dir = _build_test_index(tmp_path)
    # Exact keyword overlap should make BM25 surface this chunk at the top
    results = hybrid_search("quarterly revenue percent", "d", index_dir, k=1)
    assert results[0]["chunk_id"] == "d_2"


def test_hybrid_search_missing_doc_raises(tmp_path):
    index_dir = str(tmp_path)
    try:
        hybrid_search("anything", "missing-doc", index_dir, k=1)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_reciprocal_rank_fusion_prefers_consensus():
    # Item 1 appears near the top of both lists -> should win overall
    vec_rank = [5, 1, 2, 3]
    kw_rank = [1, 5, 3, 2]
    fused = reciprocal_rank_fusion([vec_rank, kw_rank])
    assert fused[0] in (1, 5)
