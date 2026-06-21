"""
src/evaluate.py
A lightweight groundedness check that needs no extra LLM call: for
every sentence in the generated answer, find its highest cosine
similarity to any retrieved chunk. Sentences far from all retrieved
text are flagged as potentially unsupported (hallucinated).

This is the kind of automatic regression check you'd wire into CI or
into a "trust score" shown next to each answer in the UI.
"""
import re
from typing import Dict, List

from src.embeddings import embed_texts

GROUNDED_THRESHOLD = 0.45  # cosine similarity cutoff; tune per domain


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\[\d+\]", "", text)  # strip citation markers before scoring
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def groundedness_score(answer: str, chunks: List[Dict]) -> Dict:
    """
    Returns:
      {
        "overall_grounded_ratio": float,  # fraction of sentences considered grounded
        "sentence_scores": [{"sentence", "max_similarity", "grounded"}, ...]
      }
    """
    sentences = split_sentences(answer)
    if not sentences or not chunks:
        return {"overall_grounded_ratio": 0.0, "sentence_scores": []}

    sentence_embs = embed_texts(sentences)
    chunk_embs = embed_texts([c["text"] for c in chunks])

    sims = sentence_embs @ chunk_embs.T  # cosine similarity (embeddings are normalized)
    best_per_sentence = sims.max(axis=1)

    sentence_scores = [
        {
            "sentence": s,
            "max_similarity": float(score),
            "grounded": bool(score >= GROUNDED_THRESHOLD),
        }
        for s, score in zip(sentences, best_per_sentence)
    ]
    grounded_ratio = sum(s["grounded"] for s in sentence_scores) / len(sentence_scores)
    return {"overall_grounded_ratio": grounded_ratio, "sentence_scores": sentence_scores}
