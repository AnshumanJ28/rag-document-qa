"""
tests/conftest.py
We replace the real sentence-transformer embedding call with a small
deterministic, hashing-based embedding. This keeps the test suite fast
and fully offline (no model download needed in CI) while still giving
semantically-similar test strings closer vectors than unrelated ones,
so retrieval ranking tests remain meaningful.
"""
import numpy as np
import pytest

import src.embeddings as embeddings_module

FAKE_DIM = 32


def _fake_embed_texts(texts):
    vectors = []
    for text in texts:
        base = np.zeros(FAKE_DIM, dtype="float64")
        for word in text.lower().split():
            word_rng = np.random.default_rng(abs(hash(word)) % (2**32))
            base += word_rng.normal(size=FAKE_DIM)
        if np.allclose(base, 0):
            fallback_rng = np.random.default_rng(abs(hash(text)) % (2**32))
            base = fallback_rng.normal(size=FAKE_DIM)
        base = base / (np.linalg.norm(base) + 1e-8)
        vectors.append(base.astype("float32"))
    return np.vstack(vectors)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(embeddings_module, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr("src.ingest.embed_texts", _fake_embed_texts)
    monkeypatch.setattr("src.retriever.embed_texts", _fake_embed_texts)
    monkeypatch.setattr("src.evaluate.embed_texts", _fake_embed_texts)
    yield
