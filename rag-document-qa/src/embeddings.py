"""
src/embeddings.py
Single source of truth for turning text into vectors, so ingest.py,
retriever.py, and evaluate.py all embed in exactly the same way.
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load (once) and cache the local, free sentence-transformers model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list) -> np.ndarray:
    """
    Embed a list of strings into L2-normalized float32 vectors.
    Normalizing means cosine similarity == inner product, which lets us
    use a fast FAISS IndexFlatIP for retrieval.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")
