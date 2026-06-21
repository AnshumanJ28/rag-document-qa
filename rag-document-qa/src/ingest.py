"""
src/ingest.py
Parses documents (PDF/TXT), chunks them with overlap, embeds the
chunks, and persists them as a FAISS index + a JSON metadata sidecar
(used later for both citations and BM25 keyword search).
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import faiss
import fitz  # PyMuPDF
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embeddings import embed_texts

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_pages(file_path: str) -> List[Dict]:
    """Return [{"page": int, "text": str}, ...] for a PDF or TXT/MD file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(file_path)
        pages = [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]
        doc.close()
        return pages
    elif suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [{"page": 1, "text": text}]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def chunk_pages(
    pages: List[Dict],
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split each page's text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter. We split per-page (rather than on
    the whole concatenated document) so every chunk keeps a single,
    accurate page number for citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_idx = 0
    for page in pages:
        if not page["text"].strip():
            continue
        for piece in splitter.split_text(page["text"]):
            if not piece.strip():
                continue
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{chunk_idx}",
                    "doc_id": doc_id,
                    "page": page["page"],
                    "text": piece.strip(),
                }
            )
            chunk_idx += 1
    return chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Inner-product index over L2-normalized vectors == cosine similarity search."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def ingest_document(file_path: str, doc_id: str, index_dir: str) -> Dict:
    """
    Full ingestion pipeline: parse -> chunk -> embed -> persist.
    Writes:
      <index_dir>/<doc_id>.faiss       (vector index)
      <index_dir>/<doc_id>.meta.json   (chunk text + page/doc metadata)
    """
    os.makedirs(index_dir, exist_ok=True)

    pages = extract_pages(file_path)
    chunks = chunk_pages(pages, doc_id)
    if not chunks:
        raise ValueError("No extractable text found in document.")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, os.path.join(index_dir, f"{doc_id}.faiss"))
    with open(os.path.join(index_dir, f"{doc_id}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return {"doc_id": doc_id, "num_chunks": len(chunks), "num_pages": len(pages)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.ingest <file_path> <doc_id> [index_dir]")
        sys.exit(1)

    file_path, doc_id = sys.argv[1], sys.argv[2]
    index_dir = sys.argv[3] if len(sys.argv) > 3 else "index_store"
    result = ingest_document(file_path, doc_id, index_dir)
    print(result)
