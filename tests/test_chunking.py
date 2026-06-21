"""
tests/test_chunking.py
Verifies the chunking strategy: chunks respect the configured size,
adjacent chunks actually overlap, and page numbers are preserved
correctly for citation purposes.
"""
from src.ingest import chunk_pages


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    pages = [{"page": 1, "text": text}]
    chunks = chunk_pages(pages, doc_id="doc1", chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= 220  # small slack for splitter boundary rules
        assert c["doc_id"] == "doc1"
        assert c["page"] == 1


def test_chunking_overlap_shares_content():
    text = " ".join(f"word{i}" for i in range(500))
    pages = [{"page": 1, "text": text}]
    chunks = chunk_pages(pages, doc_id="doc1", chunk_size=200, chunk_overlap=50)

    first_words = set(chunks[0]["text"].split())
    second_words = set(chunks[1]["text"].split())
    assert first_words & second_words, "Expected overlapping content between adjacent chunks"


def test_chunking_preserves_page_numbers():
    pages = [
        {"page": 1, "text": "Page one content. " * 20},
        {"page": 2, "text": "Page two content. " * 20},
    ]
    chunks = chunk_pages(pages, doc_id="doc1", chunk_size=100, chunk_overlap=20)
    pages_seen = {c["page"] for c in chunks}
    assert pages_seen == {1, 2}


def test_empty_page_is_skipped():
    pages = [{"page": 1, "text": "   "}, {"page": 2, "text": "Real content here."}]
    chunks = chunk_pages(pages, doc_id="doc1")
    assert all(c["page"] == 2 for c in chunks)


def test_chunk_ids_are_unique_and_ordered():
    pages = [{"page": 1, "text": "word " * 300}]
    chunks = chunk_pages(pages, doc_id="docX", chunk_size=200, chunk_overlap=50)
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunk_ids == [f"docX_{i}" for i in range(len(chunks))]
