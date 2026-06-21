"""
tests/test_generator.py
Tests prompt construction and citation parsing without making a real
Gemini API call (the client is mocked), so this suite runs in CI
without needing a GEMINI_API_KEY secret.
"""
from unittest.mock import MagicMock

from src.generator import build_prompt, extract_cited_indices, generate_answer

CHUNKS = [
    {"chunk_id": "c1", "doc_id": "d", "page": 1, "text": "Revenue grew 12% in Q3."},
    {"chunk_id": "c2", "doc_id": "d", "page": 2, "text": "Costs decreased by 5%."},
]


def test_build_prompt_includes_numbered_context_and_question():
    prompt = build_prompt("How did revenue change?", CHUNKS)
    assert "[1]" in prompt and "[2]" in prompt
    assert "How did revenue change?" in prompt
    assert "Revenue grew 12%" in prompt


def test_extract_cited_indices_deduplicates_and_sorts():
    answer = "Revenue grew [1] while costs fell [2] and again [2]."
    assert extract_cited_indices(answer) == [1, 2]


def test_generate_answer_maps_citations_to_chunks():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = MagicMock(text="Revenue grew 12% in Q3 [1].")
    result = generate_answer("How did revenue change?", CHUNKS, client=fake_client)
    assert result["citations"] == [
        {"id": 1, "doc_id": "d", "page": 1, "text": "Revenue grew 12% in Q3."}
    ]


def test_generate_answer_with_no_chunks_returns_fallback():
    result = generate_answer("Anything?", [])
    assert "don't have enough information" in result["answer"]
    assert result["citations"] == []
