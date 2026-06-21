"""
tests/test_groundedness.py
Sanity-checks the evaluation framework: an answer that closely matches
a retrieved chunk should score as grounded; an unrelated claim should
score as ungrounded.
"""
from src.evaluate import groundedness_score, split_sentences

CHUNKS = [
    {
        "chunk_id": "c1",
        "doc_id": "d",
        "page": 1,
        "text": "The company reported revenue of fifty million dollars in Q3.",
    }
]


def test_grounded_answer_scores_high():
    answer = "The company reported revenue of fifty million dollars in Q3. [1]"
    result = groundedness_score(answer, CHUNKS)
    assert result["overall_grounded_ratio"] == 1.0


def test_ungrounded_answer_scores_low():
    answer = "The moon landing happened in 1969."
    result = groundedness_score(answer, CHUNKS)
    assert result["overall_grounded_ratio"] == 0.0


def test_split_sentences_strips_citation_markers():
    sentences = split_sentences("First claim. [1] Second claim. [2]")
    assert sentences == ["First claim.", "Second claim."]


def test_empty_answer_or_no_chunks_returns_zero():
    assert groundedness_score("", CHUNKS)["overall_grounded_ratio"] == 0.0
    assert groundedness_score("Some answer.", [])["overall_grounded_ratio"] == 0.0
