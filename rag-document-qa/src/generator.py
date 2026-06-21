"""
src/generator.py
Builds a citation-aware prompt from retrieved chunks, calls Gemini,
and maps the model's [n] citation markers back to source chunks
(doc_id, page, text) for display in the UI.

Uses Google's free-tier Gemini API (Flash model) via the google-genai
SDK. Set a GEMINI_API_KEY environment variable (get one free, no
credit card needed, at https://aistudio.google.com/apikey).
"""
import os
import re
from typing import Dict, List, Optional

from google import genai
from google.genai import types

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question "
    "using ONLY the numbered context passages provided below. "
    "Every factual claim must be followed by a citation marker like [1] "
    "or [2] referring to the passage number it came from. "
    "If the answer is not contained in the passages, say "
    "\"I don't have enough information in this document to answer that.\" "
    "Do not use outside knowledge, and do not invent citation numbers "
    "that weren't provided."
)


def build_prompt(question: str, chunks: List[Dict]) -> str:
    context_blocks = [
        f"[{i}] (page {chunk['page']}): {chunk['text']}" for i, chunk in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(context_blocks)
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with [n] citations after each claim):"
    )


def extract_cited_indices(answer: str) -> List[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def call_llm(prompt: str, client: Optional["genai.Client"] = None, model: str = MODEL_NAME) -> str:
    client = client or genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
        ),
    )
    return response.text


def generate_answer(question: str, chunks: List[Dict], client: Optional["genai.Client"] = None) -> Dict:
    """
    Returns: {"answer": str, "citations": [{"id", "doc_id", "page", "text"}, ...]}
    """
    if not chunks:
        return {
            "answer": "I don't have enough information in this document to answer that.",
            "citations": [],
        }

    prompt = build_prompt(question, chunks)
    raw_answer = call_llm(prompt, client=client)

    cited_indices = extract_cited_indices(raw_answer)
    citations = []
    for i in cited_indices:
        if 1 <= i <= len(chunks):
            c = chunks[i - 1]
            citations.append({"id": i, "doc_id": c["doc_id"], "page": c["page"], "text": c["text"]})

    return {"answer": raw_answer, "citations": citations}


if __name__ == "__main__":
    import sys

    from src.retriever import hybrid_search

    if len(sys.argv) < 3:
        print("Usage: python -m src.generator <doc_id> <question> [index_dir]")
        sys.exit(1)

    doc_id, question = sys.argv[1], sys.argv[2]
    index_dir = sys.argv[3] if len(sys.argv) > 3 else "index_store"

    retrieved = hybrid_search(question, doc_id, index_dir, k=5)
    result = generate_answer(question, retrieved)
    print(result["answer"])
    print("\nCitations:")
    for c in result["citations"]:
        print(f"  [{c['id']}] page {c['page']}: {c['text'][:100]}...")
