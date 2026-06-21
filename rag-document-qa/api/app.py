"""
api/app.py
FastAPI service exposing the RAG pipeline.

  POST /ingest  - upload a document, build and persist its index
  POST /ask     - ask a question against an already-ingested doc_id,
                  OR upload a file and ask in one call

Run from the repo root with:
  uvicorn api.app:app --reload --port 8000
"""
import os
import shutil
import tempfile
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.evaluate import groundedness_score
from src.generator import generate_answer
from src.ingest import ingest_document
from src.retriever import hybrid_search

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "index_store")

app = FastAPI(title="RAG Document QA API")


class Citation(BaseModel):
    id: int
    doc_id: str
    page: int
    text: str


class AskResponse(BaseModel):
    doc_id: str
    answer: str
    citations: List[Citation]
    grounded_ratio: float


def _save_upload_to_tmp(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())[:8]
    tmp_path = _save_upload_to_tmp(file)
    try:
        result = ingest_document(tmp_path, doc_id, INDEX_DIR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(tmp_path)
    return {"doc_id": doc_id, **result}


@app.post("/ask", response_model=AskResponse)
async def ask(
    question: str = Form(...),
    doc_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    k: int = Form(5),
):
    if not doc_id and not file:
        raise HTTPException(status_code=400, detail="Provide either doc_id or a file to ingest.")

    if file is not None:
        doc_id = str(uuid.uuid4())[:8]
        tmp_path = _save_upload_to_tmp(file)
        try:
            ingest_document(tmp_path, doc_id, INDEX_DIR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            os.remove(tmp_path)

    try:
        chunks = hybrid_search(question, doc_id, INDEX_DIR, k=k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = generate_answer(question, chunks)
    grounding = groundedness_score(result["answer"], chunks)

    return AskResponse(
        doc_id=doc_id,
        answer=result["answer"],
        citations=result["citations"],
        grounded_ratio=grounding["overall_grounded_ratio"],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
