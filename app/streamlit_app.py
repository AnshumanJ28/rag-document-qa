"""
app/streamlit_app.py
Upload a document, ask questions, see grounded answers with source
citations. Talks to the FastAPI service over HTTP.

Run with:
  streamlit run app/streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Document QA", page_icon="📄")
st.title("📄 RAG Document QA")
st.caption("Upload a PDF or text document, then ask questions grounded in its content.")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None

uploaded = st.file_uploader("Upload a PDF or text document", type=["pdf", "txt", "md"])

if uploaded is not None and uploaded.name != st.session_state.filename:
    with st.spinner("Parsing, chunking, and indexing your document..."):
        try:
            resp = requests.post(
                f"{API_URL}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue())},
                timeout=120,
            )
        except requests.exceptions.ConnectionError:
            st.error(f"Could not reach the API at {API_URL}. Is it running?")
            resp = None

    if resp is not None:
        if resp.ok:
            data = resp.json()
            st.session_state.doc_id = data["doc_id"]
            st.session_state.filename = uploaded.name
            st.success(f"Indexed {data['num_chunks']} chunks across {data['num_pages']} page(s).")
        else:
            st.error(f"Ingestion failed: {resp.text}")

if st.session_state.doc_id:
    question = st.text_input("Ask a question about the document")
    if st.button("Ask") and question:
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    data={"question": question, "doc_id": st.session_state.doc_id, "k": 5},
                    timeout=120,
                )
            except requests.exceptions.ConnectionError:
                st.error(f"Could not reach the API at {API_URL}. Is it running?")
                resp = None

        if resp is not None:
            if resp.ok:
                data = resp.json()
                st.markdown("### Answer")
                st.write(data["answer"])
                st.caption(
                    f"Groundedness score: {data['grounded_ratio']:.0%} of sentences "
                    "supported by retrieved text"
                )
                if data["citations"]:
                    st.markdown("### Sources")
                    for c in data["citations"]:
                        with st.expander(f"[{c['id']}] Page {c['page']}"):
                            st.write(c["text"])
            else:
                st.error(f"Request failed: {resp.text}")
else:
    st.info("Upload a document above to get started.")
