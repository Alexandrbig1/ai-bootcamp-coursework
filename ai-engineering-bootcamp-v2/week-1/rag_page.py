"""Week 2 RAG demo page: Ingest, Retrieve and Ask tabs on top of the FastAPI service.

Run this page (from this folder):
  .venv/bin/streamlit run rag_page.py

Every button calls the API at API_BASE_URL from .env (default: local), never OpenAI or Pinecone.
The page shows only "local" or "live", never the URL, so screenshots are safe to share.
"""

from pathlib import Path

import httpx
import streamlit as st

# Same API_BASE_URL lookup and local/live label as the ingest script, so the two always agree.
from ingest_folder import BASE_URL, target_label

TARGET = target_label(BASE_URL)
MODELS = ["gpt-4o-mini", "gpt-4o"]  # the API's allowlist; any other model is a 422
TIMEOUT = 180.0  # a sleeping Render free instance can take a minute to wake up
DEFAULT_QUESTION = "How many days a week can employees work remotely?"
SCORE_COLUMN = st.column_config.ProgressColumn(
    "score", min_value=0.0, max_value=1.0, format="%.3f"
)


def call_api(method: str, path: str, **kwargs) -> tuple[int, dict]:
    """One API call; returns (HTTP status, JSON body), with status 0 if the API is unreachable."""

    try:
        response = httpx.request(method, f"{BASE_URL}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.HTTPError as exc:  # name the error kind only: its message can include the URL
        return 0, {"detail": f"Could not reach the {TARGET} API ({type(exc).__name__})"}
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"detail": "the API sent a response that isn't JSON"}


def show_error(status: int, data: dict) -> None:
    """The API's own message: 400 and 422 explain bad input, 502 an answer that failed checks."""

    detail = data.get("detail") or data.get("error")
    if isinstance(detail, list):  # FastAPI validation errors: one entry per bad field
        detail = "; ".join(f"{error['loc'][-1]}: {error['msg']}" for error in detail)
    st.error(f"HTTP {status}: {detail}" if status else detail)


def usd(value: float) -> str:
    """Dollars to 8 decimals, like the API: embedding a question costs well under $0.000001."""

    return f"${value:.8f}"


def show_answer(data: dict) -> None:
    """The answer or the refusal first, then tokens, cost and latency, then the chunks shown."""

    answer = data["answer"]
    if answer["refused"]:
        st.warning(
            f"**Refused:** the retrieved chunks don't contain the answer.\n\n{answer['answer']}"
        )
    else:
        st.success(answer["answer"])
        citations = " ".join(f"`{chunk_id}`" for chunk_id in answer["citations"])
        st.markdown(f"**Citations:** {citations}")

    tokens_col, embedding_col, cost_col, latency_col, model_col = st.columns(5)
    tokens_col.metric("tokens used", data["tokens_used"])
    embedding_col.metric("embedding tokens", data["embedding_tokens"])
    cost_col.metric("cost", usd(data["cost_usd"]))
    latency_col.metric("latency", f"{data['latency_ms']:,} ms")
    model_col.metric("model", data["model"])

    st.markdown("**Retrieved chunks** (what the model was shown, best first)")
    if not data["retrieved"]:
        st.info("No chunks matched, so the model wasn't called.")
    else:
        cited = set(answer["citations"])
        rows = [
            {**chunk, "rank": rank, "cited": chunk["chunk_id"] in cited}
            for rank, chunk in enumerate(data["retrieved"], 1)
        ]
        st.dataframe(
            rows,
            column_order=["rank", "chunk_id", "document_id", "score", "cited"],
            column_config={"score": SCORE_COLUMN},
            hide_index=True,
        )
    st.caption(f"confidence {answer['confidence']:.2f} · sources_needed {answer['sources_needed']}")
    with st.expander("Raw JSON response"):
        st.json(data)


st.set_page_config(page_title="Week 2 RAG Demo", layout="wide")
st.title("Week 2 — RAG over your documents")
st.caption(
    f"Every button calls the FastAPI service (**{TARGET}**). "
    "Nothing on this page talks to OpenAI or Pinecone directly."
)

st.sidebar.markdown(f"**API target:** :{'green' if TARGET == 'live' else 'blue'}[{TARGET}]")
if st.sidebar.button("Check health", key="health_check"):
    status, data = call_api("GET", "/health")
    with st.sidebar:
        if status == 200:
            st.success(f"{data['vector_count']} vectors in {data['index']} / {data['namespace']}")
        else:
            show_error(status, data)

ingest_tab, retrieve_tab, ask_tab = st.tabs(["Ingest", "Retrieve", "Ask"])

with ingest_tab:
    st.subheader("POST /ingest")
    st.caption("Chunk, embed and upsert one document. Sending a document_id again replaces it.")
    with st.form("ingest_form"):
        upload = st.file_uploader("Upload a .txt file, or paste text below", type=["txt"])
        doc_col, source_col = st.columns(2)
        document_id = doc_col.text_input(
            "document_id", key="ingest_doc", placeholder="a-z, 0-9, - and _, e.g. travel-policy"
        )
        source = source_col.text_input(
            "source (optional)", key="ingest_source", placeholder="defaults to the file name"
        )
        text = st.text_area("text", key="ingest_text", height=200)
        submitted = st.form_submit_button("Ingest", type="primary", key="ingest_submit")
    if submitted:
        if upload is not None:  # an uploaded file wins; empty fields default to its name
            text = upload.getvalue().decode("utf-8", errors="replace")
            document_id = document_id or Path(upload.name).stem
            source = source or upload.name
        payload = {"text": text, "document_id": document_id, "source": source or None}
        with st.spinner("Chunking, embedding and upserting..."):
            st.session_state["ingest_result"] = call_api("POST", "/ingest", json=payload)
    if "ingest_result" in st.session_state:
        status, data = st.session_state["ingest_result"]
        if status == 200:
            st.success(
                f"{data['status'].capitalize()} `{data['document_id']}`: "
                f"{data['chunks_indexed']} chunks indexed"
            )
            tokens_col, cost_col = st.columns(2)
            tokens_col.metric("embedding tokens", data["embedding_tokens"])
            cost_col.metric("cost", usd(data["cost_usd"]))
            chunk_ids = " ".join(f"`{chunk_id}`" for chunk_id in data["chunk_ids"])
            st.markdown(f"**Chunk IDs:** {chunk_ids}")
        else:
            show_error(status, data)

with retrieve_tab:
    st.subheader("GET /debug/retrieve")
    st.caption("Retrieval only, no LLM: the chunks /ask would be shown for this question.")
    with st.form("retrieve_form"):
        question = st.text_input("Question", DEFAULT_QUESTION, key="retrieve_q")
        k_col, doc_col = st.columns(2)
        k = k_col.slider("k (chunks)", 1, 10, 5, key="retrieve_k")
        document_id = doc_col.text_input("document_id filter (optional)", key="retrieve_doc")
        submitted = st.form_submit_button("Retrieve", type="primary", key="retrieve_submit")
    if submitted:
        params = {"q": question, "k": k}
        if document_id:
            params["document_id"] = document_id
        with st.spinner("Embedding the question and searching..."):
            st.session_state["retrieve_result"] = call_api("GET", "/debug/retrieve", params=params)
    if "retrieve_result" in st.session_state:
        status, data = st.session_state["retrieve_result"]
        if status != 200:
            show_error(status, data)
        elif not data["matches"]:
            st.info("No chunks matched: the namespace is empty, or no such document.")
        else:
            st.dataframe(
                [{"rank": rank, **match} for rank, match in enumerate(data["matches"], 1)],
                column_order=["rank", "chunk_id", "source", "score", "preview"],
                column_config={
                    "score": SCORE_COLUMN,
                    "preview": st.column_config.TextColumn("preview", width="large"),
                },
                hide_index=True,
            )
            st.caption(
                f"Question embedding: {data['embedding_tokens']} tokens, {usd(data['cost_usd'])}"
            )

with ask_tab:
    st.subheader("POST /ask")
    st.caption(
        "Retrieve, ground, generate: the answer cites chunk IDs, or refuses when the documents "
        "don't say. Try 'What is the mileage rate?', or 'What is the parental leave policy?' "
        "(not in the docs, so it must refuse)."
    )
    with st.form("ask_form"):
        question = st.text_input("Question", DEFAULT_QUESTION, key="ask_q")
        model_col, k_col, doc_col = st.columns(3)
        model = model_col.selectbox("model", MODELS, key="ask_model")
        top_k = k_col.slider("top_k", 1, 10, 5, key="ask_top_k")
        document_id = doc_col.text_input("document_id filter (optional)", key="ask_doc")
        force_bad = st.checkbox(
            "force_bad: break the first attempt on purpose (the retry should recover)",
            key="ask_force_bad",
        )
        submitted = st.form_submit_button("Ask", type="primary", key="ask_submit")
    if submitted:
        payload = {"question": question, "model": model, "top_k": top_k, "force_bad": force_bad}
        if document_id:
            payload["document_id"] = document_id
        with st.spinner("Retrieving chunks and asking the model..."):
            st.session_state["ask_result"] = call_api("POST", "/ask", json=payload)
    if "ask_result" in st.session_state:
        status, data = st.session_state["ask_result"]
        if status == 200:
            show_answer(data)
        else:
            show_error(status, data)
