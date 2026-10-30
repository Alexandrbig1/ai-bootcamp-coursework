"""Week 1 live demo — five stages in one file, built up live in class.

Week 2 grounds /ask in ingested documents (RAG) and adds /ingest, /debug/retrieve and /health;
chunking, embeddings, Pinecone and the grounding prompt live in rag.py.
"""

import time
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

import rag

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Stage 4 default — cheap model so a public URL can't run up the bill; swap at request time.
DEFAULT_MODEL = "gpt-4o-mini"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
# Doubles as the allowlist: /ask rejects any model that is not priced here.
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
}

# Week 2 — input limits shared by /ask, /ingest and /debug/retrieve.
MAX_QUESTION_CHARS = 1000
MAX_INGEST_CHARS = 100_000
SLUG_PATTERN = r"^[a-z0-9_-]+$"  # no "#": it separates the document_id from the chunk number


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    # Week 2 — grounding: the chunks the answer came from, or a refusal.
    citations: list[str]
    refused: bool


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str = Field(max_length=MAX_QUESTION_CHARS)  # Week 2 — capped
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.
    # Week 2 — retrieval knobs: how many chunks to fetch, optionally from one document only.
    top_k: int = Field(default=5, ge=1, le=10)
    document_id: str | None = Field(default=None, pattern=SLUG_PATTERN, max_length=100)


class RetrievedRef(BaseModel):
    """One chunk the model was shown, best match first."""

    chunk_id: str
    document_id: str
    score: float


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float  # Week 2 — now includes the question's embedding
    # Week 2 — what the model was shown, and the question's embedding tokens.
    retrieved: list[RetrievedRef]
    embedding_tokens: int


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(messages: rag.Messages, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    Week 2: takes grounding messages (rules, chunks, question) rather than a bare question.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(messages: rag.Messages, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            *messages,
            {
                "role": "user",
                "content": (
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed, "
                    "citations, refused. Set confidence to the string 'very high' (not a number)."
                ),
            },
        ],
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer one question from the ingested docs: retrieval, grounding, guardrails, cost."""

    model = body.model or DEFAULT_MODEL
    # Wallet guard: reject unpriced models before any tokens are spent.
    if model not in MODEL_PRICES_PER_1K:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported model '{model}'. Allowed: {', '.join(MODEL_PRICES_PER_1K)}",
        )
    # Week 2 — a blank question has nothing to embed.
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question is empty or whitespace-only")

    # Week 2 — one timer for the whole request: embed, retrieve, generate, retries included.
    start = time.perf_counter()
    matches, embedding_tokens = rag.retrieve(
        body.question, top_k=body.top_k, document_id=body.document_id
    )
    embedding_cost = rag.embedding_cost_usd(embedding_tokens)
    retrieved = [
        RetrievedRef(chunk_id=m.chunk_id, document_id=m.document_id, score=round(m.score, 4))
        for m in matches
    ]

    # Nothing to ground on (empty namespace, or no such document): refuse without the model.
    if not matches:
        return AskResponse(
            answer=Answer(
                answer=rag.REFUSAL_TEXT,
                confidence=0.0,
                sources_needed=True,
                citations=[],
                refused=True,
            ),
            tokens_used=0,
            model=model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cost_usd=round(embedding_cost, 8),
            retrieved=[],
            embedding_tokens=embedding_tokens,
        )

    messages = rag.grounding_messages(body.question, matches)
    retrieved_ids = {m.chunk_id for m in matches}
    last_error: str | None = None

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    messages, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    messages, model
                )

            # Week 2 — valid JSON can still cite chunks it never saw or refuse sloppily;
            # check_grounding raises ValueError, so those answers get the same one retry.
            rag.check_grounding(answer.answer, answer.citations, answer.refused, retrieved_ids)

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens) + embedding_cost

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 8),
                retrieved=retrieved,
                embedding_tokens=embedding_tokens,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed validation after retry: {last_error}",
    )


# Week 2 — ingest: plain text in; chunked, embedded and upserted to Pinecone (see rag.py).
class IngestRequest(BaseModel):
    """One document per call; sending the same document_id again replaces it."""

    text: str = Field(max_length=MAX_INGEST_CHARS)
    document_id: str = Field(pattern=SLUG_PATTERN, max_length=100)  # becomes the chunk ID prefix
    source: str | None = Field(default=None, max_length=200)


class IngestResponse(BaseModel):
    """What was written, including the chunk IDs that /ask will cite."""

    document_id: str
    chunks_indexed: int
    status: str  # "created" or "replaced"
    chunk_ids: list[str]
    embedding_tokens: int
    cost_usd: float


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk, embed and upsert one document, replacing any earlier version of it."""

    chunks = rag.chunk_text(body.text)
    # Malformed fields are FastAPI's 422; well-formed text with nothing to index is a 400.
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="text is empty or whitespace-only: nothing to index",
        )

    chunk_ids, replaced, embedding_tokens = rag.upsert_document(
        body.document_id,
        chunks,
        # Pinecone metadata can't hold null, so a missing source falls back to the document_id.
        source=body.source or body.document_id,
    )
    return IngestResponse(
        document_id=body.document_id,
        chunks_indexed=len(chunk_ids),
        status="replaced" if replaced else "created",
        chunk_ids=chunk_ids,
        embedding_tokens=embedding_tokens,
        # 8 decimals, not Week 1's 6: a short embedding costs less than $0.000001.
        cost_usd=round(rag.embedding_cost_usd(embedding_tokens), 8),
    )


# Week 2 — retrieval only, no LLM: prove the right chunks come back before /ask relies on them.
PREVIEW_CHARS = 300


class RetrievedChunk(BaseModel):
    """One match, with enough text to judge whether it is the right chunk."""

    chunk_id: str
    document_id: str
    source: str
    score: float  # cosine similarity: higher means closer
    preview: str


class RetrieveResponse(BaseModel):
    """Top-k matches, best first, plus what embedding the question cost."""

    matches: list[RetrievedChunk]
    embedding_tokens: int
    cost_usd: float


def text_preview(text: str) -> str:
    """The start of a chunk on one line, so a list of matches reads at a glance."""

    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS].rstrip() + "..."


@app.get("/debug/retrieve")
def debug_retrieve(
    q: Annotated[str, Query(max_length=MAX_QUESTION_CHARS)],
    k: Annotated[int, Query(ge=1, le=10)] = 5,
    document_id: Annotated[str | None, Query(pattern=SLUG_PATTERN, max_length=100)] = None,
) -> RetrieveResponse:
    """Question in, top-k chunks and scores out: the retrieval half of /ask, testable on its own."""

    if not q.strip():
        raise HTTPException(status_code=400, detail="q is empty or whitespace-only")
    matches, embedding_tokens = rag.retrieve(q, top_k=k, document_id=document_id)
    return RetrieveResponse(
        matches=[
            RetrievedChunk(
                chunk_id=match.chunk_id,
                document_id=match.document_id,
                source=match.source,
                score=round(match.score, 4),
                preview=text_preview(match.text),
            )
            for match in matches
        ],
        embedding_tokens=embedding_tokens,
        cost_usd=round(rag.embedding_cost_usd(embedding_tokens), 8),
    )


class HealthResponse(BaseModel):
    """Whether Pinecone answered, and how many vectors this service's namespace holds."""

    status: str  # "ok" or "error"
    index: str
    namespace: str
    vector_count: int | None = None
    error: str | None = None


@app.get("/health", response_model_exclude_none=True)
def health(response: Response) -> HealthResponse:
    """A real Pinecone round trip, so a missing key or index shows up here first, as a 503."""

    try:
        count = rag.namespace_vector_count()
    # Any failure means "not ready": report its kind, never its details.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        response.status_code = 503
        reason = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        return HealthResponse(
            status="error", index=rag.INDEX_NAME, namespace=rag.NAMESPACE, error=reason
        )
    return HealthResponse(
        status="ok", index=rag.INDEX_NAME, namespace=rag.NAMESPACE, vector_count=count
    )
