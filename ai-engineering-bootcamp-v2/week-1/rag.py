"""Week 2 RAG layer: chunking, embeddings and the Pinecone index behind the API in main.py."""

import os
import re
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Index, Pinecone

# main.py imports this module before its own load_dotenv call runs, so load .env here as well.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Locked early: switching the embedding model means re-embedding every document.
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
EMBED_PRICE_PER_1M_USD = 0.02

INDEX_NAME = os.getenv("PINECONE_INDEX") or "coursework-rag"
NAMESPACE = os.getenv("PINECONE_NAMESPACE") or "northwind"

CHUNK_SIZE = 800  # characters, overlap included
CHUNK_OVERLAP = 100

# Pinecone caps one upsert at 1,000 records / 2 MB, and a 1536-float vector is ~20 KB on the wire.
UPSERT_BATCH = 50
# Pinecone caps one delete at 1,000 IDs.
DELETE_BATCH = 1000

_SEPARATOR = "\n\n"
# Decorative lines such as ===== carry no meaning but use up chunk space and blur the vectors.
_RULE_LINE = re.compile(r"^\s*([=\-_*~#])\1{2,}\s*$")
# Fallbacks for a paragraph too long for one chunk, tried in order, each with the text that rejoins pieces.
_SPLITTERS = (
    (re.compile(r"(?<=[.!?])\s+"), " "),  # sentence ends
    (re.compile(r"\n"), "\n"),  # line breaks
    (re.compile(r"\s+"), " "),  # any whitespace
)


@cache
def _openai() -> OpenAI:
    """One embeddings client per process; main.py keeps its own client for chat."""

    return OpenAI()


@cache
def pinecone_client() -> Pinecone:
    """Built on first use, so the app still starts without the key and can name what's missing."""

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    return Pinecone(api_key=api_key)


@cache
def get_index() -> Index:
    """Handle for data operations; resolving the index host costs a lookup, so do it once."""

    return pinecone_client().Index(name=INDEX_NAME)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks of at most `size` characters, keeping paragraphs whole where possible.

    Paragraphs (separated by blank lines) are packed greedily. Each chunk after the first
    starts with up to `overlap` characters from the end of the previous one, cut at a word
    boundary, so text on a seam appears in both chunks.
    """

    longest_piece = size - overlap - len(_SEPARATOR)  # leaves room for a full overlap
    if overlap < 0 or longest_piece < 1:
        raise ValueError("overlap must be >= 0 and leave room for text within size")

    lines = ["" if _RULE_LINE.match(line) else line.strip() for line in text.splitlines()]
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", "\n".join(lines)) if p.strip()]
    pieces = [piece for paragraph in paragraphs for piece in _split(paragraph, longest_piece)]

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(_SEPARATOR) + len(piece) <= size:
            current += _SEPARATOR + piece
            continue
        if current:
            chunks.append(current)
        tail = _tail(current, overlap)
        current = f"{tail}{_SEPARATOR}{piece}" if tail else piece
    if current:
        chunks.append(current)
    return chunks


def _split(text: str, max_len: int, level: int = 0) -> list[str]:
    """Break text longer than max_len at sentence ends, then line breaks, then spaces, then anywhere."""

    if len(text) <= max_len:
        return [text]
    if level == len(_SPLITTERS):
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]

    pattern, joiner = _SPLITTERS[level]
    pieces: list[str] = []
    current = ""
    for part in pattern.split(text):
        for small in _split(part, max_len, level + 1):
            if current and len(current) + len(joiner) + len(small) <= max_len:
                current += joiner + small
            else:
                if current:
                    pieces.append(current)
                current = small
    if current:
        pieces.append(current)
    return pieces


def _tail(chunk: str, overlap: int) -> str:
    """The last `overlap` characters of a chunk, starting at a word boundary."""

    if not chunk or overlap == 0:
        return ""
    tail = chunk[-overlap:]
    if len(chunk) > overlap and not chunk[-overlap - 1].isspace():
        space = re.search(r"\s", tail)
        tail = tail[space.end() :] if space else ""  # drop the word the cut landed in
    return tail.strip()


def embed(texts: list[str]) -> tuple[list[list[float]], int]:
    """Embed texts with the locked model; returns one vector per text, in order, plus billed tokens."""

    # One request is enough: the 100k-character ingest cap keeps us far below OpenAI's 2,048-input limit.
    response = _openai().embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
    return vectors, response.usage.total_tokens


def embedding_cost_usd(tokens: int) -> float:
    """Embedding spend in dollars; OpenAI prices embeddings per 1M tokens."""

    return tokens / 1_000_000 * EMBED_PRICE_PER_1M_USD


def upsert_document(document_id: str, chunks: list[str], source: str) -> tuple[list[str], bool, int]:
    """
    Embed and upsert one document's chunks as "{document_id}#{i}", then delete any chunks
    left over from a longer earlier version. Returns (chunk_ids, replaced, embedding_tokens).
    """

    vectors, embedding_tokens = embed(chunks)
    chunk_ids = [f"{document_id}#{i}" for i in range(len(chunks))]

    index = get_index()
    # The "#" matters: slugs can't contain it, so the prefix "handbook#" never matches "handbook-v2#0".
    pages = index.list(prefix=f"{document_id}#", namespace=NAMESPACE)
    existing = {item.id for page in pages for item in page.vectors}

    index.upsert(
        vectors=[
            {
                "id": chunk_id,
                "values": values,
                "metadata": {"document_id": document_id, "chunk_index": i, "source": source, "text": chunk},
            }
            for i, (chunk_id, values, chunk) in enumerate(zip(chunk_ids, vectors, chunks))
        ],
        namespace=NAMESPACE,
        batch_size=UPSERT_BATCH,
        show_progress=False,
    )

    # Upsert first, prune second, so the document never disappears from the index mid-update.
    stale = sorted(existing - set(chunk_ids))
    for start in range(0, len(stale), DELETE_BATCH):
        index.delete(ids=stale[start : start + DELETE_BATCH], namespace=NAMESPACE)
    return chunk_ids, bool(existing), embedding_tokens
