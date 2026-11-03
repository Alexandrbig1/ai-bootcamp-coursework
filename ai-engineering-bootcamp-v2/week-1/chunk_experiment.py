#!/usr/bin/env python3
"""Compare chunk sizes on retrieval only: same corpus, same golden set, one namespace per size.

Run: .venv/bin/python chunk_experiment.py
It embeds the corpus twice (well under a tenth of a cent) and makes no LLM calls. Both namespaces
are deleted at the end, whatever happens, and the real corpus is never read or written.
"""

import json
import sys
import time
from pathlib import Path

from pinecone import PineconeError

import rag
from eval_golden import GOLDEN_SET, WORKDIR, fraction, markdown_table, retrieval_only, save_run

CORPUS_DIR = Path(WORKDIR) / "data" / "northwind"
SETTINGS = (("exp-800", 800, 100), ("exp-400", 400, 50))  # (namespace, chunk size, overlap)
K = 5
SETTLE_TIMEOUT = 120.0  # seconds; Pinecone's counts lag writes


def ingest(namespace: str, size: int, overlap: int) -> tuple[int, int]:
    """Chunk, embed and upsert the corpus into one namespace; returns (chunks, embedding tokens)."""

    total_chunks = total_tokens = 0
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        chunks = rag.chunk_text(path.read_text(encoding="utf-8"), size=size, overlap=overlap)
        ids, _, tokens = rag.upsert_document(path.stem, chunks, path.name, namespace=namespace)
        total_chunks += len(ids)
        total_tokens += tokens
    return total_chunks, total_tokens


def wait_for_count(namespace: str, expected: int) -> bool:
    """Poll until Pinecone counts the expected vectors in a namespace, or give up."""

    deadline = time.monotonic() + SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        if rag.namespace_vector_count(namespace) == expected:
            return True
        time.sleep(2)
    return False


def delete_namespaces() -> None:
    """Delete both experiment namespaces and confirm Pinecone no longer counts them."""

    for namespace, _, _ in SETTINGS:
        if not namespace.startswith("exp-"):  # belt and braces: never the real corpus
            raise RuntimeError(f"refusing to delete namespace {namespace!r}")
        try:
            rag.get_index().delete_namespace(name=namespace)
        except PineconeError as exc:  # e.g. the run stopped before writing to it
            print(f"Could not delete namespace '{namespace}': {type(exc).__name__}")
            continue
        gone = wait_for_count(namespace, 0)
        print(f"Deleted namespace '{namespace}'" if gone else f"'{namespace}' still has vectors")


def main() -> int:
    """Run both settings, print the comparison, and always clean up."""

    rows = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    results = []
    try:
        for namespace, size, overlap in SETTINGS:
            print(f"Embedding the corpus at {size}/{overlap} into '{namespace}'...")
            chunks, tokens = ingest(namespace, size, overlap)
            if not wait_for_count(namespace, chunks):
                print(f"Pinecone didn't count {chunks} vectors in '{namespace}' in time; stopping.")
                return 1
            run = retrieval_only(rows, namespace, K)
            results.append(
                {"size": size, "overlap": overlap, "chunks": chunks, "tokens": tokens, **run}
            )

        print()
        print(
            markdown_table(
                ["chunk size / overlap", "chunks", "embedding tokens", "hit@1", "hit@5", "MRR"],
                [
                    [
                        f"{result['size']}/{result['overlap']}",
                        result["chunks"],
                        result["tokens"],
                        fraction(result["summary"]["hit@1"], result["summary"]["rows"]),
                        fraction(result["summary"]["hit@5"], result["summary"]["rows"]),
                        f"{result['summary']['mrr']:.4f}",
                    ]
                    for result in results
                ],
            )
        )
        total_tokens = sum(result["tokens"] for result in results)
        total_cost = rag.embedding_cost_usd(total_tokens)
        print(f"\nIngest embeddings: {total_tokens} tokens, ${total_cost:.6f}")
        path = save_run("chunk-experiment", {"k": K, "settings": results})
        print(f"Saved {path.relative_to(WORKDIR)}\n")
    finally:
        delete_namespaces()
    return 0


if __name__ == "__main__":
    sys.exit(main())
