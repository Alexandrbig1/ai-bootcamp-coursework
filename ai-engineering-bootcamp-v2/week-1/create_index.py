#!/usr/bin/env python3
"""Create the Pinecone index once, then print its dimension and vector count.

Run: .venv/bin/python create_index.py   (safe to re-run: an existing index is left as it is)
Counts lag writes by a few seconds, so a fresh ingest may not be counted yet.
"""

import sys

from pinecone import ServerlessSpec

import rag


def main() -> int:
    """Create the index if needed, check its shape, print its counts; exit code 1 on a mismatch."""

    pc = rag.pinecone_client()
    if pc.has_index(rag.INDEX_NAME):
        print(f"Index '{rag.INDEX_NAME}' already exists")
    else:
        # The SDK waits until the new index is ready before returning.
        pc.create_index(
            name=rag.INDEX_NAME,
            vector_type="dense",
            dimension=rag.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # the free plan's only region
        )
        print(f"Created index '{rag.INDEX_NAME}'")

    # An index made for another embedding model would reject every upsert, so check it first.
    description = pc.describe_index(rag.INDEX_NAME)
    if description.dimension != rag.EMBED_DIM or description.metric != "cosine":
        print(
            f"Expected dimension {rag.EMBED_DIM} and cosine; "
            f"found {description.dimension} and {description.metric!r}"
        )
        return 1

    stats = rag.get_index().describe_index_stats()
    namespace = stats.namespaces.get(rag.NAMESPACE)
    count = namespace.vector_count if namespace else 0
    total = stats.total_vector_count
    print(f"dimension: {stats.dimension}, metric: cosine")
    print(f"vectors in namespace '{rag.NAMESPACE}': {count} (whole index: {total})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:  # a missing PINECONE_API_KEY lands here, by name
        sys.exit(f"Error: {exc}")
