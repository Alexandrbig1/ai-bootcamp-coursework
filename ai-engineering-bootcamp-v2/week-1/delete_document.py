#!/usr/bin/env python3
"""Delete one document's chunks from Pinecone, then print the namespace's vector count.

Run: .venv/bin/python delete_document.py parking-demo
A local script on purpose, never an endpoint: only someone holding the Pinecone key can delete.
It acts on PINECONE_NAMESPACE (default northwind) and finds chunks by the ID prefix
"<document_id>#", so a document with a similar name is left alone.
"""

import sys
import time

import rag

SETTLE_TIMEOUT = 60.0  # seconds; Pinecone shows deletes after a short delay


def main() -> int:
    """Delete the chunks and wait until Pinecone agrees; exit code 1 if it never does."""

    if len(sys.argv) != 2:
        print("Usage: python delete_document.py <document_id>")
        return 2
    document_id = sys.argv[1]

    chunk_ids = rag.document_chunk_ids(document_id)
    before = rag.namespace_vector_count()
    print(f"Namespace '{rag.NAMESPACE}' holds {before} vectors")
    if not chunk_ids:
        print(f"No chunks found for '{document_id}'; nothing deleted")
        return 0

    print(f"Deleting {len(chunk_ids)} chunks of '{document_id}': {sorted(chunk_ids)}")
    rag.delete_chunks(chunk_ids)

    expected = before - len(chunk_ids)
    deadline = time.monotonic() + SETTLE_TIMEOUT
    while rag.document_chunk_ids(document_id) or rag.namespace_vector_count() != expected:
        if time.monotonic() > deadline:
            print(
                f"Not settled after {SETTLE_TIMEOUT:.0f}s: expected {expected} vectors, "
                f"Pinecone reports {rag.namespace_vector_count()}. Run create_index.py to check."
            )
            return 1
        time.sleep(2)

    print(f"Vectors in namespace '{rag.NAMESPACE}': {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
