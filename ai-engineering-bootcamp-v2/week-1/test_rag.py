#!/usr/bin/env python3
"""RAG smoke tests against a local server that uses a throwaway Pinecone namespace.

Run: .venv/bin/python test_rag.py
Needs OPENAI_API_KEY and PINECONE_API_KEY in .env. Calls gpt-4o-mini only; a run costs well
under a cent. The namespace is random per run and deleted at the end, whatever happens, so the
real corpus is never touched.
"""

import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable

import httpx

import rag  # only for cleanup; every call below names the test namespace explicitly
from test_all_stages import WORKDIR, check, free_port, wait_up

NAMESPACE = f"test-rag-{uuid.uuid4().hex[:8]}"
DOC_ID = "plant-policy"
QUESTION = "What is the quarterly budget for office plants?"

# Facts no model knows from training, in three paragraphs too long to share a chunk.
DOC_V1 = """Zephyr Labs Office Plant Policy

Budget. The office plant budget is 75 euros per quarter for each team. Purchases above that
amount need written approval from the Facilities lead before anything is ordered. Receipts go
to the shared Facilities inbox within five working days, and unused budget does not roll over
into the next quarter. Teams may pool their budgets for a larger shared plant, as long as every
contributing team lead agrees in writing.

Watering. Plants are watered every Tuesday and Friday by the volunteer on the weekly rota. The
rota is posted on the kitchen whiteboard every Monday morning, and swaps are agreed directly
between volunteers. During office closures, the Facilities lead arranges cover. Overwatering
kills more plants than neglect, so volunteers follow the care card attached to each pot.

Safety. Cacti and other spiny plants must be kept at least two metres away from server racks
and walkways. Spilled soil is reported to Facilities on the same day so it can be cleaned up
before it reaches any equipment. Plants that cause allergies are moved on request, no
discussion needed, and plants that are poisonous to pets stay out of the office dog area."""

DOC_V2 = """Zephyr Labs Office Plant Policy (v2)

Budget. The office plant budget is 90 euros per quarter for each team."""


def start_server(port: int) -> subprocess.Popen:
    """main:app with PINECONE_NAMESPACE set to the test namespace; .env never overrides it."""

    return subprocess.Popen(
        [
            str(WORKDIR / ".venv/bin/uvicorn"),
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=WORKDIR,
        env={**os.environ, "PINECONE_NAMESPACE": NAMESPACE},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_until(condition: Callable[[], bool], timeout: float = 60.0) -> bool:
    """Pinecone is eventually consistent: poll until a write shows up, or give up."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(2)
    return False


def call(method: str, url: str, **kwargs) -> tuple[int, dict]:
    """One HTTP call; returns (status, JSON body), or the raw text when there is no JSON."""

    response = httpx.request(method, url, timeout=120.0, **kwargs)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"raw": response.text}


def indexed_ids(base: str) -> set[str]:
    """Chunk IDs that retrieval returns for the test document right now."""

    params = {"q": "office plant policy", "k": 10, "document_id": DOC_ID}
    _, data = call("GET", f"{base}/debug/retrieve", params=params)
    return {match["chunk_id"] for match in data.get("matches", [])}


def test_unpriced_model(base: str) -> bool:
    """Unpriced model: 422 before any tokens are spent"""

    status, data = call("POST", f"{base}/ask", json={"question": QUESTION, "model": "gpt-4"})
    return check("422", status == 422, f"HTTP {status}: {data.get('detail')}")


def test_empty_namespace(base: str) -> bool:
    """Empty namespace: refuse without calling the model"""

    status, data = call("POST", f"{base}/ask", json={"question": QUESTION})
    answer = data.get("answer", {})
    ok = check("status", status == 200, f"HTTP {status}")
    ok &= check("refused", answer.get("refused") is True, f"answer={answer.get('answer')!r}")
    tokens = data.get("tokens_used")
    ok &= check("no model call", tokens == 0, f"tokens_used={tokens}")
    return ok


def test_ingest(base: str) -> bool:
    """Ingest: a 3-chunk document becomes searchable"""

    status, data = call("POST", f"{base}/ingest", json={"text": DOC_V1, "document_id": DOC_ID})
    ids = data.get("chunk_ids", [])
    created = status == 200 and data.get("status") == "created" and len(ids) > 1
    ok = check("created", created, f"HTTP {status}, status={data.get('status')}, chunk_ids={ids}")
    searchable = created and wait_until(lambda: indexed_ids(base) == set(ids))
    ok &= check("searchable", searchable, "retrieval returns exactly those chunks")
    return ok


def test_cited_answer(base: str) -> bool:
    """Cited answer: the fact from the document, with retrieved chunk IDs"""

    status, data = call("POST", f"{base}/ask", json={"question": QUESTION})
    answer = data.get("answer", {})
    text, citations = answer.get("answer", ""), answer.get("citations", [])
    retrieved = {item["chunk_id"] for item in data.get("retrieved", [])}
    ok = check("status", status == 200, f"HTTP {status}")
    ok &= check("grounded", answer.get("refused") is False and "75" in text, f"answer={text!r}")
    cited = bool(citations) and set(citations) <= retrieved
    ok &= check("citations", cited, f"citations={citations}")
    cost = data.get("cost_usd", 0)
    ok &= check("cost", cost > 0 and data.get("embedding_tokens", 0) > 0, f"cost_usd={cost}")
    return ok


def test_refusal(base: str) -> bool:
    """Refusal: the answer isn't in the document"""

    question = "What is the parental leave policy?"
    status, data = call("POST", f"{base}/ask", json={"question": question})
    answer = data.get("answer", {})
    text, citations = answer.get("answer"), answer.get("citations")
    ok = check("status", status == 200, f"HTTP {status}")
    refused = answer.get("refused") is True and citations == []
    ok &= check("refused", refused, f"refused={answer.get('refused')}, citations={citations}")
    ok &= check("exact text", text == rag.REFUSAL_TEXT, f"answer={text!r}")
    return ok


def test_force_bad(base: str) -> bool:
    """force_bad: attempt 1 breaks the schema, the retry recovers"""

    status, data = call("POST", f"{base}/ask", json={"question": QUESTION, "force_bad": True})
    citations = data.get("answer", {}).get("citations", [])
    ok = check("recovers", status == 200, f"HTTP {status} (the retry should recover)")
    ok &= check("cited", bool(citations), f"citations={citations}")
    return ok


def test_reingest(base: str) -> bool:
    """Re-ingest: a 1-chunk version replaces the 3-chunk one, with no leftovers"""

    status, data = call("POST", f"{base}/ingest", json={"text": DOC_V2, "document_id": DOC_ID})
    ids = data.get("chunk_ids", [])
    replaced = data.get("status") == "replaced" and ids == [f"{DOC_ID}#0"]
    ok = check("replaced", replaced, f"HTTP {status}, status={data.get('status')}, chunk_ids={ids}")
    clean = replaced and wait_until(lambda: indexed_ids(base) == set(ids))
    ok &= check("no leftovers", clean, "retrieval returns only the new chunk")
    return ok


TESTS = [
    test_unpriced_model,
    test_empty_namespace,  # must run before anything is ingested
    test_ingest,
    test_cited_answer,
    test_refusal,
    test_force_bad,
    test_reingest,
]


def delete_test_namespace() -> None:
    """Delete the test namespace and everything in it."""

    if not NAMESPACE.startswith("test-rag-"):  # belt and braces: never the real corpus
        raise RuntimeError(f"refusing to delete namespace {NAMESPACE!r}")
    try:
        rag.get_index().delete_namespace(name=NAMESPACE)
        print(f"\nDeleted test namespace {NAMESPACE}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"\nCould not delete test namespace {NAMESPACE}: {type(exc).__name__}")


def main() -> int:
    """Start the server, run the tests in order, and always clean up."""

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    print(f"Test namespace: {NAMESPACE}")
    proc = start_server(port)
    results: list[tuple[str, bool]] = []
    try:
        if not wait_up(base):
            print(f"FAIL: server did not start on {base}")
            return 1
        for test in TESTS:
            print(f"\n=== {test.__doc__} ===")
            results.append((test.__name__, test(base)))
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        delete_test_namespace()

    print("\n" + "=" * 40 + "\nSUMMARY")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
