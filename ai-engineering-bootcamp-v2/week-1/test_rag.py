#!/usr/bin/env python3
"""RAG smoke tests: a local server on a throwaway Pinecone namespace, plus offline /ask checks.

Run: .venv/bin/python test_rag.py
Needs OPENAI_API_KEY and PINECONE_API_KEY in .env. Calls gpt-4o-mini only; a run costs well
under a cent. The namespace is random per run and deleted at the end, whatever happens, so the
real corpus is never touched. The offline tests stub retrieval and the model to reach failures
a real model won't produce on demand; they make no API calls.
"""

import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import main as api  # "main" is this file's own entry point below
import rag  # for cleanup and the offline tests; calls that reach Pinecone name the test namespace
from test_all_stages import WORKDIR, check, free_port, wait_up

NAMESPACE = f"test-rag-{uuid.uuid4().hex[:8]}"
DOC_ID = "plant-policy"
QUESTION = "What is the quarterly budget for office plants?"

# Offline tests: one model call as the API bills it, and the chunk the stubbed retrieval returns.
BILLED = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
RETRIEVED = rag.Match(
    chunk_id="doc#0", document_id="doc", source="doc.txt", score=0.9, text="Some text."
)

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


def model_answer(
    text: str = "A real answer.", citations: list[str] | None = None, refused: bool = False
) -> api.Answer:
    """An Answer as the model might write it; it cites the retrieved chunk unless told otherwise."""

    return api.Answer(
        answer=text,
        confidence=0.9,
        sources_needed=False,
        citations=[RETRIEVED.chunk_id] if citations is None else citations,
        refused=refused,
    )


def ask_offline(*replies: api.Answer | Exception) -> tuple[int, dict, int]:
    """
    POST /ask in-process, with retrieval stubbed and one fake model reply per attempt. A reply is
    an Answer, or an exception that parsing raises; every attempt bills BILLED, as a real call
    would. Returns (HTTP status, JSON body, attempts made).
    """

    pending = iter(replies)

    def fake_parse(**_kwargs) -> SimpleNamespace:
        reply = next(pending)

        def parse() -> SimpleNamespace:
            if isinstance(reply, Exception):
                raise reply
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=reply))])

        return SimpleNamespace(
            http_response=SimpleNamespace(json=lambda: {"usage": BILLED}), parse=parse
        )

    completions = SimpleNamespace(with_raw_response=SimpleNamespace(parse=fake_parse))
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with (
        patch.object(api, "client", fake_client),
        patch.object(rag, "retrieve", return_value=([RETRIEVED], 3)),
    ):
        response = TestClient(api.app).post("/ask", json={"question": "Anything?"})
    return response.status_code, response.json(), len(replies) - len(list(pending))


def test_unpriced_model(base: str) -> bool:
    """Unpriced model: 422 before any tokens are spent"""

    status, data = call("POST", f"{base}/ask", json={"question": QUESTION, "model": "gpt-4"})
    return check("422", status == 422, f"HTTP {status}: {data.get('detail')}")


def test_usage_adds_up_offline(_base: str) -> bool:
    """Usage (offline): tokens and cost cover every attempt, including the one that failed"""

    two_calls = api.compute_cost_usd(api.DEFAULT_MODEL, 200, 20) + rag.embedding_cost_usd(3)
    ok = True
    for label, first_reply in (
        ("parsing failed", ValueError("no parseable output")),
        ("wrong citation", model_answer(citations=["ghost#0"])),
    ):
        status, data, attempts = ask_offline(first_reply, model_answer())
        tokens, cost = data.get("tokens_used"), data.get("cost_usd")
        counted = status == 200 and attempts == 2 and tokens == 220 and cost == round(two_calls, 8)
        ok &= check(label, counted, f"{label}, then OK: {attempts} calls, tokens={tokens}, ${cost}")
    return ok


def test_refusal_format_offline(_base: str) -> bool:
    """Refusal format (offline): the code rewrites any refusal itself, with no retry"""

    exact = {"answer": rag.REFUSAL_TEXT, "citations": [], "refused": True}
    ok = True
    for label, reply in (
        ("refused=true, own wording, one citation", model_answer("No idea.", refused=True)),
        ("refusal text, refused=false, one citation", model_answer(rag.REFUSAL_TEXT)),
    ):
        status, data, attempts = ask_offline(reply)
        answer = data.get("answer", {})
        got = {key: answer.get(key) for key in exact}
        rewritten = status == 200 and attempts == 1 and got == exact
        ok &= check(label, rewritten, f"{label}: HTTP {status}, {attempts} call, {got}")
    return ok


def test_bad_citations_offline(_base: str) -> bool:
    """Citations (offline): a real answer with bad citations gets one retry, then a 502"""

    ok = True
    for label, reply in (
        ("no citation", model_answer(citations=[])),
        ("unretrieved citation", model_answer(citations=["ghost#0"])),
    ):
        status, data, attempts = ask_offline(reply, reply)
        detail = data.get("detail")
        ok &= check(label, status == 502 and attempts == 2, f"HTTP {status}, {attempts}: {detail}")
    return ok


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
    """force_bad: attempt 1 breaks the schema, the retry recovers, and both calls are counted"""

    _, normal = call("POST", f"{base}/ask", json={"question": QUESTION})
    status, data = call("POST", f"{base}/ask", json={"question": QUESTION, "force_bad": True})
    citations = data.get("answer", {}).get("citations", [])
    ok = check("recovers", status == 200, f"HTTP {status} (the retry should recover)")
    ok &= check("cited", bool(citations), f"citations={citations}")
    # Counting only the retry would land near one normal call; two calls land well above it.
    tokens, one_call = data.get("tokens_used", 0), normal.get("tokens_used", 0)
    counted = one_call > 0 and tokens > 1.5 * one_call
    ok &= check("tokens", counted, f"tokens_used={tokens}, one normal call={one_call}")
    cost, one_cost = data.get("cost_usd", 0), normal.get("cost_usd", 0)
    counted = one_cost > 0 and cost > 1.5 * one_cost
    ok &= check("cost", counted, f"cost_usd={cost}, one normal call={one_cost}")
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


def test_delete_document(base: str) -> bool:
    """delete_document.py: removes one document and leaves a similar name alone"""

    sibling = f"{DOC_ID}-v2"
    call("POST", f"{base}/ingest", json={"text": DOC_V2, "document_id": sibling})
    # The script subtracts from Pinecone's count, and counts lag writes: let it settle first.
    settled = wait_until(lambda: call("GET", f"{base}/health")[1].get("vector_count") == 2)
    ok = check("settled", settled, "Pinecone counts 2 vectors: the document and its sibling")

    result = subprocess.run(
        [sys.executable, "delete_document.py", DOC_ID],
        cwd=WORKDIR,
        env={**os.environ, "PINECONE_NAMESPACE": NAMESPACE},
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout.strip().splitlines() or ["(no output)"]
    ok &= check("script", result.returncode == 0, f"exit {result.returncode}: {output[-1]}")

    def listed() -> set[str]:
        pages = rag.get_index().list(prefix=DOC_ID, namespace=NAMESPACE)
        return {item.id for page in pages for item in page.vectors}

    left = {f"{sibling}#0"}
    ok &= check("sibling kept", wait_until(lambda: listed() == left), f"only {left} is left")
    return ok


TESTS = [
    test_unpriced_model,
    test_usage_adds_up_offline,
    test_refusal_format_offline,
    test_bad_citations_offline,
    test_empty_namespace,  # must run before anything is ingested
    test_ingest,
    test_cited_answer,
    test_refusal,
    test_force_bad,
    test_reingest,
    test_delete_document,  # must run after test_reingest, which leaves one chunk of the document
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
