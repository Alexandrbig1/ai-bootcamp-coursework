#!/usr/bin/env python3
"""POST every .txt file in a folder to /ingest, one document per file.

Run: .venv/bin/python ingest_folder.py data/northwind
The target is API_BASE_URL from .env (default http://127.0.0.1:8000); a value set in the shell
wins over .env. Each file's name without .txt is its document_id, and its name is the source.
"""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
BASE_URL = (os.getenv("API_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")


def target_label(url: str) -> str:
    """Say 'local' or 'live' so the live URL never ends up in terminal logs or screenshots."""

    return "local" if urlparse(url).hostname in {"127.0.0.1", "localhost"} else "live"


def main() -> int:
    """Ingest each file in turn; exit code 1 if any file failed."""

    if len(sys.argv) != 2:
        print("Usage: python ingest_folder.py <folder>")
        return 2
    files = sorted(Path(sys.argv[1]).glob("*.txt"))
    if not files:
        print(f"No .txt files in {sys.argv[1]}")
        return 1

    print(f"Ingesting {len(files)} files into the {target_label(BASE_URL)} API")
    total_chunks = total_tokens = failed = 0
    total_cost = 0.0
    for path in files:
        payload = {
            "text": path.read_text(encoding="utf-8"),
            "document_id": path.stem,
            "source": path.name,
        }
        try:
            # Generous timeout: a sleeping Render free instance can take a minute to wake up.
            response = httpx.post(f"{BASE_URL}/ingest", json=payload, timeout=180.0)
        except httpx.HTTPError as exc:
            print(f"  {path.name}: request failed ({type(exc).__name__})")
            failed += 1
            continue
        if response.status_code != 200:
            print(f"  {path.name}: HTTP {response.status_code} {response.text[:200]}")
            failed += 1
            continue
        data = response.json()
        total_chunks += data["chunks_indexed"]
        total_tokens += data["embedding_tokens"]
        total_cost += data["cost_usd"]
        print(
            f"  {path.name:28} {data['chunks_indexed']:3} chunks  {data['status']:8}"
            f"  {data['embedding_tokens']:5} tokens"
        )

    print(
        f"Total: {total_chunks} vectors from {len(files) - failed} of {len(files)} files, "
        f"{total_tokens} embedding tokens, ${total_cost:.6f}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
