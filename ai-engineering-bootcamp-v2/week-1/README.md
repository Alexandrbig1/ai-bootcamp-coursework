# Week 1 — `/ask` Demo (5 stages)

Build a typed LLM endpoint step by step. Each stage is a standalone FastAPI app you can run and compare.

## Setup

```bash
cp .env.example .env          # OPENAI_API_KEY=sk-...
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo stages

| Stage | File | What you learn |
|-------|------|----------------|
| 1 | `serve_stage1.py` | Bare `/ask` — string answer + `tokens_used` |
| 2 | `serve_stage2.py` | Structured output via Pydantic + `completions.parse` |
| 3 | `serve_stage3.py` | Validation guardrail + retry (`force_bad` demo knob) |
| 4 | `serve_stage4.py` | Per-request `model` override + `latency_ms` |
| 5 | `serve_stage5.py` / `main.py` | Full system + `cost_usd` readout |

Run one stage at a time (only one server on port 8000):

```bash
uvicorn serve_stage1:app --host 127.0.0.1 --port 8000 --reload
# or the full system:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Streamlit demo runner

Interactive UI for all five stages:

```bash
streamlit run demo_page.py
```

Open http://localhost:8501. Set **API base URL** to `http://127.0.0.1:8000` and start the matching stage server in another terminal.

## Test with curl

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG in one sentence?"}'
```

Stage 5 example (model + cost):

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking?", "model": "gpt-4o-mini"}'
```

## Smoke-test all stages

Requires `.venv` and a valid `OPENAI_API_KEY`:

```bash
python test_all_stages.py
```

## Week 2 — RAG over your own documents

`/ask` now answers only from documents ingested into Pinecone: it cites the chunks it used and refuses when they don't hold the answer. Same service, same entry point (`uvicorn main:app`); the RAG code lives in `rag.py`.

### Architecture

```text
POST /ingest  text → chunks (800 chars, 100 overlap, paragraph-aware)
                   → embeddings (text-embedding-3-small, 1536 dims)
                   → Pinecone upsert as "{document_id}#{n}"; re-ingesting replaces the old chunks

POST /ask     question → embedding → Pinecone top-k → grounding prompt with chunk IDs
                       → gpt-4o-mini structured output → citation checks (1 retry, then 502)
                       → answer + citations, or exactly "I don't have enough information to answer that."
```

- Pinecone serverless index `coursework-rag` (cosine, aws us-east-1), namespace `northwind`. Each chunk stores `document_id`, `chunk_index`, `source` and `text`.
- Code checks every answer: citations must be retrieved chunk IDs, an answer needs at least one, and a refusal has none. When nothing matches at all, `/ask` refuses without calling the model.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | `{text, document_id, source?}` → chunk, embed, upsert. Returns `chunk_ids`, `status` (`created` or `replaced`), `embedding_tokens`, `cost_usd`. |
| `GET /debug/retrieve?q=…&k=5&document_id=…` | Retrieval only, no LLM: the top-k chunks with scores and previews. |
| `POST /ask` | Week 1's request and response, plus optional `top_k` (1–10) and `document_id`. Adds `citations` and `refused` to the answer, and `retrieved` and `embedding_tokens` to the response. |
| `GET /health` | Index, namespace and vector count; a 503 with the reason when Pinecone isn't usable. |

### Run it

`data/northwind/` holds the course's six Northwind sample docs. The folder is git-ignored, so copy them in from the course materials first.

```bash
# .env needs OPENAI_API_KEY and PINECONE_API_KEY
# optional: PINECONE_INDEX (default coursework-rag), PINECONE_NAMESPACE (default northwind)
pip install -r requirements.txt
python create_index.py                    # once; safe to re-run
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
python ingest_folder.py data/northwind    # in a second terminal
streamlit run rag_page.py                 # Ingest / Retrieve / Ask page
python test_rag.py                        # RAG tests on a throwaway namespace
python test_all_stages.py                 # Week 1 tests; Stage 5 now goes through RAG
```

`ingest_folder.py` and `rag_page.py` call the API at `API_BASE_URL` from `.env` (default `http://127.0.0.1:8000`). Point it at the deployed service to use them live; the page only ever says "local" or "live".

### Ingesting new documents

- A folder of `.txt` files: `python ingest_folder.py path/to/folder`. Each file becomes one document, and its name without `.txt` is the `document_id` (lowercase letters, digits, `-` and `_`).
- One document: the **Ingest** tab in `rag_page.py`, or `POST /ingest`.
- Sending a `document_id` again replaces that document, with no duplicates and no leftover chunks.

### curl (replace YOUR-SERVICE)

```bash
curl -s -X POST https://YOUR-SERVICE.onrender.com/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Parking. Staff park free at the Cambridge office. Bays 1 to 10 are for visitors.", "document_id": "parking-demo", "source": "curl"}'

curl -s -G https://YOUR-SERVICE.onrender.com/debug/retrieve \
  --data-urlencode "q=How many days a week can employees work remotely?" -d k=5

# Cited answer
curl -s -X POST https://YOUR-SERVICE.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days a week can employees work remotely?"}'

# Refusal: parental leave isn't in the documents
curl -s -X POST https://YOUR-SERVICE.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the parental leave policy?"}'

curl -s https://YOUR-SERVICE.onrender.com/health
```

### What broke along the way

- **Banner lines ate the chunks.** The `=====` lines under each heading were 15–33% of every file. The chunker now drops lines made of one repeated symbol, which cut the corpus from 73 chunks to 60.
- **Week 1's Stage 5 test failed on the new fields.** It compared the exact set of response keys, so `retrieved` and `embedding_tokens` broke it. It now checks that every Week 1 field is still present.
- **Costs rounded to zero.** A 10-token question embedding costs $0.0000002, which Week 1's 6-decimal rounding shows as 0, so `cost_usd` now has 8 decimals.
- **Right document, wrong chunk.** "How many remote days are allowed?" ranks the chunk about international remote work first and the answer chunk third. That's why `top_k` defaults to 5.
- **A score cutoff can't spot a missing answer.** The parental-leave question scores 0.53, higher than the correct mileage chunk at 0.35, so refusals come from the grounding prompt and the code checks instead.
- **A document gave orders.** The product spec says "say you need Engineering confirmation". The prompt treats chunk text as data, so "How much does a WB-9 cost?" still gets the exact refusal.
- **Pinecone is eventually consistent.** Writes show up after a short delay, so the tests poll after each ingest instead of reading straight back.

## Project layout

```
week-1/
├── main.py              # Full system (stages 1–5 combined), extended with RAG in Week 2
├── rag.py               # Week 2: chunking, embeddings, Pinecone, grounding prompt and checks
├── create_index.py      # Week 2: one-time Pinecone index setup
├── ingest_folder.py     # Week 2: POST every .txt file in a folder to /ingest
├── rag_page.py          # Week 2: Streamlit Ingest / Retrieve / Ask page
├── test_rag.py          # Week 2: RAG smoke tests on a throwaway namespace
├── serve_stage1.py … serve_stage5.py
├── demo_page.py         # Streamlit test UI
├── test_all_stages.py   # Automated stage smoke tests
├── requirements.txt
├── .env.example
└── .gitignore
```
