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
- Code checks every answer: citations must be retrieved chunk IDs and a real answer needs at least one, with one retry and then a 502. The code writes every refusal itself, in one exact format. When nothing matches at all, `/ask` refuses without calling the model.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | `{text, document_id, source?}` → chunk, embed, upsert. Returns `chunk_ids`, `status` (`created` or `replaced`), `embedding_tokens`, `cost_usd`. |
| `GET /debug/retrieve?q=…&k=5&document_id=…` | Retrieval only, no LLM: the top-k chunks with scores and previews. |
| `POST /ask` | Week 1's request and response, plus optional `top_k` (1–10) and `document_id`. Adds `citations` and `refused` to the answer, and `retrieved` and `embedding_tokens` to the response. `tokens_used` and `cost_usd` cover every model call in the request, including a failed first attempt. |
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
- To remove a document, run `python delete_document.py <document_id>`. It deletes the chunks by ID prefix and prints the vector count. It's a local script on purpose, never an endpoint, so only someone holding the Pinecone key can delete.

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
- **A document gave orders.** The product spec says "say you need Engineering confirmation", and the prompt treats chunk text as data. An early check looked like proof because "How much does a WB-9 cost?" got the exact refusal, but at `top_k=5` that chunk ranks 6th, so the model never saw it. The real test is `top_k=10`, where the chunk is in the context and the bot still refused 3 times out of 3.
- **Pinecone is eventually consistent.** Writes show up after a short delay, so the tests poll after each ingest instead of reading straight back.
- **Failed attempts weren't counted** (found in code review). With `force_bad`, the first call raised before its usage was read, and a retry overwrote the previous attempt's numbers, so `tokens_used` and `cost_usd` understated the spend. Usage now comes from the raw response, read before parsing can fail, and adds up across attempts.
- **A sloppy refusal could end in a 502** (found in code review). The model had to write the refusal character for character, so any variation, such as a curly apostrophe, failed the check and used up the retry. Now the code owns the format: when the model sets `refused=true` or answers with exactly the refusal text, the code writes the refusal itself. Real answers still need valid citations, with one retry and then a 502.

### Golden set

`golden_set.json` holds 8 questions whose expected answers were checked by hand against the documents: 6 answerable, one from each document, and 2 that must refuse. A retrieval hit means the row's evidence phrase appears in a retrieved chunk, so the same set works at any chunk size.

```bash
python eval_golden.py                                               # full run, against the local server
python eval_golden.py --retrieval-only --namespace northwind --k 5  # retrieval only: no server, no LLM
python chunk_experiment.py                                          # 800/100 vs 400/50, retrieval only
```

Runs are saved to `eval_results/`, which is git-ignored.

**Results.** One full run on 2026-11-02 at the default `top_k=5` over the 60 Northwind chunks, with gpt-4o-mini answering and a gpt-4o-mini judge checking each answer against the chunks it cites. Rank is where the evidence phrase first appears in the retrieved chunks.

| Row | Question | Expected answer | Rank | Refusal OK | Correct | Faithful |
|---|---|---|---|---|---|---|
| `remote-days` | How many remote days are allowed? | Up to three days per week | 3 | yes | yes | yes |
| `mileage-rate` | What is the mileage rate? | 45 pence per mile, for travel over 50 miles | 1 | yes | yes | yes |
| `lost-laptop` | How quickly must a lost laptop be reported? | Within one hour | 1 | yes | yes | yes |
| `wb9-payload` | What is the WB-9 payload limit? | 25 kg | 1 | yes | yes | yes |
| `pol220-owner` | Who owns policy POL-220? | Head of IT | 2 | yes | yes | yes |
| `cambridge-hours` | What time does the Cambridge office open? | 07:30 on weekdays | 2 | yes | yes | yes |
| `parental-leave` | What is the parental leave policy? | Must refuse | - | yes | yes | - |
| `wb9-price` | How much does a WB-9 cost? | Must refuse | - | yes | yes | - |

| Score | Result | Covers |
|---|---|---|
| hit@5 | 6/6 | answerable rows |
| hit@1 | 3/6 | answerable rows |
| MRR | 0.72 | answerable rows |
| Refusal accuracy | 8/8 | all rows |
| Correctness | 8/8 | all rows |
| Faithfulness | 6/6 | answerable rows |
| Total cost | $0.0020 | `/ask` $0.0015 + judge $0.0005 |

**Chunk-size comparison.** Retrieval only, top 5, with each size in its own namespace.

| Chunk size / overlap | Chunks | Embedding tokens | hit@1 | hit@5 | MRR |
|---|---|---|---|---|---|
| 800/100 (default) | 60 | 8,901 | 3/6 | 6/6 | 0.72 |
| 400/50 | 142 | 9,276 | 4/6 | 5/6 | 0.75 |

At 400/50, `remote-days` and `cambridge-hours` move up to rank 1, `mileage-rate` drops to rank 2, and `pol220-owner` falls out of the top 5. These numbers don't justify changing the default: `/ask` sends all five chunks to the model, so reaching the top 5 matters more than ranking first, and 400/50 needs 2.4 times the vectors.

**The WB-9 decision.** "How much does a WB-9 cost?" must refuse. The documents state no price. Section 9 of the product spec says "If the answer is not here, say you need Engineering confirmation", but that sentence speaks to Support and Sales about "can it do X?" questions, and the bot treats chunk text as data.

**What each result taught me**

- **Retrieval (hit@5 6/6, hit@1 3/6, MRR 0.72):** The right chunk always reaches the model but ranks first only half the time, which is fine while `/ask` sends all five chunks.
- **Refusals (8/8):** The prompt holds on a missing topic and on a near miss, though two refusal rows can't prove much.
- **Correctness (8/8):** With the right chunk in the context, the model picks the right fact, but in a set this small one flipped row moves a score by at least 12 points.
- **Faithfulness (6/6):** A judge that always says yes would score the same, so it also got three deliberately bad answers, and it rejected all three.
- **Cost ($0.0020):** A full run is cheap enough to repeat after any change to retrieval or the prompt.
- **Chunk size:** Summary scores can hide what matters, since 400/50 wins on MRR while losing the exact-ID row, so compare per question.
- **WB-9 decision:** A refusal proves nothing about an instruction the model never saw, so check what was in the context before crediting the prompt.

## Project layout

```
week-1/
├── main.py              # Full system (stages 1–5 combined), extended with RAG in Week 2
├── rag.py               # Week 2: chunking, embeddings, Pinecone, grounding prompt and checks
├── create_index.py      # Week 2: one-time Pinecone index setup
├── ingest_folder.py     # Week 2: POST every .txt file in a folder to /ingest
├── delete_document.py   # Week 2: local script that removes one document's chunks
├── golden_set.json      # Week 2: 8 questions with expected answers and evidence phrases
├── eval_golden.py       # Week 2: scores /ask against the golden set, with a judge
├── chunk_experiment.py  # Week 2: compares chunk sizes on retrieval only
├── rag_page.py          # Week 2: Streamlit Ingest / Retrieve / Ask page
├── test_rag.py          # Week 2: RAG smoke tests on a throwaway namespace
├── serve_stage1.py … serve_stage5.py
├── demo_page.py         # Streamlit test UI
├── test_all_stages.py   # Automated stage smoke tests
├── requirements.txt
├── .env.example
└── .gitignore
```
