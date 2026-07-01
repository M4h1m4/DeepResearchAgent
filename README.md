# Deep Research Agent

A dual-mode RAG (Retrieval-Augmented Generation) system built with **FastAPI**, **LangGraph**, and **Pinecone**. It combines a pre-seeded knowledge base, user document uploads, and live web search behind two modes: a low-latency **Fast RAG** path for interactive queries, and an iterative, multi-hop **Deep Research** agent for complex questions. Ships with a web UI, an evaluation framework (RAGAS / BEIR), guardrails (PII + hallucination), and an MCP server.

## Features

- 🔀 **Dual-mode RAG**
  - **Fast RAG** — single-pass retrieve + generate, with sub-100 ms vector retrieval
  - **Deep Research** — LangGraph pipeline that plans sub-queries, iterates over gaps, and synthesizes findings across sources
- 🧠 **Intelligent query routing** — auto-detects temporal ("latest", "2026") and summarization intent to pick the right source and mode (see [Routing](#query-routing))
- 🗂️ **Three retrieval sources** — a pre-seeded Pinecone knowledge base, per-session **document uploads** (PDF / TXT / DOCX), and **live web search** (Tavily)
- 🌱 **Auto-seeded knowledge base** — populates Pinecone from a HuggingFace dataset (SQuAD Wikipedia) on first boot
- 🛡️ **Guardrails** — PII detection/redaction and an LLM-as-judge faithfulness/hallucination check with a confidence score
- 🔗 **Inline citations** — Deep Research answers cite `[N]` markers that map to a linked Sources list
- 📊 **Evaluation framework** — retrieval (Precision@K, Recall@K, MRR, NDCG) and generation (faithfulness, answer relevance) metrics over RAGAS / BEIR datasets
- ⏱️ **Session management** — TTL-based sessions with background cleanup for uploaded documents
- 🔌 **MCP server + client agent** — expose the research tools over the Model Context Protocol
- 💻 **Web UI** — chat interface with mode badges, confidence chips, linked sources, and a research trace
- 📝 **Structured JSON logging** for observability

## Tech stack

| Layer | Technology |
|-------|-----------|
| API / web | FastAPI, Uvicorn, static web UI |
| Vector store | **Pinecone** (serverless, cosine) |
| Relational | SQLAlchemy + SQLite (metadata, sessions, query logs) |
| LLM & embeddings | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) |
| Orchestration | LangGraph (Deep Research state machine) |
| Web search | Tavily |
| Guardrails | LLM-as-judge faithfulness + PII detection |
| Evaluation | RAGAS, BEIR |
| Protocol | MCP (Model Context Protocol) |
| Packaging | uv |

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- [uv](https://github.com/astral-sh/uv) package manager
- API keys: **OpenAI** (required), **Pinecone** (required), **Tavily** (required for web search / Deep Research)

## Installation

```bash
git clone https://github.com/M4h1m4/DeepResearchAgent.git
cd DeepResearchAgent
uv sync
```

Create a `.env` from the template and fill in your keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
TAVILY_API_KEY=tvly-...
ENVIRONMENT=development
```

The Pinecone index is **created automatically** on first startup, and the knowledge base is seeded once (~30 s). No manual index setup required.

## Configuration

All settings live in `config/settings.py` and can be overridden via `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI key (required) | – |
| `PINECONE_API_KEY` | Pinecone key (required) | – |
| `TAVILY_API_KEY` | Tavily key for web search | – |
| `OPENAI_MODEL` | Generation model | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `DATABASE_URL` | SQLAlchemy URL | `sqlite:///./data/rag_metadata.db` |
| `PINECONE_INDEX_NAME` | Index name | `deep-research-kb` |
| `PINECONE_CLOUD` / `PINECONE_REGION` | Serverless location | `aws` / `us-east-1` |
| `SEED_ON_STARTUP` | Seed KB on first boot | `true` |
| `SEED_MAX_PASSAGES` | Passages to seed | `500` |
| `SIMILARITY_THRESHOLD` | Candidate chunk gate (0–1) | `0.3` |
| `WEB_FALLBACK_THRESHOLD` | Below this top KB score → web search | `0.45` |
| `MAX_RESEARCH_ITERATIONS` | Deep Research iteration cap | `3` |
| `SESSION_TTL_HOURS` | Uploaded-document session lifetime | `24` |

## Modes

### Fast RAG
Single-pass retrieve + generate. With no document it answers from the Pinecone knowledge base (or falls back to web search when the KB has no confident match); with a document attached it stays grounded to that document.

### Deep Research
A LangGraph state machine that plans sub-queries, runs retrieval + a web-search gate per sub-query, analyzes knowledge gaps, iterates (up to 3×), and synthesizes a cited final answer.

### Comparison (measured, end-to-end)

| | Fast RAG | Deep Research |
|---|---|---|
| Latency | ~5–20 s (sub-100 ms vector retrieval) | ~100–250 s |
| Strategy | single-pass | multi-hop, iterative |
| Sources | doc *or* KB *or* one web search | doc/KB **+** web, synthesized |
| Best for | direct questions | complex, comparative, or current-info questions |

> **Note:** "sub-100 ms" refers to the Pinecone vector lookup; end-to-end latency is dominated by LLM generation and the faithfulness guardrail.

## Query routing

The `/query` endpoint picks the effective mode and source automatically:

- **Summarization** (with a document) → always **Fast** (reads the whole doc in order).
- **Temporal** query ("latest", "now", "2026", …) with **no document** → **Fast** + forced web search.
- Everything else → runs the **user's selected mode** unchanged. No automatic upgrade to Deep.

Source selection: Fast + document → document only; Fast + no document → KB (web fallback if top score < `WEB_FALLBACK_THRESHOLD`); Deep → retrieval (doc or KB) blended with web.

## API

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a document (PDF/TXT/DOCX), returns a `session_id` |
| `POST` | `/query?mode=fast\|deep` | Query (KB, or a document via `session_id`) |
| `POST` | `/query/deep` | Force Deep Research |
| `DELETE` | `/sessions/{session_id}` | Delete an uploaded-document session |
| `GET` | `/health` | Health check |
| `POST` | `/evals/fast-rag` · `/evals/deep-research` · `/evals/compare` | Run evaluations |

```bash
# Fast query against the knowledge base
curl -X POST "http://localhost:8000/api/v1/query?mode=fast" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is photosynthesis?", "session_id": null}'

# Deep research query
curl -X POST "http://localhost:8000/api/v1/query?mode=deep" \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare the latest adversarial ML defenses"}'
```

Interactive docs: `http://localhost:8000/docs`.

## Evaluation

The framework benchmarks both modes on RAGAS / BEIR datasets:

- **Retrieval:** Precision@K, Recall@K, MRR, NDCG
- **Generation:** faithfulness, answer relevance, context precision/recall

```bash
curl -X POST "http://localhost:8000/api/v1/evals/fast-rag?dataset_name=fiqa"
curl -X POST "http://localhost:8000/api/v1/evals/deep-research?dataset_name=hotpot_qa"
```

## Running locally

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` for the web UI.

## Deployment

### Replit (free tier)
Import the GitHub repo, add `OPENAI_API_KEY`, `PINECONE_API_KEY`, `TAVILY_API_KEY`, and `ENVIRONMENT=production` in the **Secrets** tab, then hit **Run**. The workspace webview URL serves the app (note: free-tier Repls sleep when inactive). A Reserved-VM deployment (`.replit` `[deployment]`) is available on paid plans for an always-on URL.

### Docker
```bash
docker build -t deep-research-agent .
docker run -p 8000:8000 --env-file .env deep-research-agent
```
> The Dockerfile uses `uv sync --frozen`, so `uv.lock` must be committed for Docker-based builds.

## Project structure

```
app/
├── api/routes.py              # FastAPI routes + query routing
├── services/
│   ├── rag_service.py         # Fast RAG (retrieval + web fallback)
│   ├── deep_research_service.py
│   ├── vector_service.py      # Pinecone operations
│   ├── guardrails.py          # PII + hallucination/faithfulness
│   ├── knowledge_seeder.py    # HuggingFace KB seeding
│   └── session_service.py     # TTL session management
├── research/                  # LangGraph: graph, nodes, state, tools
├── evals/                     # metrics, datasets, evaluators, runners, reports
├── mcp/server.py              # MCP server
├── database.py · models.py · schemas.py · main.py
config/                        # settings, logging
mcp_client_agent/              # MCP client agent
static/index.html              # web UI
```
