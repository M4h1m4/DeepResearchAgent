# Deep Research Agent

A dual-mode RAG (Retrieval-Augmented Generation) system built with FastAPI, SQLAlchemy, and ChromaDB. This application enables document ingestion, vector-based semantic search, and intelligent question-answering using OpenAI's language models and embeddings. The system features two distinct modes optimized for different use cases: **Fast RAG** for real-time queries and **Deep Research** for comprehensive multi-step knowledge discovery.

## Features

- 📄 **Document Processing**: Support for PDF, TXT, and DOCX files
- 🔍 **Semantic Search**: Vector-based retrieval using ChromaDB with OpenAI embeddings
- 🤖 **Dual-Mode RAG System**: 
  - **Fast RAG Mode**: Real-time, single-step retrieval and generation
  - **Deep Research Mode**: Multi-step iterative research with LangGraph orchestration
- 📊 **Evaluation Framework**: Comprehensive performance evaluation using RAGAS/BEIR benchmarks
- 🛡️ **Guardrails**: Input/output validation, PII detection, hallucination prevention
- 💾 **Metadata Management**: SQLAlchemy-based metadata storage (SQLite)
- 🔗 **RESTful API**: FastAPI endpoints for document management and querying
- 📊 **JSON Logging**: Structured logging with JSON format for better observability
- 🚀 **Production Ready**: Clean architecture with separation of concerns

## Tech Stack

- **Framework**: FastAPI
- **Database**: SQLAlchemy (SQLite) for metadata, ChromaDB for vector storage
- **LLM & Embeddings**: OpenAI (GPT-4 Turbo, text-embedding-3-small)
- **Orchestration**: LangGraph for Deep Research mode (state machine workflows)
- **Document Processing**: PyPDF, python-docx
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter
- **Evaluation**: RAGAS, BEIR benchmarks for performance metrics
- **Guardrails**: spaCy, Presidio for PII detection and safety checks
- **Package Management**: uv
- **Logging**: python-json-logger

## Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DeepResearchAgent
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

3. **Create a `.env` file** in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4-turbo-preview
   EMBEDDING_MODEL=text-embedding-3-small
   DATABASE_URL=sqlite:///./data/rag_metadata.db
   CHROMA_PERSIST_DIRECTORY=./data/chroma_db
   ENVIRONMENT=development
   ```

## Configuration

The application uses environment variables for configuration. All settings are defined in `config/settings.py` and can be overridden via `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `OPENAI_MODEL` | LLM model for generation | `gpt-4-turbo-preview` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `DATABASE_URL` | SQLAlchemy database URL | `sqlite:///./data/rag_metadata.db` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB persistence directory | `./data/chroma_db` |
| `ENVIRONMENT` | Environment (development/production) | `development` |
| `CHUNK_SIZE` | Text chunk size for splitting | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RETRIEVAL` | Number of chunks to retrieve | `5` |
| `SIMILARITY_THRESHOLD` | Minimum similarity threshold (0-1) | `0.3` |

## System Modes

### Fast RAG Mode

**Purpose**: Optimized for real-time, single-step question answering with sub-second response times.

**How it works**:
1. **Query Embedding**: User query is embedded using OpenAI embeddings
2. **Vector Search**: ChromaDB performs similarity search to retrieve top-K relevant chunks
3. **Answer Generation**: GPT-4 generates answer using retrieved context in a single pass
4. **Response**: Returns answer with sources and retrieved chunks

**Best for**:
- ✅ Simple, direct questions that can be answered from a single document
- ✅ Real-time applications requiring fast responses (< 2 seconds)
- ✅ Straightforward information retrieval tasks
- ✅ Single-hop questions (e.g., "What is machine learning?")

**Example Use Case**: Customer support chatbot answering FAQ questions from knowledge base

---

### Deep Research Mode

**Purpose**: Comprehensive multi-step research for complex questions requiring information synthesis across multiple sources.

**How it works**:
1. **Research Planning**: Analyzes query and creates a research plan with sub-queries
2. **Iterative Research**: Executes multiple RAG queries using LangGraph state machine
3. **Gap Analysis**: Identifies information gaps and generates follow-up queries
4. **Knowledge Synthesis**: Synthesizes findings from multiple sources into comprehensive answer
5. **Final Answer**: Returns detailed answer with all findings and sources

**Best for**:
- ✅ Complex questions requiring information from multiple documents
- ✅ Multi-hop reasoning (e.g., "Which city did the author of 'The Great Gatsby' die in?")
- ✅ Research tasks requiring comprehensive analysis
- ✅ Questions needing synthesis of information across domains
- ✅ When accuracy and completeness are prioritized over speed

**Example Use Case**: Research assistant analyzing a complex topic across multiple research papers

---

### Mode Comparison

| Feature | Fast RAG Mode | Deep Research Mode |
|---------|--------------|-------------------|
| **Response Time** | < 2 seconds | 5-15 seconds |
| **Query Complexity** | Single-hop | Multi-hop |
| **Retrieval Strategy** | Single vector search | Iterative searches with sub-queries |
| **Answer Quality** | Good for simple questions | Excellent for complex questions |
| **Use Case** | Real-time Q&A | Comprehensive research |
| **Orchestration** | Direct RAG pipeline | LangGraph state machine |

## Performance Evaluation

The system includes a comprehensive evaluation framework that benchmarks both modes against standardized metrics using RAGAS and BEIR benchmark datasets.

### Evaluation Metrics

#### 1. **Retrieval Metrics**
Measures how well the system finds relevant documents:
- **Precision@K**: Of the top K retrieved chunks, how many are actually relevant?
- **Recall@K**: Of all relevant chunks, how many did we successfully retrieve?
- **MRR (Mean Reciprocal Rank)**: Position of the first relevant result (higher = better)
- **NDCG (Normalized Discounted Cumulative Gain)**: Quality of ranking considering position

#### 2. **Generation Metrics**
Measures the quality of generated answers using RAGAS:
- **Faithfulness**: Is the answer based on the retrieved context? (prevents hallucinations)
- **Answer Relevance**: Does the answer directly address the question?
- **Context Precision**: Are the retrieved chunks relevant to the query?
- **Context Recall**: Did we retrieve all relevant chunks?
- **Answer Correctness**: How accurate is the answer compared to ground truth?
- **Answer Similarity**: How similar is the answer to the ground truth?

#### 3. **Latency Metrics**
Measures system performance and speed:
- **Total Time**: End-to-end response time
- **Retrieval Time**: Time to search vector database
- **Generation Time**: Time for LLM to generate answer
- **Embedding Time**: Time to create query embeddings

### Benchmark Datasets

The system is evaluated on standard benchmark datasets:
- **Fast RAG Mode**: Tested on `fiqa` (Financial Q&A) - single-hop questions
- **Deep Research Mode**: Tested on `hotpot_qa` (Multi-hop Q&A) - complex reasoning tasks

### Evaluation Results

Both modes are continuously evaluated to ensure:
- ✅ **Retrieval Quality**: High precision and recall for finding relevant information
- ✅ **Answer Quality**: Faithful, relevant, and accurate answers
- ✅ **Performance**: Optimal latency for each use case
- ✅ **Reliability**: Consistent performance across different query types

The evaluation framework enables:
- Performance comparison between Fast RAG and Deep Research modes
- Continuous monitoring of system improvements
- Benchmarking against industry standards
- Identification of areas for optimization

## Usage

### Starting the Server

```bash
uv run python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### API Endpoints

#### 1. Upload Document

Upload a document (PDF, TXT, DOCX) for ingestion:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@your_document.pdf" \
  -F "title=Document Title"
```

**Response**:
```json
{
  "message": "Document uploaded and ingested successfully",
  "document": {
    "id": 1,
    "title": "Document Title",
    "file_path": "data/documents/your_document.pdf",
    "file_type": "pdf",
    "file_hash": "abc123...",
    "chunk_count": 15,
    "created_at": "2026-01-17T10:00:00"
  }
}
```

#### 2. Query Documents (Fast RAG Mode)

Ask questions about uploaded documents using Fast RAG mode (default):

```bash
curl -X POST "http://localhost:8000/api/v1/query?mode=fast" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "top_k": 5
  }'
```

**Response**:
```json
{
  "answer": "The document discusses...",
  "sources": ["document_title_1", "document_title_2"],
  "retrieved_chunks": [
    {
      "content": "Chunk text...",
      "document_id": 1,
      "chunk_index": 0,
      "metadata": {}
    }
  ],
  "response_time_ms": 1234
}
```

#### 2b. Deep Research Query

Use Deep Research mode for complex questions requiring multi-step analysis:

```bash
curl -X POST "http://localhost:8000/api/v1/query?mode=deep" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the advantages and disadvantages of different machine learning approaches mentioned across the documents"
  }'
```

**Response**:
```json
{
  "answer": "Based on comprehensive analysis across multiple sources...",
  "sources": ["document_title_1", "document_title_2", "document_title_3"],
  "findings": [
    {
      "sub_query": "What are the advantages of supervised learning?",
      "answer": "...",
      "retrieved_chunks": [...]
    },
    {
      "sub_query": "What are the disadvantages of unsupervised learning?",
      "answer": "...",
      "retrieved_chunks": [...]
    }
  ],
  "response_time_ms": 8500
}
```

#### 3. List Documents

Get all uploaded documents:

```bash
curl "http://localhost:8000/api/v1/documents"
```

#### 4. Get Document

Get a specific document by ID:

```bash
curl "http://localhost:8000/api/v1/documents/1"
```

#### 5. Delete Document

Delete a document and its associated chunks:

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/1"
```

#### 6. Run Evaluations

Evaluate system performance on benchmark datasets:

```bash
# Evaluate Fast RAG mode on fiqa dataset
curl -X POST "http://localhost:8000/api/v1/evals/fast-rag?dataset_name=fiqa"

# Evaluate Deep Research mode on hotpot_qa dataset
curl -X POST "http://localhost:8000/api/v1/evals/deep-research?dataset_name=hotpot_qa"

# Compare both modes on the same dataset
curl -X POST "http://localhost:8000/api/v1/evals/compare?dataset_name=pubmed_qa"
```

#### 7. Health Check

Check API health status:

```bash
curl "http://localhost:8000/api/v1/health"
```

## Project Structure

```
DeepResearchAgent/
├── app/
│   ├── api/
│   │   └── routes.py              # FastAPI route handlers
│   ├── services/
│   │   ├── document_service.py    # Document processing & chunking
│   │   ├── vector_service.py       # ChromaDB vector store operations
│   │   ├── rag_service.py          # Fast RAG orchestration
│   │   └── deep_research_service.py # Deep Research orchestration
│   ├── research/                   # Deep Research mode components
│   │   ├── nodes.py               # LangGraph workflow nodes
│   │   ├── state.py               # State management
│   │   └── tools.py                # Research tools (RAG tool)
│   ├── evals/                      # Evaluation framework
│   │   ├── metrics.py             # Metrics calculation (retrieval, generation, latency)
│   │   ├── datasets.py             # Benchmark dataset loading (RAGAS, BEIR)
│   │   ├── evaluators.py           # Mode-specific evaluators
│   │   ├── runners.py              # Evaluation orchestration
│   │   └── reports.py              # Report generation
│   ├── guardrails/                 # Safety and quality checks
│   │   ├── input_guardrails.py    # Input validation
│   │   ├── output_guardrails.py    # Output validation
│   │   ├── safety.py               # Content safety checks
│   │   ├── quality.py              # Answer quality checks
│   │   ├── pii_detection.py       # PII detection and redaction
│   │   └── hallucination.py       # Hallucination detection
│   ├── database.py                 # SQLAlchemy database setup
│   ├── models.py                   # SQLAlchemy ORM models
│   ├── schemas.py                  # Pydantic request/response models
│   └── main.py                     # FastAPI application entry point
├── config/
│   ├── logging_config.py           # JSON logging configuration
│   ├── settings.py                 # Application settings
│   └── __init__.py
├── data/                           # Data directory (ignored by git)
│   ├── documents/                  # Uploaded documents
│   ├── chroma_db/                  # ChromaDB vector store
│   ├── eval_datasets/              # Cached benchmark datasets
│   ├── eval_reports/               # Evaluation reports
│   └── rag_metadata.db              # SQLite metadata database
├── .env                            # Environment variables (create this)
├── .gitignore
├── pyproject.toml                  # Project configuration & dependencies
├── uv.lock                         # Dependency lock file
└── README.md
```

## Architecture

### Document Ingestion Flow

1. **Upload**: Document is uploaded via API and saved to `data/documents/`
2. **Processing**: `DocumentService` extracts text and splits into chunks
3. **Storage**: 
   - Metadata stored in SQLite via SQLAlchemy
   - Chunks embedded and stored in ChromaDB
4. **Indexing**: Vector embeddings created using OpenAI embeddings API

### Query Flow

#### Fast RAG Mode Flow

1. **Query**: User submits a question
2. **Embedding**: Query is embedded using the same embedding model
3. **Retrieval**: ChromaDB performs similarity search to find relevant chunks
4. **Filtering**: Results filtered by similarity threshold
5. **Generation**: GPT-4 generates answer using retrieved context
6. **Response**: Answer, sources, and retrieved chunks returned

#### Deep Research Mode Flow

1. **Query**: User submits a complex question
2. **Planning**: LangGraph analyzes query and creates research plan with sub-queries
3. **Iterative Research**: 
   - Execute RAG tool for each sub-query
   - Retrieve relevant chunks for each sub-query
   - Generate findings for each sub-query
4. **Gap Analysis**: Identify missing information and generate follow-up queries
5. **Synthesis**: Synthesize all findings into comprehensive answer
6. **Response**: Detailed answer with all findings, sources, and research iterations

## Development

### Running in Development Mode

```bash
uv run python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload on code changes.

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black .
uv run ruff check .
```

### Dependency Management

- **Add dependency**: `uv add package-name`
- **Add dev dependency**: `uv add --dev package-name`
- **Sync dependencies**: `uv sync`
- **Update dependencies**: `uv lock --upgrade`

## Logging

The application uses JSON-structured logging for better observability. Logs include:

- Timestamp (`asctime`)
- Log level (`levelname`)
- Module name (`name`)
- Message (`message`)
- Contextual data (`extra` fields)

Example log entry:
```json
{
  "asctime": "2026-01-17 13:29:23",
  "name": "app.services.rag_service",
  "levelname": "INFO",
  "message": "Processing RAG query",
  "query": "What is this document about?",
  "top_k": 5
}
```

## Troubleshooting

### Common Issues

1. **"I couldn't find relevant information"**: 
   - Lower `SIMILARITY_THRESHOLD` in `.env` (default: 0.3)
   - Ensure documents are successfully uploaded and indexed
   - Check ChromaDB collection count in logs

2. **Module not found errors**:
   - Run `uv sync` to install dependencies
   - Ensure you're using `uv run` to execute commands

3. **Database errors**:
   - Ensure `data/` directory exists
   - Check `DATABASE_URL` in `.env`
   - Delete `data/rag_metadata.db` to reset (will lose data)

4. **OpenAI API errors**:
   - Verify `OPENAI_API_KEY` is set in `.env`
   - Check API key validity and quota

