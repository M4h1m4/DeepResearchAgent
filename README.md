# Deep Research Agent

A Fast RAG (Retrieval-Augmented Generation) system built with FastAPI, SQLAlchemy, and ChromaDB. This application enables document ingestion, vector-based semantic search, and intelligent question-answering using OpenAI's language models and embeddings.

## Features

- 📄 **Document Processing**: Support for PDF, TXT, and DOCX files
- 🔍 **Semantic Search**: Vector-based retrieval using ChromaDB with OpenAI embeddings
- 🤖 **AI-Powered Q&A**: Retrieval-Augmented Generation using GPT-4 Turbo
- 💾 **Metadata Management**: SQLAlchemy-based metadata storage (SQLite)
- 🔗 **RESTful API**: FastAPI endpoints for document management and querying
- 📊 **JSON Logging**: Structured logging with JSON format for better observability
- 🚀 **Production Ready**: Clean architecture with separation of concerns

## Tech Stack

- **Framework**: FastAPI
- **Database**: SQLAlchemy (SQLite) for metadata, ChromaDB for vector storage
- **LLM & Embeddings**: OpenAI (GPT-4 Turbo, text-embedding-3-small)
- **Document Processing**: PyPDF, python-docx
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter
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

#### 2. Query Documents

Ask questions about uploaded documents:

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
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

#### 6. Health Check

Check API health status:

```bash
curl "http://localhost:8000/api/v1/health"
```

## Project Structure

```
DeepResearchAgent/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI route handlers
│   ├── services/
│   │   ├── document_service.py # Document processing & chunking
│   │   ├── vector_service.py   # ChromaDB vector store operations
│   │   └── rag_service.py      # RAG orchestration
│   ├── database.py             # SQLAlchemy database setup
│   ├── models.py               # SQLAlchemy ORM models
│   ├── schemas.py              # Pydantic request/response models
│   └── main.py                 # FastAPI application entry point
├── config/
│   ├── logging_config.py       # JSON logging configuration
│   ├── settings.py             # Application settings
│   └── __init__.py
├── data/                       # Data directory (ignored by git)
│   ├── documents/              # Uploaded documents
│   ├── chroma_db/              # ChromaDB vector store
│   └── rag_metadata.db         # SQLite metadata database
├── .env                        # Environment variables (create this)
├── .gitignore
├── pyproject.toml              # Project configuration & dependencies
├── uv.lock                     # Dependency lock file
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

1. **Query**: User submits a question
2. **Embedding**: Query is embedded using the same embedding model
3. **Retrieval**: ChromaDB performs similarity search to find relevant chunks
4. **Filtering**: Results filtered by similarity threshold
5. **Generation**: GPT-4 generates answer using retrieved context
6. **Response**: Answer, sources, and retrieved chunks returned

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

