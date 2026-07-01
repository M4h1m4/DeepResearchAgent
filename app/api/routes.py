import os
import re
import uuid
import asyncio
import aiofiles
from typing import Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.rag_service import RAGService, is_summarization_query
from app.services.session_service import SessionService
from app.schemas import UploadResponse, RAGQuery, QueryResponse, SourceInfo
from app.models import QueryLog
from app.document_processor import FileType
from app.services.deep_research_service import DeepResearchService

from config import settings
from config.logging_config import get_logger

try:
    from app.evals.runners import EvaluationRunner
    from app.evals.datasets import DatasetManager, EvaluationDataset
    _evals_available = True
except Exception:
    _evals_available = False

logger = get_logger(__name__)

router = APIRouter()
session_service = SessionService()

# Queries containing these words need live web data — auto-upgrade to deep research
# which has the Tavily web search gate.
_TEMPORAL_KEYWORDS = frozenset({
    "latest", "recent", "recently", "current", "currently", "today", "now",
    "this year", "this month", "this week", "breaking", "just announced",
    "new", "newest", "emerging", "state of the art", "state-of-the-art",
    "cutting edge", "cutting-edge", "2024", "2025", "2026",
})


# Match keywords as whole words only. Substring matching wrongly fired on words
# that merely contain a keyword (e.g. "now" inside "known"), forcing an unwanted
# — and slow — upgrade to deep research.
_TEMPORAL_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _TEMPORAL_KEYWORDS) + r")\b"
)


def _has_temporal_intent(query: str) -> bool:
    return bool(_TEMPORAL_PATTERN.search(query.lower()))

# Lazy singletons — instantiated on first request, not at import time.
# This prevents a Pinecone/OpenAI connection failure at startup from crashing
# the entire app before it can serve any traffic.
_rag_service: Optional[RAGService] = None
_deep_research_service: Optional[DeepResearchService] = None
_evaluation_runner = None
_dataset_manager = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_deep_research_service() -> DeepResearchService:
    global _deep_research_service
    if _deep_research_service is None:
        _deep_research_service = DeepResearchService()
    return _deep_research_service


def get_evaluation_runner():
    global _evaluation_runner
    if _evaluation_runner is None and _evals_available:
        _evaluation_runner = EvaluationRunner()
    return _evaluation_runner


def get_dataset_manager():
    global _dataset_manager
    if _dataset_manager is None and _evals_available:
        _dataset_manager = DatasetManager()
    return _dataset_manager


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a document. Embeddings live only for the session TTL (default 24 h of inactivity)."""
    logger.info("Document upload request", extra={"file_name": file.filename})
    try:
        file_ext = file.filename.split(".")[-1].lower() if file.filename else ""
        try:
            file_type = FileType(file_ext)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(ft.value for ft in FileType)}",
            )

        effective_session_id = session_id or str(uuid.uuid4())
        tmp_path = f"data/documents/tmp_{effective_session_id}_{file.filename}"

        async with aiofiles.open(tmp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        try:
            # Offload the blocking ingest (embeddings + Pinecone upsert) to a worker
            # thread so it doesn't stall the event loop for other users.
            result = await asyncio.to_thread(
                get_rag_service().ingest_session_document,
                session_id=effective_session_id,
                file_path=tmp_path,
                filename=file.filename,
                file_type=file_type,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Record session so TTL cleanup can track it
        session_service.touch(db, effective_session_id)
        logger.info("Session document ingested", extra={"session_id": effective_session_id})
        return UploadResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload error", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Immediately delete all vectors and the session record."""
    try:
        session_service.delete(db, session_id)
        return {"deleted": session_id}
    except Exception as e:
        logger.error("Session delete error", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: RAGQuery,
    mode: str = "fast",
    db: Session = Depends(get_db),
):
    logger.info("Query request", extra={"query": request.query[:200], "mode": mode, "session_id": request.session_id})
    try:
        # Refresh TTL whenever the user actively queries their session
        if request.session_id:
            session_service.touch(db, request.session_id)

        # Temporal queries ("latest", "current", "2026", etc.) need live web data.
        temporal = _has_temporal_intent(request.query)

        # Summarization always runs as fast RAG, which pulls the whole document in
        # order for a clean summary. Deep's multi-hop pipeline would splinter
        # "summarize" into research sub-queries and pad the summary with unrelated
        # web results, so summarization overrides the user's selected mode. This is
        # the only automatic mode change — nothing auto-upgrades to deep.
        summarizing = bool(request.session_id) and is_summarization_query(request.query)
        if summarizing:
            mode = "fast"

        # No document: keep the user's chosen mode and just force fast RAG's cheap
        # web-search path rather than jumping to the slow deep-research pipeline.
        force_web = temporal and not request.session_id

        # A temporal query with a document attached does NOT auto-upgrade to deep —
        # the user's selected mode is respected. In fast mode the answer stays
        # grounded to the document; the user picks deep explicitly to combine the
        # document with live web results.

        if mode == "deep":
            return await _deep_query(request, db)

        # Offload blocking retrieval + LLM call to a worker thread so concurrent
        # users are served in parallel instead of queueing behind the event loop.
        result = await asyncio.to_thread(
            get_rag_service().query,
            db=db,
            query=request.query,
            top_k=request.top_k,
            filter_dict=request.filter_dict,
            model=request.model,
            session_id=request.session_id,
            force_web=force_web,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) if isinstance(s, dict) else s for s in result.get("sources", [])],
            retrieved_chunks=result.get("retrieved_chunks", []),
            response_time_ms=result["response_time_ms"],
            research_metadata={"mode": "fast", "guardrails": result.get("guardrails")},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Query error", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/query/deep", response_model=QueryResponse)
async def query_deep(request: RAGQuery, db: Session = Depends(get_db)):
    try:
        if request.session_id:
            session_service.touch(db, request.session_id)
        return await _deep_query(request, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Deep query error", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error executing deep research: {str(e)}")


async def _deep_query(request: RAGQuery, db: Session) -> QueryResponse:
    # The whole LangGraph run (many blocking LLM/Pinecone/Tavily calls) is offloaded
    # to a worker thread. asyncio.to_thread copies the contextvars, so each concurrent
    # request keeps its own model/session_id in the RAG tool context — no cross-talk.
    result = await asyncio.to_thread(
        get_deep_research_service().research,
        query=request.query,
        model=request.model,
        session_id=request.session_id,
    )
    db.add(QueryLog(
        query=request.query,
        response=result["answer"],
        retrieved_chunks=result.get("chunk_ids", []),
        response_time_ms=result["response_time_ms"],
    ))
    db.commit()

    source_infos = [SourceInfo(**s) if isinstance(s, dict) else s for s in result.get("sources", [])]
    return QueryResponse(
        answer=result["answer"],
        sources=source_infos,
        retrieved_chunks=result.get("chunk_ids", []),
        response_time_ms=result["response_time_ms"],
        research_metadata={
            "mode": "deep",
            "iterations": result["iteration_count"],
            "sub_queries": result["sub_queries"],
            "research_plan": result.get("research_plan", ""),
            "guardrails": result.get("guardrails"),
        },
    )


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    checks = {}
    overall = "healthy"

    # Database
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {e}"
        overall = "unhealthy"

    # Pinecone
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)
        pc.describe_index(settings.pinecone_index_name)
        checks["pinecone"] = "healthy"
    except Exception as e:
        checks["pinecone"] = f"unhealthy: {e}"
        overall = "unhealthy"

    # OpenAI
    try:
        import openai
        openai.OpenAI(api_key=settings.openai_api_key).models.list()
        checks["openai"] = "healthy"
    except Exception as e:
        checks["openai"] = f"unhealthy: {e}"
        overall = "unhealthy"

    return JSONResponse(
        status_code=200 if overall == "healthy" else 503,
        content={
            "status": overall,
            "app_name": settings.app_name,
            "version": settings.app_version,
            "checks": checks,
            "evals_available": _evals_available,
        },
    )


@router.post("/evals/fast-rag")
def evaluate_fast_rag(dataset_name: Optional[str] = None, dataset: Optional[Dict] = None, db: Session = Depends(get_db)):
    if not _evals_available:
        raise HTTPException(status_code=503, detail="Evaluation dependencies not installed.")
    eval_dataset = EvaluationDataset.from_dict(dataset) if dataset else None
    return get_evaluation_runner().run_fast_rag_evaluation(dataset_name=dataset_name, dataset=eval_dataset)


@router.post("/evals/deep-research")
def evaluate_deep_research(dataset_name: Optional[str] = None, dataset: Optional[Dict] = None, db: Session = Depends(get_db)):
    if not _evals_available:
        raise HTTPException(status_code=503, detail="Evaluation dependencies not installed.")
    eval_dataset = EvaluationDataset.from_dict(dataset) if dataset else None
    return get_evaluation_runner().run_deep_research_evaluation(dataset_name=dataset_name, dataset=eval_dataset)


@router.post("/evals/compare")
def compare_modes(dataset_name: Optional[str] = None, dataset: Optional[Dict] = None, db: Session = Depends(get_db)):
    if not _evals_available:
        raise HTTPException(status_code=503, detail="Evaluation dependencies not installed.")
    eval_dataset = EvaluationDataset.from_dict(dataset) if dataset else None
    return get_evaluation_runner().compare_modes(dataset_name=dataset_name, dataset=eval_dataset)
