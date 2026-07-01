import os
import time
import uuid

from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config import settings
from config.logging_config import get_logger
from app.models import QueryLog
from app.document_processor import DocumentProcessor, FileType
from app.services.vector_service import VectorStore
from app.services.guardrails import PIIDetector, HallucinationGuard

logger = get_logger(__name__)

_pii_detector = PIIDetector()
_hallucination_guard = HallucinationGuard()

_tavily_client = None


def _get_tavily():
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client
    if not settings.tavily_api_key:
        return None
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        _tavily_client = TavilySearchResults(max_results=5, tavily_api_key=settings.tavily_api_key)
        logger.info("Tavily web search initialised in RAG service")
    except Exception as e:
        logger.warning("Could not initialise Tavily in RAG service", extra={"error": str(e)})
        _tavily_client = None
    return _tavily_client

_SYSTEM_TEMPLATE = (
    "You are a helpful AI assistant that answers questions based on the provided context.\n"
    "Guidelines:\n"
    "- Use only the information provided in the context to answer questions\n"
    "- If the context does not provide enough information, say so\n"
    "- Cite sources when referring to specific information\n"
    "- If you are uncertain, express the uncertainty"
)

_HUMAN_TEMPLATE = (
    "Context information:\n{context}\n\n"
    "Question: {question}\n\n"
    "Please provide a comprehensive answer based on the context above."
)

# Used when the user asks to summarize / get an overview of their uploaded document.
# Retrieves ALL chunks in document order rather than the top-k most similar chunks.
_SUMMARIZE_SYSTEM = (
    "You are a document summarization assistant. "
    "You will be given the full contents of an uploaded document, split into ordered sections. "
    "Produce a clear, comprehensive response to the user's request. "
    "Focus on the document's actual subject matter: its main topics, key arguments, findings, "
    "conclusions, and implications. "
    "Do NOT focus on structural elements like the acknowledgments, table of contents, "
    "or bibliography unless the user specifically asks about them. "
    "Write in flowing prose."
)

_SUMMARIZE_HUMAN = (
    "Document content (sections in order):\n\n{context}\n\n"
    "User request: {query}\n\n"
    "Respond thoroughly based on the document content above."
)

_SUMMARIZE_KEYWORDS = frozenset({
    "summarize", "summarise", "summary", "summarization", "summarisation",
    "overview", "abstract", "synopsis",
    "what is this document", "what does this document",
    "what is this about", "what is discussed", "what does it discuss",
    "main points", "key points", "key findings", "key takeaways",
    "describe this document", "tell me about this document",
    "what is covered", "give me an overview", "give me a summary",
})


def is_summarization_query(query: str) -> bool:
    """Return True when the user wants a summary/overview of the uploaded document."""
    q = query.lower()
    return any(kw in q for kw in _SUMMARIZE_KEYWORDS)


def _get_llm(model: Optional[str] = None) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required. Add it to your .env file.")
    return ChatOpenAI(
        model=model or settings.openai_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openai_api_key,
        request_timeout=30,
    )


class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()

    def _web_search_fallback(self, query: str, model: str, start_time: float, db: Session) -> Dict:
        """Called when no document is uploaded and the global KB has no relevant results."""
        logger.info("Falling back to web search", extra={"query": query[:200]})
        client = _get_tavily()
        if client is None:
            return {
                "answer": (
                    "I couldn't find relevant information in the knowledge base. "
                    "Web search is not configured — add a TAVILY_API_KEY to enable it."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "response_time_ms": int((time.time() - start_time) * 1000),
            }

        try:
            raw = client.invoke(query)
            snippets, sources = [], []
            for r in raw:
                if isinstance(r, dict):
                    url = r.get("url", "")
                    content = r.get("content", "")
                    title = r.get("title") or url
                    sources.append({"id": None, "title": title, "source": url})
                    if content:
                        snippets.append(f"[{title}]\n{content}")

            if not snippets:
                return {
                    "answer": "I couldn't find relevant information in the knowledge base or on the web.",
                    "sources": [],
                    "retrieved_chunks": [],
                    "response_time_ms": int((time.time() - start_time) * 1000),
                }

            context = "\n\n".join(snippets)
            llm = _get_llm(model)
            messages = [
                SystemMessage(content=_SYSTEM_TEMPLATE),
                HumanMessage(content=_HUMAN_TEMPLATE.format(context=context, question=query)),
            ]
            answer = llm.invoke(messages).content

            pii_result = _pii_detector.detect_and_redact(answer)
            if pii_result.has_pii:
                answer = pii_result.redacted_text

            response_time_ms = int((time.time() - start_time) * 1000)
            db.add(QueryLog(query=query, response=answer, retrieved_chunks=[], response_time_ms=response_time_ms))
            db.commit()

            logger.info("Web search fallback complete", extra={"sources": len(sources)})
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": [],
                "response_time_ms": response_time_ms,
                "guardrails": {
                    "pii_detected": pii_result.has_pii,
                    "pii_entity_types": [e["type"] for e in pii_result.entities],
                    "faithfulness_score": 1.0,
                    "is_grounded": True,
                    "unsupported_claims": [],
                },
            }
        except Exception as e:
            logger.warning("Web search fallback failed", extra={"error": str(e)})
            return {
                "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                "sources": [],
                "retrieved_chunks": [],
                "response_time_ms": int((time.time() - start_time) * 1000),
            }

    def ingest_session_document(
        self,
        session_id: str,
        file_path: str,
        filename: str,
        file_type: FileType,
    ) -> Dict:
        logger.info(
            "Ingesting session document",
            extra={"session_id": session_id, "file_name": filename, "file_type": file_type},
        )
        text = self.document_processor.extract_text(file_path, file_type)
        chunks_data = self.document_processor.chunk_text(text, document_id=0)

        chunks = [
            {
                "id": None,
                "text": cd["text"],
                "document_id": session_id,
                "chunk_index": cd["chunk_index"],
                "vector_id": str(uuid.uuid4()),
                "metadata": {"filename": filename, "session_id": session_id},
            }
            for cd in chunks_data
        ]

        self.vector_store.add_documents(chunks, namespace=session_id)
        logger.info("Session document ingested", extra={"session_id": session_id, "chunk_count": len(chunks)})
        return {"session_id": session_id, "filename": filename, "chunk_count": len(chunks)}

    def delete_session(self, session_id: str) -> None:
        self.vector_store.delete_namespace(session_id)
        logger.info("Session deleted", extra={"session_id": session_id})

    def query(
        self,
        db: Session,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[dict] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        force_web: bool = False,
    ) -> Dict:
        start_time = time.time()
        effective_model = model or settings.openai_model

        logger.info(
            "Processing RAG query",
            extra={"query": query[:200], "top_k": top_k or settings.top_k_retrieval, "model": effective_model, "session_id": session_id, "force_web": force_web},
        )

        # Temporal / live-data query with no uploaded document: the seeded KB is
        # static Wikipedia and can't answer "latest/current" asks, so skip retrieval
        # and go straight to web search. With a document attached we respect the doc
        # as the source of truth and fall through to normal grounding instead.
        if force_web and not session_id:
            return self._web_search_fallback(query, effective_model, start_time, db)

        # Detect summarization intent — retrieve ALL session chunks in document order
        # instead of the top-k most similar chunks (which would miss most of the document).
        summarization = session_id is not None and is_summarization_query(query)

        if summarization:
            effective_top_k = 500  # fetch all chunks (documents rarely exceed this)
        else:
            effective_top_k = top_k or settings.top_k_retrieval

        results = self.vector_store.search(
            query,
            top_k=effective_top_k,
            filter_dict=filter_dict,
            session_id=session_id,
            session_only=summarization,
        )

        if summarization:
            # Restore original document reading order via chunk_index
            results.sort(key=lambda r: int(r["metadata"].get("chunk_index", 0)))
            filtered_results = results
        else:
            max_distance = 1 - settings.similarity_threshold
            if results:
                filtered_results = [r for r in results if r["distance"] <= max_distance]
                if not filtered_results:
                    if session_id:
                        # Session doc: nothing confident, but keep the best available result
                        # rather than returning empty — the document is the source of truth.
                        filtered_results = [results[0]]
                    # else: no session, nothing relevant — fall through to web search below
            else:
                filtered_results = []

        # Relevance gate (no-document path): even if a chunk cleared the loose
        # similarity_threshold, a weak top score means the KB doesn't actually
        # have the answer — go to web search instead of answering from a junk chunk.
        if not session_id and results:
            if max(r["score"] for r in results) < settings.web_fallback_threshold:
                return self._web_search_fallback(query, effective_model, start_time, db)

        if not filtered_results:
            # No session and KB found nothing above the similarity threshold.
            # Fall back to live web search rather than returning a dead-end.
            if not session_id:
                return self._web_search_fallback(query, effective_model, start_time, db)
            return {
                "answer": "I couldn't find relevant information in the uploaded document to answer your question.",
                "sources": [],
                "retrieved_chunks": [],
                "response_time_ms": int((time.time() - start_time) * 1000),
            }

        chunk_ids = []
        if summarization:
            # Plain sections — no source headers, maintain reading flow
            context_parts = []
            total_chars = 0
            for result in filtered_results:
                if total_chars + len(result["text"]) > 200_000:
                    break
                context_parts.append(result["text"])
                total_chars += len(result["text"])
                chunk_ids.append(result["metadata"].get("chunk_id"))
            context = "\n\n---\n\n".join(context_parts)
            messages = [
                SystemMessage(content=_SUMMARIZE_SYSTEM),
                HumanMessage(content=_SUMMARIZE_HUMAN.format(context=context, query=query)),
            ]
        else:
            context_parts = []
            for i, result in enumerate(filtered_results, 1):
                metadata = result["metadata"]
                filename = metadata.get("filename", f"Document {metadata.get('document_id', i)}")
                context_parts.append(
                    f"[Source {i} - {filename}, Chunk: {metadata.get('chunk_index')}]\n{result['text']}"
                )
                chunk_ids.append(metadata.get("chunk_id"))
            context = "\n\n".join(context_parts)
            messages = [
                SystemMessage(content=_SYSTEM_TEMPLATE),
                HumanMessage(content=_HUMAN_TEMPLATE.format(context=context, question=query)),
            ]

        llm = _get_llm(effective_model)
        response = llm.invoke(messages)
        answer = response.content

        # Guardrail 1: redact PII from answer
        pii_result = _pii_detector.detect_and_redact(answer)
        if pii_result.has_pii:
            answer = pii_result.redacted_text

        # Guardrail 2: score hallucination
        hal_result = _hallucination_guard.check(question=query, answer=answer, context=context)
        if not hal_result.is_grounded:
            answer = (
                f"⚠️ Note: Parts of this answer may not be fully supported by retrieved sources "
                f"(faithfulness score: {hal_result.faithfulness_score:.2f}).\n\n{answer}"
            )

        seen = set()
        sources = []
        for result in filtered_results:
            meta = result["metadata"]
            fname = meta.get("filename", "Knowledge Base")
            sid = meta.get("session_id", "")
            key = (fname, sid)
            if key not in seen:
                seen.add(key)
                sources.append({"id": None, "title": fname, "source": sid or "global-kb"})

        response_time_ms = int((time.time() - start_time) * 1000)
        db.add(QueryLog(query=query, response=answer, retrieved_chunks=chunk_ids, response_time_ms=response_time_ms))
        db.commit()

        logger.info("RAG query completed", extra={"response_time_ms": response_time_ms, "source_count": len(sources)})

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunk_ids,
            "response_time_ms": response_time_ms,
            "guardrails": {
                "pii_detected": pii_result.has_pii,
                "pii_entity_types": [e["type"] for e in pii_result.entities],
                "faithfulness_score": hal_result.faithfulness_score,
                "is_grounded": hal_result.is_grounded,
                "unsupported_claims": hal_result.unsupported_claims,
            },
        }
