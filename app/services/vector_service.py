import time
import uuid
from typing import List, Dict, Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

_EMBEDDING_DIM = 1536


class VectorStore:

    def __init__(self):
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required. Add it to your .env file.")

        logger.info("Initializing Pinecone vector store", extra={"index": settings.pinecone_index_name})
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

        stats = self.index.describe_index_stats()
        logger.info("Pinecone index ready", extra={"index": self.index_name, "count": stats.total_vector_count})

    def _ensure_index(self) -> None:
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info("Creating Pinecone index", extra={"index": self.index_name})
            self.pc.create_index(
                name=self.index_name,
                dimension=_EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
            for _ in range(30):
                if self.pc.describe_index(self.index_name).status.ready:
                    break
                time.sleep(2)

    def _embedder(self) -> OpenAIEmbeddings:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Add it to your .env file.")
        return OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)

    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: Optional[List[List[float]]] = None,
        namespace: str = "",
    ) -> List[str]:
        logger.info("Adding documents to Pinecone", extra={"chunk_count": len(chunks), "namespace": namespace or "default"})

        if embeddings is None:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self._embedder().embed_documents(texts)

        vectors = []
        ids = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.get("vector_id") or str(uuid.uuid4())
            ids.append(chunk_id)
            metadata: Dict = {
                "text": chunk["text"],
                "document_id": str(chunk["document_id"]),
                "chunk_index": int(chunk["chunk_index"]),
                "chunk_id": str(chunk.get("id") or ""),
            }
            for k, v in chunk.get("metadata", {}).items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
                elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                    metadata[k] = v
            vectors.append((chunk_id, embedding, metadata))

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size], namespace=namespace)

        logger.info("Documents added to Pinecone", extra={"chunk_count": len(ids)})
        return ids

    def _query_namespace(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_dict: Optional[Dict],
        namespace: str,
    ) -> List[Dict]:
        query_kwargs: Dict = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace,
        }
        if filter_dict:
            query_kwargs["filter"] = {k: {"$eq": str(v)} for k, v in filter_dict.items()}
        results = self.index.query(**query_kwargs)
        formatted = []
        for match in results.matches:
            meta = dict(match.metadata)
            text = meta.pop("text", "")
            formatted.append({
                "text": text,
                "metadata": meta,
                "score": float(match.score),
                "distance": 1.0 - float(match.score),
            })
        return formatted

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
        session_id: Optional[str] = None,
        session_only: bool = False,  # kept for API compatibility, ignored (session_id implies session-only)
    ) -> List[Dict]:
        top_k = top_k or settings.top_k_retrieval
        total_count = self.get_count()

        logger.info(
            "Performing Pinecone search",
            extra={"query": query[:100], "top_k": top_k, "session_id": session_id, "total_count": total_count},
        )

        if total_count == 0:
            logger.warning("Pinecone index is empty")
            return []

        query_embedding = self._embedder().embed_query(query)

        try:
            if session_id:
                # Document uploaded — search the session namespace only.
                # Never mix in the global KB so it cannot override the user's document.
                results = self._query_namespace(query_embedding, top_k, filter_dict, namespace=session_id)
            else:
                # No document uploaded — fall back to the global knowledge base.
                results = self._query_namespace(query_embedding, top_k, filter_dict, namespace="")
        except Exception as e:
            logger.error("Pinecone search error", extra={"error": str(e)}, exc_info=True)
            raise

        logger.info("Search complete", extra={"result_count": len(results)})
        return results

    def delete_namespace(self, namespace: str) -> None:
        logger.info("Deleting Pinecone namespace", extra={"namespace": namespace})
        self.index.delete(delete_all=True, namespace=namespace)
        logger.info("Namespace deleted", extra={"namespace": namespace})

    def get_count(self) -> int:
        return self.index.describe_index_stats().total_vector_count

    def get_embedding(self, text: str) -> List[float]:
        return self._embedder().embed_query(text)
