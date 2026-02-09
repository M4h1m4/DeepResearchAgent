from typing import List, Dict, Tuple, Optional 
import uuid 

import chromadb
from chromadb.config import Settings as ChromaSettings 
from langchain_openai import OpenAIEmbeddings 
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
from langchain.vectorstores import Chroma 

from config import settings 
from config.logging_config import get_logger 

logger = get_logger(__name__)

class VectorStore:

    def __init__(self):
        logger.info(
            "Initializing Vector Store", 
            extra={
                "persist_directory": settings.chroma_persist_directory, 
                "embedding_model": settings.embedding_model
            }
        )
        self.persist_directory = settings.chroma_persist_directory
        self.embeddings = OpenAIEmbeddings(
            model = settings.embedding_model, 
            openai_api_key=settings.openai_api_key
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory, 
            settings = ChromaSettings(anonymized_telemetry=False)
        )
        logger.debug("Chroma client initialized", extra={"persist_directory": self.persist_directory})

        self.collection_name = "documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            "Chroma collection ready",
            extra={"collection_name": self.collection_name, "collection_count": self.collection.count()}
        )

        # Initialize Chroma vectorstore - use collection parameter for langchain-chroma
        try:
            # Try new langchain-chroma API
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize with collection_name, trying alternative: {e}")
            # Fallback: use collection directly
            self.vectorstore = Chroma(
                client=self.client,
                collection=self.collection,
                embedding_function=self.embeddings,
            )
        
        logger.info(
            "Vector store initialization completed",
            extra={"collection_name": self.collection_name}
        )


    def add_documents(self, chunks: List[Dict]) -> List[str]:
        logger.info(
            "Adding document into vector store", 
            extra={
                "chunk_count": len(chunks)
            }
        )

        texts = [chunk["text"] for chunk in chunks]
        metadatas = []
        ids = []

        for chunk in chunks:
            metadata={
                "document_id": chunk["document_id"], 
                "chunk_index": chunk["chunk_index"],
                "chunk_id":  chunk.get("id"), 
                **chunk.get("metadata", {})
            }
            metadatas.append(metadata)

            chunk_id = chunk.get("vector_id") or str(uuid.uuid4())
            ids.append(chunk_id)
        
        logger.debug(
            "Prepared chunks for vector store",
            extra={
                "total_chunks": len(chunks),
                "document_ids": list(set(chunk["document_id"] for chunk in chunks))
            }
        )

        try:
            self.vectorstore.add_texts( #Embeddings are created here
            texts=texts, 
            metadatas=metadatas, 
            ids=ids
        )
        logger.info(
            "Documents added to vector store",
            extra={"chunk_count": len(ids), "vector_ids": ids[:5]}  # Log first 5 IDs
            )
        except Exception as e:
            logger.error(
                "Error adding documents to vector store",
                extra={"error": str(e), "chunk_count": len(ids)},
                exc_info=True
            )
            raise

        # Verify documents were added
        collection_count = self.collection.count()
        logger.debug(
            "Vector store collection count after add",
            extra={"collection_count": collection_count, "added_count": len(ids)}
        )

        return ids 
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None, 
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        top_k = top_k or settings.top_k_retrieval
        # Check collection count before search
        collection_count = self.collection.count()
        logger.info(
            "Performing vector search",
            extra={
                "query": query[:100],  # Log first 100 chars
                "top_k": top_k,
                "filter_dict": filter_dict,
                "collection_count": collection_count
            }
        )

        if collection_count == 0:
            logger.warning("Vector store collection is empty - no documents have been indexed")
            return []

        # LangChain Chroma's similarity_search_with_score doesn't support where parameter
        # Filtering must be done through metadata using filter parameter
        # Note: filter_dict should contain metadata keys to filter by
        filter_metadata = None
        if filter_dict:
            filter_metadata = filter_dict

        # Use similarity_search_with_score with optional filter
        # Note: Chroma filtering syntax may vary - this uses metadata-based filtering
        try:
            if filter_metadata:
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k, 
                    filter=filter_metadata
                )
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=top_k
                )
        except TypeError:
            # Fallback if filter parameter not supported - search without filter
            logger.warning(
                "Filter parameter not supported, performing search without filter",
                extra={"filter_dict": filter_dict}
            )
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k
        )

        logger.debug(
            "Vector search completed",
            extra={"result_count": len(results)}
        )

        formatted_results = []
        for doc, score in results:
            result = {
                "text": doc.page_content, 
                "metadata": doc.metadata, 
                "score": float(score),
                "distance": float(score)
            }
            formatted_results.append(result)
        
        logger.info(
            "Vector search results formatted",
            extra={
                "result_count": len(formatted_results),
                "avg_score": sum(r["score"] for r in formatted_results) / len(formatted_results) if formatted_results else 0
            }
        )

        return formatted_results

    def delete_document(self, document_id: int) -> None:
        logger.info(
            "Deleting document from vector store",
            extra={"document_id": document_id}
        )
        self.collection.delete(
            where={"document_id": document_id}
        )
        logger.info(
            "Document deleted from vector store",
            extra={"document_id": document_id}
        )

    def get_embedding(self, text: str) -> List[float]: 
        logger.debug(
            "Generating embedding",
            extra={"text_length": len(text)}
        )
        embedding = self.embeddings.embed_query(text)
        logger.debug(
            "Embedding generated",
            extra={"text_length": len(text), "embedding_dimension": len(embedding)}
        )
        return embedding 
    
    