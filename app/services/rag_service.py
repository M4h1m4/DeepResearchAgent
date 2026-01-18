import os 
import time 

from typing import List, Dict, Optional 
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.schema import HumanMessage, SystemMessage 

from config import settings 
from config.logging_config import get_logger 
from app.models import Document, Chunk, QueryLog
from app.services.document_service import DocumentProcessor
from app.services.vector_service import VectorStore

logger = get_logger(__name__)

class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm = ChatOpenAI(
            model=settings.openai_model, 
            temperature=0.7, 
            openai_api_key=settings.openai_api_key,
        )
        self._setup_prompts()

    def _setup_prompts(self):
        system_template = """ You are a helpful AI assistant that answers questions based on the 
        provided context . 
        Guidelines: 
        - Use only the information provided in the context to answer questions 
        - If the context does not provide enough information say so
        - Cite source when referring specific information 
        - If you are unceratin express the uncertainity""" 
        human_template= """context information: {context}
        Question: {question}
        Please provide a comprehensive answer based on the context above.""" 

        self.system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        self.human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        # Store template strings for direct access if needed
        self.system_template = system_template
        self.human_template = human_template

    def ingest_document(
        self, 
        db: Session, 
        file_path: str, 
        title: str, 
        file_type: str,
        metadata: Optional[Dict] = None
    ) -> Document: 
    #Ingest a document into the database and return a database object
        logger.info(
            "Starting document ingestion",
            extra={
                "file_path": file_path,
                "title": title,
                "file_type": file_type
            }
        )
        text = self.document_processor.extract_text(file_path, file_type)
        logger.debug("Text extracted from document", extra={"text_length": len(text)})
        summary = self.document_processor.generate_summary(text)
        logger.debug("Document summary generated", extra={"summary_length": len(summary)})

        file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0 

        logger.debug("Creating document record in database")


        #create the document record
        document = Document(
            title=title,
            source=file_path,
            file_type=file_type,
            file_size=file_size,
            summary=summary,
            extra_metadata=metadata or {}
        )

        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(
            "Document record created",
            extra={"document_id": document.id, "title": title}
        )

        #store the chunks 

        chunks_data = self.document_processor.chunk_text(text, document.id)

        logger.debug(
            "Text chunked",
            extra={"document_id": document.id, "chunk_count": len(chunks_data)}
        )

        logger.debug("Storing chunks in database")

        chunks = []

        for chunk_data in chunks_data:
            chunk = Chunk(
                document_id=chunk_data["document_id"],
                chunk_index=chunk_data["chunk_index"],
                text=chunk_data["text"],
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                chunk_metadata=chunk_data["metadata"]
            )
            db.add(chunk)
            chunks.append(chunk)
        db.commit()

        for chunk in chunks:
            db.refresh(chunk)
        logger.info(
            "Chunks stored in database",
            extra={"document_id": document.id, "chunk_count": len(chunks)}
        )

        #prepare chunks for vector store
        vector_chunks =[]
        for chunk, chunk_data in zip(chunks, chunks_data):
            vector_chunk = {
                "id": chunk.id,
                "text": chunk.text,
                "document_id": document.id,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.chunk_metadata or {}
            }
            vector_chunks.append(vector_chunk)

        logger.debug("Adding chunks to vector store")
        vector_ids = self.vector_store.add_documents(vector_chunks)
        logger.info(
            "Chunks added to vector store",
            extra={"document_id": document.id, "vector_chunk_count": len(vector_ids)}
        )
        
        # Update chunks with vector IDs
        for chunk, vector_id in zip(chunks, vector_ids):
            if not chunk.chunk_metadata:
                chunk.chunk_metadata = {}
            chunk.chunk_metadata["vector_id"] = vector_id
        db.commit()
        
        logger.info(
            "Document ingestion completed",
            extra={
                "document_id": document.id,
                "title": title,
                "chunk_count": len(chunks)
            }
        )
        
        return document

    def query(self, db: Session, query: str, top_k: Optional[int]=None, 
                filter_dict: Optional[dict]=None
    ) -> Dict: 
        start_time = time.time()
        logger.info(
            "Processing RAG query",
            extra={
                "query": query[:200],  # Log first 200 chars
                "top_k": top_k or settings.top_k_retrieval,
                "filter_dict": filter_dict
            }
        )

        top_k = top_k or settings.top_k_retrieval

        results = self.vector_store.search(query, top_k=top_k, filter_dict=filter_dict)
        logger.debug( 
            "Vector search completed",
            extra={
                "result_count": len(results),
                "sample_distances": [r["distance"] for r in results[:3]] if results else []
            }
        )
        
        # Chroma returns cosine distance (lower = better)
        # Convert distance to similarity: similarity = 1 - distance
        # Then filter by similarity threshold
        # Note: For cosine distance, 0 = identical, 1 = opposite
        # Typical good matches: distance 0.1-0.5 (similarity 0.9-0.5)
        # However, for RAG, we should accept lower similarities if they're the best we have
        # Lower threshold means accepting less similar results (more lenient)
        # Higher threshold means only accepting very similar results (more strict)
        
        # If we have results but they're below threshold, use them anyway if they're the best we have
        # Only filter if we have many results and can be selective
        if len(results) > 0:
            # Use a more lenient threshold, or accept top results even if below threshold
            # For now, use threshold but be more lenient (threshold is already distance-based)
            max_distance = 1 - settings.similarity_threshold  # Convert similarity threshold to max distance
            # For cosine distance, accept results with distance <= max_distance
            # But if all results are above max_distance, use the best ones anyway
            filtered_results = [
                r for r in results
                if r["distance"] <= max_distance
            ]
            # If nothing passes threshold but we have results, use the best one anyway
            if not filtered_results and results:
                logger.debug(
                    "No results met threshold, using best available result",
                    extra={
                        "threshold_distance": max_distance,
                        "best_distance": results[0]["distance"],
                        "best_similarity": 1 - results[0]["distance"]
                    }
                )
                filtered_results = [results[0]]  # Use best result even if below threshold
        else:
            filtered_results = []

        logger.debug(
            "Filtered results by similarity threshold",
            extra={
                "original_count": len(results),
                "filtered_count": len(filtered_results),
                "similarity_threshold": settings.similarity_threshold,
                "sample_distances": [r["distance"] for r in results[:3]] if results else [],
                "sample_similarities": [1 - r["distance"] for r in results[:3]] if results else []
            }
        )


        if not filtered_results:
            logger.warning(
                "No relevant chunks found for query",
                extra={
                    "query": query[:200],
                    "similarity_threshold": settings.similarity_threshold,
                    "original_result_count": len(results)
                }
            )
            return {
                "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                "sources": [],
                "retrieved_chunks": [],
                "response_time_ms": int((time.time() - start_time) * 1000)
            }

        
        #preparing context parts 

        context_parts = []
        chunk_ids = []
        for i, result in enumerate(filtered_results, 1):
            chunk_text = result["text"]
            metadata = result["metadata"]
            document_id = metadata.get("document_id")
            chunk_id = metadata.get("chunk_id")

            context_parts.append(f"[Source {i} - Document ID: {document_id}, Chunk: {metadata.get('chunk_index')}]\n{chunk_text}")
            chunk_ids.append(chunk_id)

        context = "\n\n".join(context_parts)

        logger.debug(
            "Generating answer with LLM",
            extra={
                "context_length": len(context),
                "query_length": len(query),
                "chunk_count": len(filtered_results)
            }
        )

        # Create messages using prompt templates
        try:
            system_messages = self.system_prompt.format_messages() if hasattr(self.system_prompt, 'format_messages') else [SystemMessage(content=self.system_template)]
            human_messages = self.human_prompt.format_messages(context=context, question=query) if hasattr(self.human_prompt, 'format_messages') else [HumanMessage(content=self.human_template.format(context=context, question=query))]
            messages = system_messages + human_messages
        except (AttributeError, TypeError):
            # Fallback to direct template access
            messages = [
                SystemMessage(content=self.system_template),
                HumanMessage(content=self.human_template.format(context=context, question=query))
            ]

        response = self.llm(messages)
        answer = response.content 
        logger.info(
            "LLM answer generated",
            extra={
                "answer_length": len(answer)
            }
        )

        source_doc_ids = list(set(r["metadata"].get("document_id") for r in filtered_results))
        sources = db.query(Document).filter(Document.id.in_(source_doc_ids)).all()

        logger.debug(
            "Source documents retrieved",
            extra={"source_count": len(sources), "source_doc_ids": source_doc_ids}
        )

        response_time_ms = int((time.time() - start_time) * 1000)

        query_log = QueryLog(
            query=query, 
            response=answer, 
            retrieved_chunks=chunk_ids, 
            response_time_ms=response_time_ms
        )
        db.add(query_log)
        db.commit()
        logger.debug("Query logged to database", extra={"query_log_id": query_log.id})
        logger.info(
            "RAG query completed",
            extra={
                "query": query[:200],
                "response_time_ms": response_time_ms,
                "chunk_count": len(chunk_ids),
                "source_count": len(sources)
            }
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "id": doc.id, 
                    "title": doc.title, 
                    "source": doc.source
                }
                for doc in sources
            ],
            "retrieved_chunks": chunk_ids, 
            "response_time_ms": response_time_ms
        }


    def delete_document(self, db:Session, document_id: int) -> bool: 
        logger.info("Deleting document", extra={"document_id": document_id})
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            logger.warning(
                "Document not found for deletion", extra={"document_id": document_id}
            )
            return False 
        
        self.vector_store.delete_document(document_id)
        logger.debug("Document deleted from vector store", extra={"document_id": document_id})


        db.delete(document)
        db.commit()
        logger.info("Document deleted from database", extra={"document_id": document_id})

        return True 






