import aiofiles
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.rag_service import RAGService
from app.schemas import DocumentMetadata, RAGQuery, QueryResponse
from app.models import Document, QueryLog
from app.document_processor import FileType
from app.services.deep_research_service import DeepResearchService
from app.schemas import SourceInfo
from app.evals.runners import EvaluationRunner
from app.evals.datasets import DatasetManager, EvaluationDataset

from config import settings 
from config.logging_config import get_logger 

router = APIRouter()
rag_service = RAGService()
deep_research_service = DeepResearchService()

evaluation_runner = EvaluationRunner()
dataset_manager = DatasetManager()


logger = get_logger(__name__)

# os.makedirs("data/documents", exist_ok=True)
# os.makedirs("data/chroma_db", exist_ok=True)

@router.post("/documents/upload", response_model=DocumentMetadata)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None), 
    db: Session = Depends(get_db)
):
    logger.info(
        "Document upload request received",
        extra={
            "uploaded_filename": file.filename,
            "content_type": file.content_type,
            "title": title
        }
    )
    try: 
        file_ext = file.filename.split(".")[-1].lower() if file.filename else ""
        try:
            file_type = FileType(file_ext)
        except ValueError:
            logger.warning(
                "Unsupported file type",
                extra={"file_type": file_ext, "uploaded_filename": file.filename}
            )
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Supported: {', '.join([ft.value for ft in FileType])}"
            )
        
        file_path = f"data/documents/{file.filename}"
        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
        logger.debug(
            "File saved",
            extra={"file_path": file_path, "file_size": len(content)}
        )

        document_title = title or file.filename

        document = rag_service.ingest_document(
            db=db,
            file_path=file_path,
            title=document_title,
            file_type=file_type
        )
        logger.info(
            "Document uploaded and ingested successfully",
            extra={"document_id": document.id, "title": document.title}
        )

        return DocumentMetadata.model_validate(document)

    except HTTPException:
        raise 
    except Exception as e:
        logger.error(
            "Error uploading document",
            extra={"uploaded_filename": file.filename, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

    
@router.get("/documents", response_model=List[DocumentMetadata])
def list_documents(skip: int=0, limit: int = 100, db: Session=Depends(get_db)):
    logger.info("Listing documents", extra={"skip": skip, "limit": limit})
    documents = db.query(Document).offset(skip).limit(limit).all()
    logger.debug("Documents retrieved", extra={"count": len(documents)})
    return [DocumentMetadata.model_validate(doc) for doc in documents]  

@router.get("/documents/{document_id}", response_model=DocumentMetadata)
def get_document(document_id: int, db: Session = Depends(get_db)):
    logger.info("Getting document", extra={"document_id": document_id})
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        logger.warning("Document not found", extra={"document_id": document_id})
        raise HTTPException(status_code=404, detail="Document not found")
    logger.debug("Document retrieved", extra={"document_id": document_id, "title": document.title})
    return DocumentMetadata.model_validate(document)  

@router.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session=Depends(get_db)):
    logger.info("Delete document request", extra={"document_id": document_id})
    success = rag_service.delete_document(db, document_id)
    if not success:
        logger.warning("Document not found for deletion", extra={"document_id": document_id})
        raise HTTPException(status_code=404, detail="Document not found")
    logger.info("Document deleted successfully", extra={"document_id": document_id})
    return {"message": "Document deleted successfully"}


@router.post("/query/deep", response_model=QueryResponse)
async def deep_research_query(
    query: RAGQuery, 
    db: Session = Depends(get_db)
): 
    logger.info(
        "Deep research query received",
        extra={
            "query": query.query,
            "endpoint": "/api/v1/query/deep"
        }
    )

    try:
        result = deep_research_service.research(query.query)
        query_log = QueryLog(
            query=query.query,
            response=result["answer"],
            retrieved_chunks=result.get("chunk_ids", []),
            response_time_ms=result["response_time_ms"]
        )
        db.add(query_log)
        db.commit()
        logger.info(
            "Deep research query completed",
            extra={
                "query": query.query,
                "answer_length": len(result["answer"]),
                "iterations": result["iteration_count"],
                "response_time_ms": result["response_time_ms"]
            }
        )
        
        # Convert sources to SourceInfo objects
        source_infos = [
            SourceInfo(**source) if isinstance(source, dict) else source
            for source in result.get("sources", [])
        ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=source_infos,
            retrieved_chunks=result.get("chunk_ids", []),
            response_time_ms=result["response_time_ms"],
            research_metadata={  # Additional metadata for deep research
                "mode": "deep",
                "iterations": result["iteration_count"],
                "sub_queries": result["sub_queries"],
                "research_plan": result.get("research_plan", "")
            }
        )
        
    except Exception as e:
        logger.error(
            "Error in deep research query",
            extra={
                "query": query.query,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error executing deep research: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse)
async def query(request: RAGQuery, mode: str = "fast", db: Session=Depends(get_db)):
    logger.info(
        "Query request received",
        extra={
            "query": request.query[:200],
            "top_k": request.top_k,
            "filter_dict": request.filter_dict,
            "mode": mode
        }
    )
    try:
        if mode == "deep":
            return await deep_research_query(request, db)
        else:
        result = rag_service.query(
            db=db, 
            query=request.query,
            top_k=request.top_k, 
            filter_dict=request.filter_dict
        )
        logger.info(
            "Query processed successfully",
            extra={
                    "mode": mode,
                "response_time_ms": result.get("response_time_ms"),
                "chunk_count": len(result.get("retrieved_chunks", []))
            }
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(
            "Error processing query",
            extra={
                "query": request.query[:200], 
                "error": str(e)
            },
                exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version
    }

@router.post("/evals/fast-rag")
def evaluate_fast_rag(
    dataset_name: Optional[str] = None, 
    dataset: Optional[Dict] = None, 
    db: Session = Depends(get_db)
): 
    if dataset:
        eval_dataset = EvaluationDataset.from_dict(dataset)
    else:
        eval_dataset = None
    
    results = evaluation_runner.run_fast_rag_evaluation(
        dataset_name=dataset_name,
        dataset=eval_dataset
    )
    return results

@router.post("/evals/deep-research")
def evaluate_deep_research(
    dataset_name: Optional[str] = None,
    dataset: Optional[Dict] = None,
    db: Session = Depends(get_db)
):
    """
    Recommended dataset: "hotpot_qa" (multi-hop Q&A requiring reasoning)
    Example: POST /evals/deep-research?dataset_name=hotpot_qa
    """
    if dataset:
        eval_dataset = EvaluationDataset.from_dict(dataset)
    else:
        eval_dataset = None
    
    results = evaluation_runner.run_deep_research_evaluation(
        dataset_name=dataset_name,
        dataset=eval_dataset
    )
    return results

@router.post("/evals/compare")
def compare_modes(
    dataset_name: Optional[str] = None,
    dataset: Optional[Dict] = None,
    db: Session = Depends(get_db)
):
    if dataset:
        eval_dataset = EvaluationDataset.from_dict(dataset)
    else:
        eval_dataset = None
    
    results = evaluation_runner.compare_modes(
        dataset_name=dataset_name,
        dataset=eval_dataset
    )
    return results