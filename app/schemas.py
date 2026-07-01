from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class UploadResponse(BaseModel):
    session_id: str = Field(..., description="Session ID scoping these documents")
    filename: str = Field(..., description="Uploaded file name")
    chunk_count: int = Field(..., description="Number of chunks indexed")


class RAGQuery(BaseModel):
    query: str = Field(..., description="User query")
    mode: Optional[str] = Field("fast", description="Query mode: 'fast' or 'deep'")
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve")
    filter_dict: Optional[Dict] = Field(None, description="Metadata filters")
    model: Optional[str] = Field(None, description="OpenAI model (e.g. gpt-4o-mini, gpt-4o)")
    session_id: Optional[str] = Field(None, description="Session ID to include uploaded documents")


class SourceInfo(BaseModel):
    id: Optional[int] = None
    title: str
    source: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents")
    retrieved_chunks: List = Field(default_factory=list, description="Retrieved chunk IDs")
    response_time_ms: int = Field(..., description="Response time in milliseconds")
    research_metadata: Optional[Dict] = Field(default=None, description="Additional metadata for deep research mode")
