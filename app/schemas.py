from pydantic import BaseModel, Field 
from typing import Optional, List, Dict 
from datetime import datetime 

class DocumentBase(BaseModel):
    title: str = Field(..., description="Document title")
    source: Optional[str] = Field(None, description="Document source/path")
    file_type: Optional[str] = Field(None, description="File type")
    summary: Optional[str] = Field(None, description="Document summary")
    extra_metadata: Optional[Dict] = Field(default_factory=dict, description="Additional metadata")

class DocumentCreate(DocumentBase):
    pass

class DocumentMetadata(DocumentBase):
    id: int
    file_size: Optional[int]
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}

class RAGQuery(BaseModel):
    query: str = Field(..., description="User query")
    mode: Optional[str] = Field("fast", description="Query mode: 'fast' or 'deep'")
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve")
    filter_dict: Optional[Dict] = Field(None, description="Metadata filters")


class SourceInfo(BaseModel):
    id: int
    title: str
    source: Optional[str]


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents")
    retrieved_chunks: List[int] = Field(default_factory=list, description="Retrieved chunk IDs")
    response_time_ms: int = Field(..., description="Response time in milliseconds")
    research_metadata: Optional[Dict] = Field(default=None, description="Additional metadata for deep research mode")


