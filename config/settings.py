from pydantic_settings import BaseSettings 
from typing import Optional 

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"

    database_url: str = "sqlite:///./data/rag_metadata.db"

    chroma_persist_directory: str = "./data/chroma_db"

    app_name: str = "DeepResearchAgent"
    app_version: str = "1.0.0"
    environment: str = "development"
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.3  # Cosine similarity threshold (0.3 = 30% similarity, accepts distance up to 0.7) 

    class Config:
        env_file =".env"
        case_sensitive = False 


settings = Settings()
    
