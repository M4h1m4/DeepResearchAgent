from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.7

    # Relational DB — use DATABASE_URL env var in production (e.g. Neon/Supabase PostgreSQL)
    database_url: str = "sqlite:///./data/rag_metadata.db"

    # Pinecone vector DB (replaces local ChromaDB)
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "deep-research-kb"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Tavily web search (used as fallback when RAG KB has no results)
    tavily_api_key: Optional[str] = None

    # Redis — used for LangGraph checkpointing (Upstash free tier recommended for Replit)
    # If not set, falls back to in-memory MemorySaver (state lost on restart)
    redis_url: Optional[str] = None

    # HuggingFace knowledge base seeder
    # seed_max_passages: 500 is recommended for Replit free tier (512 MB RAM limit).
    # Peak RAM during seeding with 5000 passages pushes the process to ~460 MB, which
    # risks OOM on constrained environments. 500 passages seeds in ~30s vs ~5 min,
    # and still provides a meaningful knowledge base. Increase via SEED_MAX_PASSAGES
    # env var on unconstrained deployments (e.g. Railway, Fly.io, local).
    hf_dataset_name: str = "rajpurkar/squad_v2"
    seed_max_passages: int = 500
    seed_on_startup: bool = True

    app_name: str = "DeepResearchAgent"
    app_version: str = "1.0.0"
    environment: str = "development"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.3
    # No-document path: if the top KB chunk scores below this, the KB doesn't
    # really have the answer → fall back to web search instead of a junk chunk.
    web_fallback_threshold: float = 0.45

    # Session lifecycle — uploaded document vectors are deleted after this idle period
    session_ttl_hours: int = 24
    session_cleanup_interval_minutes: int = 60

    # Deep-research gap-analysis loop cap. Sub-query retrieval within each round runs
    # in parallel, so each iteration is much faster than before; 3 keeps full multi-hop
    # research depth. Lower to 1-2 if you want to trade depth for latency.
    max_research_iterations: int = 3
    deep_research_top_k: int = 6
    deep_research_temperature: float = 0.7
    research_planning_temperature: float = 0.3
    gap_analysis_temperature: float = 0.4

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

