from contextlib import asynccontextmanager
import os 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from config.logging_config import setup_json_logging, get_logger
from app.database import db
from app.api.routes import router

# Setup JSON logging
setup_json_logging(log_level=settings.environment.upper() if hasattr(settings, 'environment') else "INFO")
logger = get_logger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup - creating directories")
    data_dirs = [
        "data/documents",
        "data/chroma_db"
    ]
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Directory ensured: {dir_path}", extra={"directory": dir_path})
    logger.info("Application startup completed - directories created")
    
    yield
    
    logger.info("Application shutdown initiated")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Deep Research Agent - Phase 1 Fast RAG API",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["RAG"])


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Application startup initiated")
    db.create_tables()
    logger.info("Database tables created")
    logger.info(
        "Application started successfully",
        extra={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment
        }
    )


@app.get("/")
def root():
    """Root endpoint."""
    logger.debug("Root endpoint accessed")
    return {
        "message": "Deep Research Agent API",
        "version": settings.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)