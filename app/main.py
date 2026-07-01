import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import settings
from config.logging_config import setup_json_logging, get_logger
from app.database import db
from app.api.routes import router

setup_json_logging(log_level="DEBUG" if settings.environment == "development" else "INFO")
logger = get_logger(__name__)


async def _session_cleanup_loop() -> None:
    """Runs every session_cleanup_interval_minutes and purges idle sessions."""
    interval = settings.session_cleanup_interval_minutes * 60
    while True:
        await asyncio.sleep(interval)
        try:
            from app.services.session_service import SessionService
            session_gen = db.get_session()
            db_session = next(session_gen)
            try:
                deleted = SessionService().delete_expired(db_session)
                if deleted:
                    logger.info(
                        "Background session cleanup",
                        extra={"deleted": deleted, "ttl_hours": settings.session_ttl_hours},
                    )
            finally:
                db_session.close()
        except Exception as e:
            logger.error("Session cleanup loop error", extra={"error": str(e)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    for dir_path in ["data/documents", "static"]:
        os.makedirs(dir_path, exist_ok=True)
    db.create_tables()
    logger.info("Application started", extra={"version": settings.app_version})

    async def _seed_in_background():
        try:
            from app.services.knowledge_seeder import seed_knowledge_base
            await asyncio.to_thread(seed_knowledge_base)
        except Exception as e:
            logger.warning("Knowledge base seeding failed — continuing without seed", extra={"error": str(e)})

    asyncio.create_task(_seed_in_background())

    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    logger.info(
        "Session cleanup task started",
        extra={"ttl_hours": settings.session_ttl_hours, "interval_minutes": settings.session_cleanup_interval_minutes},
    )

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Application shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Deep Research Agent — dual-mode RAG system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["RAG"])
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
    )
