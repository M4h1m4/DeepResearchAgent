from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker, Session 
from typing import Generator 

from config import settings 
from config.logging_config import get_logger
from app.models import Base

logger = get_logger(__name__)

class Database:
    def __init__(self):
        logger.info(
            "Initializing the database connection", 
            extra={"database_url": settings.database_url.split("@")[-1] if "@" in settings.database_url
            else settings.database_url  }
        )
        self.engine = create_engine( #creates a database connection pool
            settings.database_url, 
            connect_args = {"check_same_thread": False} if "sqlite" in settings.database_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine) 
        """sessionmaker creates a factory that generates sessions
        Each session represents a database transaction
        sessions manage database transaction. 
        autocommit and autoflush are set to false to give manual control over transactions 
        """
        logger.info("Database engine and session factory created")

    def create_tables(self):
        """Create all database tables."""
        logger.info("Creating database tables")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables are created successfully")

    def get_session(self) -> Generator[Session, None, None]:
        """Get database session."""
        session = self.SessionLocal()
        #SessionLocal creates a session in local
        try:
            yield session 
        finally:
            session.close()

db = Database()

def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get database session."""
    yield from db.get_session()
