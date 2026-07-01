"""
Session lifecycle management.

Every upload or query that carries a session_id calls touch() to record/refresh
the session's last_accessed_at timestamp. A background loop in main.py calls
delete_expired() periodically to purge idle sessions from Pinecone and the DB.
"""

from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session

from app.models import SessionRecord
from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

_vector_store = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        from app.services.vector_service import VectorStore
        _vector_store = VectorStore()
    return _vector_store


class SessionService:

    def touch(self, db: Session, session_id: str) -> None:
        """Create the session record if new, or update last_accessed_at if it exists."""
        record = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if record:
            record.last_accessed_at = datetime.utcnow()
        else:
            db.add(SessionRecord(session_id=session_id))
        db.commit()

    def delete_expired(self, db: Session) -> List[str]:
        """
        Find sessions idle for longer than session_ttl_hours, delete their
        Pinecone namespace, and remove the DB record.
        Returns the list of deleted session_ids.
        """
        cutoff = datetime.utcnow() - timedelta(hours=settings.session_ttl_hours)
        expired = (
            db.query(SessionRecord)
            .filter(SessionRecord.last_accessed_at < cutoff)
            .all()
        )

        if not expired:
            return []

        deleted = []
        vector_store = _get_vector_store()

        for record in expired:
            try:
                vector_store.delete_namespace(record.session_id)
                db.delete(record)
                deleted.append(record.session_id)
                logger.info("Expired session deleted", extra={"session_id": record.session_id})
            except Exception as e:
                logger.error(
                    "Failed to delete expired session",
                    extra={"session_id": record.session_id, "error": str(e)},
                )

        db.commit()
        if deleted:
            logger.info("Session cleanup complete", extra={"deleted_count": len(deleted)})
        return deleted

    def delete(self, db: Session, session_id: str) -> None:
        """Immediately delete a single session (called from DELETE /sessions/{id})."""
        _get_vector_store().delete_namespace(session_id)

        record = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if record:
            db.delete(record)
            db.commit()
        logger.info("Session deleted on request", extra={"session_id": session_id})
