from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship 
from sqlalchemy.ext.declarative import declarative_base 
from datetime import datetime 

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False, index=True)
    source = Column(String(500))
    file_type = Column(String(50)) #extension
    file_size = Column(Integer)  # Size in bytes
    summary = Column(Text)  # Document summary
    extra_metadata = Column(JSON)  # Additional metadata (renamed from 'metadata' - reserved by SQLAlchemy)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}')>"


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)  # Order in document
    text = Column(Text, nullable=False)  # Chunk text content
    start_char = Column(Integer)  # Start character position in original doc
    end_char = Column(Integer)  # End character position in original doc
    chunk_metadata = Column(JSON)  # Chunk metadata (renamed from 'metadata' - reserved by SQLAlchemy)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class SessionRecord(Base):
    __tablename__ = "sessions"

    session_id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<SessionRecord(session_id='{self.session_id}', last_accessed='{self.last_accessed_at}')>"


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    response = Column(Text)
    retrieved_chunks = Column(JSON) #list of chunk ids retrievd
    response_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<QueryLog(id={self.id}, query='{self.query[:50]}...')>"