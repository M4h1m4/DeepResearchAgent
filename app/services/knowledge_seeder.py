"""
Seeds Pinecone with Wikipedia passages from a HuggingFace dataset so that
users can query the app immediately without uploading any documents.

Runs once at startup; skips if the index already has vectors.
Dataset default: rajpurkar/squad_v2 — SQuAD v2 Wikipedia passages (diverse topics).
"""
import uuid
from typing import Optional

from langchain_openai import OpenAIEmbeddings

from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


def seed_knowledge_base() -> None:
    if not settings.seed_on_startup:
        logger.info("Knowledge base seeding disabled via SEED_ON_STARTUP=false")
        return

    if not settings.openai_api_key:
        logger.warning(
            "Skipping knowledge base seeding: OPENAI_API_KEY not set in environment. "
            "Set it in .env to auto-populate the knowledge base on startup."
        )
        return

    # Lazy import to avoid loading heavy deps at module level
    from app.services.vector_service import VectorStore
    from app.database import db
    from app.models import Document

    vector_store = VectorStore()

    if vector_store.get_count() > 0:
        logger.info(
            "Knowledge base already seeded — skipping",
            extra={"vector_count": vector_store.get_count()},
        )
        return

    logger.info(
        "Seeding knowledge base from HuggingFace dataset",
        extra={"dataset": settings.hf_dataset_name, "max_passages": settings.seed_max_passages},
    )

    passages = _load_passages()
    if not passages:
        logger.error("No passages loaded from dataset — seeding aborted")
        return

    # Create a single Document record in the relational DB so RAG source lookups work
    session_gen = db.get_session()
    session = next(session_gen)
    try:
        hf_doc = Document(
            title="HuggingFace Knowledge Base (SQuAD Wikipedia)",
            source=f"huggingface/{settings.hf_dataset_name}",
            file_type="dataset",
            file_size=0,
            summary=(
                "Pre-seeded knowledge base built from HuggingFace SQuAD dataset. "
                "Contains diverse Wikipedia passages covering history, science, culture, and more."
            ),
            extra_metadata={"seeded": True, "dataset": settings.hf_dataset_name},
        )
        session.add(hf_doc)
        session.commit()
        session.refresh(hf_doc)
        hf_doc_id = hf_doc.id
    finally:
        session.close()

    logger.info("HuggingFace Document record created", extra={"document_id": hf_doc_id})

    embedder = OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)
    batch_size = 100
    total_upserted = 0

    for i in range(0, len(passages), batch_size):
        batch = passages[i : i + batch_size]
        texts = [p["text"] for p in batch]

        try:
            batch_embeddings = embedder.embed_documents(texts)
        except Exception as e:
            logger.error(
                "Embedding batch failed — skipping",
                extra={"batch_start": i, "error": str(e)},
            )
            continue

        chunks = [
            {
                "id": None,
                "text": p["text"],
                "document_id": hf_doc_id,
                "chunk_index": i + j,
                "vector_id": str(uuid.uuid4()),
                "metadata": {
                    "title": p["title"],
                    "source": p["source"],
                    "seeded": True,
                },
            }
            for j, p in enumerate(batch)
        ]

        try:
            vector_store.add_documents(chunks, embeddings=batch_embeddings)
            total_upserted += len(chunks)
        except Exception as e:
            logger.error("Pinecone upsert failed", extra={"batch_start": i, "error": str(e)})
            continue

        if (i // batch_size) % 5 == 0:
            logger.info(
                "Seeding progress",
                extra={"upserted": total_upserted, "total": len(passages)},
            )

    logger.info(
        "Knowledge base seeding complete",
        extra={"passages_seeded": total_upserted, "document_id": hf_doc_id},
    )


def _load_passages() -> list:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' package not installed. Run: pip install datasets")
        return []

    try:
        dataset = load_dataset(
            settings.hf_dataset_name,
            split="train",
            streaming=True,
        )
    except Exception as e:
        logger.error("Failed to load HuggingFace dataset", extra={"error": str(e)})
        return []

    seen_hashes: set = set()
    passages = []

    for row in dataset:
        context = (row.get("context") or "").strip()
        title = (row.get("title") or "Unknown Topic").strip()

        if len(context) < 100:
            continue

        ctx_hash = hash(context)
        if ctx_hash in seen_hashes:
            continue
        seen_hashes.add(ctx_hash)

        passages.append({
            "text": context,
            "title": title,
            "source": f"huggingface/{settings.hf_dataset_name}",
        })

        if len(passages) >= settings.seed_max_passages:
            break

    logger.info("Passages loaded from dataset", extra={"count": len(passages)})
    return passages
