import os 
import hashlib

from typing import List, Dict, Optional
from pathlib import Path 

import pypdf
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter 

from config import settings 
from config.logging_config import get_logger 

logger = get_logger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = settings.chunk_size, 
            chunk_overlap = settings.chunk_overlap,
            length_function = len,
        )

    def extract_text(self, file_path: str, file_type: str) -> str:
        logger.info(
            "Extracting text from documents",
            extra={
                "file_path": file_path, 
                "file_type": file_type
            }
        )
        try:
            if file_type == "pdf":
                text = self._extract_pdf(file_path)
            elif file_type == "txt":
                text = self._extract_txt(file_path)
            elif file_type in ["docx", "doc"]:
                text = self._extract_docx(file_path)
            else:
                logger.error(
                    "Unsupported file type", 
                    extra={"file_path": file_path, "file_type": file_type}
                )
                raise ValueError(f"Unsupported File Type: {file_type}")

            logger.info(
                "Text extraction completed",
                extra={
                    "file_path": file_path,
                    "file_type": file_type,
                    "text_length": len(text)
                }
            )
            return text 
        except Exception as e: 
            logger.error(
                "Error extracting text from document",
                extra={
                    "file_path": file_path,
                    "file_type": file_type,
                    "error": str(e)
                },
                exc_info=True
            )
            raise Exception(f"Error extracting text from {file_path}: {str(e)}")

    def _extract_pdf(self, file_path: str) -> str:
        logger.debug("Extracting from pdf", extra={"file_path": file_path})
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            logger.debug(
                "PDF opened successfully",
                extra={"file_path": file_path, "total_pages": total_pages}
            )
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                text += page_text + "\n"
                logger.debug(
                    "Extracted page text",
                    extra={
                        "file_path": file_path,
                        "page_number": page_num,
                        "page_text_length": len(page_text)
                    }
                )
        logger.info(
            "PDF text extraction completed",
            extra={"file_path": file_path, "total_text_length": len(text)}
        )
        return text 
    
    def _extract_txt(self, file_path: str) -> str:
        logger.debug("Extracting from TXT", extra={"file_path": file_path})
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        logger.debug(
            "TXT text extraction completed",
            extra={"file_path": file_path, "text_length": len(text)}
        )
        return text 

    def _extract_docx(self, file_path: str) -> str:
        logger.debug("Extracting text from DOCX file", extra={"file_path": file_path})
        doc = DocxDocument(file_path)
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        text = "\n".join(paragraphs)
        logger.debug(
            "DOCX text extraction completed",
            extra={
                "file_path": file_path,
                "text_length": len(text),
                "paragraph_count": len(paragraphs)
            }
        )
        return text 
    
    def chunk_text(self, text: str, document_id: int) -> List[Dict]:
        logger.info(
            "chunking text", 
            extra={
                "document_id": document_id, 
                "text_length": len(text), 
                "chunk_size": settings.chunk_size, 
                "chunk_overlap": settings.chunk_overlap
            }
        )
        chunks_data = []
        chunks = self.text_splitter.split_text(text)
        logger.debug(
            "Text Split into chunks",
            extra={
                "document_id": document_id, 
                "total_chunks": len(chunks)
            }
        )
        for idx, chunk_text in enumerate(chunks):
            start_char = text.find(chunk_text)
            end_char = start_char + len(chunk_text) if start_char != -1 else -1
            chunk_data={
                "document_id": document_id, 
                "chunk_index": idx, 
                "text": chunk_text, 
                "start_char": start_char if start_char != -1 else 0,
                "end_char": end_char if end_char != -1 else len(chunk_text),
                "metadata": {
                    "chunk_size": len(chunk_text),
                    "token_estimate": len(chunk_text.split()) * 1.3,  # Rough estimate
                }
            }
            chunks_data.append(chunk_data)

        logger.info(
            "Text chunking completed",
            extra={
                "document_id": document_id,
                "total_chunks": len(chunks_data)
            }
        )

        return chunks_data

    def generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate a simple summary of the document (first n chars)."""
        logger.debug(
            "Generating document summary",
            extra={"text_length": len(text), "max_length": max_length}
        )
        if len(text) <= max_length:
            summary = text
        else:
            summary = text[:max_length] + "..."
        logger.debug(
            "Summary generated",
            extra={"summary_length": len(summary), "original_length": len(text)}
        )
        return summary

    def get_file_hash(self, file_path: str) -> str:
        logger.debug("Generating file hash", extra={"file_path": file_path})
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()
        logger.debug("File hash generated",
            extra={"file_path": file_path, "hash": file_hash})
        return file_hash
