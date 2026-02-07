import os 
import hashlib

from typing import List, Dict, Optional, Union
from pathlib import Path
from io import BytesIO 
from enum import Enum

import pypdf
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter 

from config import settings 
from config.logging_config import get_logger 

logger = get_logger(__name__)


class FileType(str, Enum):
    """Supported file types for document processing."""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    DOC = "doc"

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = settings.chunk_size, 
            chunk_overlap = settings.chunk_overlap,
            length_function = len,
        )

    def extract_text(self, file_path: str, file_type: FileType) -> str:
        logger.info(
            "Extracting text from documents",
            extra={
                "file_path": file_path, 
                "file_type": file_type
            }
        )
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            error_msg = f"File not found: {file_path}"
            logger.error("File not found", extra={"file_path": file_path})
            raise FileNotFoundError(error_msg)

        if not file_path_obj.is_file():
            error_msg = f"Path is not a file: {file_path}"
            logger.error("Path is not a file", extra={"file_path": file_path})
            raise ValueError(error_msg)
        
        # Check file permissions (readable)
        if not os.access(file_path, os.R_OK):
            error_msg = f"File is not readable: {file_path}"
            logger.error("File is not readable", extra={"file_path": file_path})
            raise PermissionError(error_msg)

        try:
            if file_type == FileType.PDF:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                text = self._extract_pdf(file_data)
            elif file_type == FileType.TXT:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_data = f.read()
                text = self._extract_txt(file_data)
            elif file_type in [FileType.DOCX, FileType.DOC]:
                with open(file_path, "rb") as f:
                    file_data = BytesIO(f.read())
                text = self._extract_docx(file_data)
            else:
                logger.error(
                    "Unsupported file type", 
                    extra={"file_path": file_path, "file_type": file_type.value if isinstance(file_type, FileType) else file_type}
                )
                raise ValueError(f"Unsupported File Type: {file_type}")

            logger.info(
                "Text extraction completed",
                extra={
                    "file_path": file_path,
                    "file_type": file_type.value,
                    "text_length": len(text)
                }
            )
            return text 
        except FileNotFoundError: 
            raise 
        except ValueError:
            raise 
        except PermissionError:
            raise
        except Exception as e: 
            logger.error(
                "Error extracting text from document",
                extra={
                    "file_path": file_path,
                    "file_type": file_type.value if isinstance(file_type, FileType) else str(file_type),
                    "error": str(e)
                },
                exc_info=True
            )
            raise Exception(f"Error extracting text from {file_path}: {str(e)}")

    def _extract_pdf(self, file_data: bytes) -> str:
        
        logger.debug("Extracting from pdf", extra={"data_size": len(file_data)})
        text = ""
        failed_pages = 0
        
        try: 
            pdf_stream = BytesIO(file_data)
            pdf_reader = pypdf.PdfReader(pdf_stream)
            total_pages = len(pdf_reader.pages)
            logger.debug(
                "PDF opened successfully",
                extra={"total_pages": total_pages, "data_size": len(file_data)}
            )
            
            # Process each page, handling individual page failures
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    logger.debug(
                        "Extracted page text",
                        extra={
                            "page_number": page_num,
                            "page_text_length": len(page_text) if page_text else 0
                        }
                    )
                except Exception as page_error:
                    failed_pages += 1
                    logger.warning(
                        "Failed to extract text from PDF page",
                        extra={
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "error": str(page_error),
                            "error_type": type(page_error).__name__
                        },
                        exc_info=True
                    )
                    # Continue processing other pages
                    continue
            
            # Log extraction summary with metrics
            logger.info(
                "PDF text extraction completed",
                extra={
                    "total_pages": total_pages,
                    "successful_pages": total_pages - failed_pages,
                    "failed_pages": failed_pages,
                    "total_text_length": len(text),
                    "extraction_success_rate": (total_pages - failed_pages) / total_pages if total_pages > 0 else 0.0
                }
            )
            
            return text 
            
        except pypdf.errors.PdfReadError as e:
            # PDF is corrupted or cannot be read
            logger.error(
                "PDF read error - PDF may be corrupted or encrypted",
                extra={"data_size": len(file_data), "error": str(e), "error_type": "PdfReadError"},
                exc_info=True
            )
            raise Exception(f"Error reading PDF: {str(e)}")
            
        except Exception as e:
            logger.error(
                "Error extracting PDF text",
                extra={"data_size": len(file_data), "error": str(e), "error_type": type(e).__name__},
                exc_info=True
            )
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    def _extract_txt(self, file_data: str) -> str:
        """
        Extract text from TXT file data.
        
        Args:
            file_data: Text content (already decoded string)
            
        Returns:
            Text content
        """
        logger.debug("Extracting from TXT", extra={"text_length": len(file_data)})
        logger.debug(
            "TXT text extraction completed",
            extra={"text_length": len(file_data)}
        )
        return file_data

    def _extract_docx(self, file_data: Union[BytesIO, str]) -> str:
        logger.debug("Extracting text from DOCX file")
        try:
            doc = DocxDocument(file_data)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs]
            text = "\n".join(paragraphs)
            logger.debug(
                "DOCX text extraction completed",
                extra={
                    "text_length": len(text),
                    "paragraph_count": len(paragraphs)
                }
            )
            return text 
        except Exception as e:
            logger.error(
                "Error extracting DOCX text",
                extra={"error": str(e)},
                exc_info=True
            )
            raise Exception(f"Error extracting DOCX: {str(e)}")
    
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
