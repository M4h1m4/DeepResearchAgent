from typing import List, Dict, Any, Optional 
from dataclasses import dataclass 
from datasets import load_dataset
from beir import util, datasets
import json 
import os 
import random
from sqlalchemy.orm import Session

from config.logging_config import get_logger 

logger = get_logger(__name__)

@dataclass 
class EvaluationSample:
    query: str
    ground_truth_answer: Optional[str] = None 
    ground_truth_chunks: Optional[List[Dict[str, Any]]] = None 
    expected_sources: Optional[List[str]] = None 
    metadata: Optional[Dict[str, Any]] = None 

@dataclass 
class EvaluationDataset:
    name: str
    samples: List[EvaluationSample]
    description: Optional[str] = None 

    def __len__(self) -> int:
        return len(self.samples)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "num_samples": len(self.samples),
            "samples": [
                {
                    "query": s.query,
                    "ground_truth_answer": s.ground_truth_answer,
                    "ground_truth_chunks": s.ground_truth_chunks,
                    "expected_sources": s.expected_sources,
                    "metadata": s.metadata
                }
                for s in self.samples
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        samples = [
            EvaluationSample(
                query=s["query"],
                ground_truth_answer=s.get("ground_truth_answer"),
                ground_truth_chunks=s.get("ground_truth_chunks"),
                expected_sources=s.get("expected_sources"),
                metadata=s.get("metadata")
            )
            for s in data["samples"]
        ]
        return cls(
            name=data["name"],
            samples=samples,
            description=data.get("description")
        )

    def save(self, filepath: str):
        #Save the dataset into JSON file 
        logger.info(f"Saving the dataset to {filepath}", extra={"dataset_name": self.name})
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Dataset Saved Successfully")

    @classmethod
    def load(cls, filepath: str) -> "EvaluationDataset":
        #load dataset from JSON file 
        logger.info(f"Loading dataset from {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)
        dataset = cls.from_dict(data)
        logger.info("Dataset Loaded successfully", extra={"num_samples": len(dataset)})
        return dataset

class DatasetManager: 
    def __init__(self, datasets_dir: str = "data/eval_datasets"):
        """
        Initialize DatasetManager.
        
        Args:
            datasets_dir: Directory to cache loaded benchmark datasets
        """
        self.datasets_dir = datasets_dir
        os.makedirs(datasets_dir, exist_ok=True)
        logger.info("DatasetManager initialized", extra={"datasets_dir": datasets_dir})
    
    def load_ragas_dataset(
        self, 
        dataset_name: str = "fiqa",
        split: str = "ragas_eval",
        max_samples: Optional[int] = None
    ) -> Optional[EvaluationDataset]:
        try:
            from datasets import load_dataset
            
            logger.info(
                f"Loading RAGAS dataset: {dataset_name}", 
                extra={"split": split, "max_samples": max_samples}
            )
            # Load from HuggingFace datasets 
            hf_dataset = load_dataset(f"explodinggradients/{dataset_name}", split)
            
            # Handle both Dataset and DatasetDict
            # If it's a DatasetDict, extract the split
            if hasattr(hf_dataset, 'keys'):
                # It's a DatasetDict - get the split
                if split in hf_dataset:
                    hf_dataset = hf_dataset[split]
                elif 'ragas_eval' in hf_dataset:
                    hf_dataset = hf_dataset['ragas_eval']
                    logger.warning(f"Split '{split}' not found, using 'ragas_eval' instead")
                else:
                    # Use the first available split
                    first_split = list(hf_dataset.keys())[0]
                    hf_dataset = hf_dataset[first_split]
                    logger.warning(f"Using first available split: '{first_split}'")

            if max_samples:
                hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

            samples = [] 
            for item in hf_dataset:
                question = item.get("question") or item.get("query", "")
                ground_truth = item.get("ground_truth")
                if isinstance(ground_truth, list):
                    ground_truth = ground_truth[0] if ground_truth else None
                
                contexts = item.get("contexts", [])
                if not contexts:
                    contexts = item.get("context", [])
                if isinstance(contexts, str):
                    contexts = [contexts]

                sample = EvaluationSample(
                    query=question,
                    ground_truth_answer=ground_truth,
                    ground_truth_chunks=[{"text": ctx, "id": i} for i, ctx in enumerate(contexts)],
                    expected_sources=item.get("source", []),
                    metadata={
                        "source": "ragas",
                        "dataset": dataset_name,
                        "split": split,
                        "original_id": item.get("id")
                    }
                )
                samples.append(sample)
            
            dataset = EvaluationDataset(
                name=f"ragas_{dataset_name}",
                samples=samples,
                description=f"RAGAS benchmark dataset: {dataset_name} ({len(samples)} samples)"
            )
            
            logger.info(
                f"RAGAS dataset loaded successfully",
                extra={"dataset": dataset_name, "num_samples": len(samples)}
            )
            return dataset
            
        except Exception as e:
            error_msg = f"Failed to load RAGAS dataset {dataset_name}: {str(e)}"
            logger.error(
                f"Failed to load RAGAS dataset {dataset_name}",
                extra={"error": str(e), "dataset_name": dataset_name, "error_type": type(e).__name__},
                exc_info=True
            )
            # Re-raise the exception so the caller can see the actual error
            raise ValueError(error_msg) from e

    def load_beir_dataset(
        self, 
        dataset_name: str = "fiqa",
        max_samples: Optional[int] = 100
    ) -> Optional[EvaluationDataset]:
        try:
            logger.info(
                f"Loading BEIR dataset: {dataset_name}",
                extra={"max_samples": max_samples}
            )
            
            # Download and load dataset
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            data_path = util.download_and_unzip(url, "data/beir_datasets")
            
            # Load corpus, queries, and qrels
            corpus, queries, qrels = datasets.load_beir_data(data_path)
            
            # Convert to our format
            samples = []
            sample_count = 0
            
            for query_id, query_text in queries.items():
                if max_samples and sample_count >= max_samples:
                    break
                
                # Get relevant document IDs for this query
                relevant_doc_ids = qrels.get(query_id, {})
                relevant_chunks = []
                
                for doc_id, relevance_score in relevant_doc_ids.items():
                    if doc_id in corpus:
                        doc = corpus[doc_id]
                        relevant_chunks.append({
                            "id": doc_id,
                            "text": doc.get("text", ""),
                            "title": doc.get("title", ""),
                            "relevance_score": relevance_score
                        })
                
                if relevant_chunks:
                    sample = EvaluationSample(
                        query=query_text,
                        ground_truth_chunks=relevant_chunks,
                        metadata={
                            "source": "beir",
                            "dataset": dataset_name,
                            "query_id": query_id,
                            "num_relevant_docs": len(relevant_chunks)
                        }
                    )
                    samples.append(sample)
                    sample_count += 1
            
            dataset = EvaluationDataset(
                name=f"beir_{dataset_name}",
                samples=samples,
                description=f"BEIR benchmark dataset: {dataset_name} ({len(samples)} samples)"
            )
            
            logger.info(
                f"BEIR dataset loaded successfully",
                extra={"dataset": dataset_name, "num_samples": len(samples)}
            )
            return dataset
            
        except Exception as e:
            logger.error(
                f"Failed to load BEIR dataset {dataset_name}",
                extra={"error": str(e), "dataset_name": dataset_name},
                exc_info=True
            )
            return None

    def list_available_benchmarks(self) -> Dict[str, List[str]]:
        """
        List available benchmark datasets.
        
        Returns:
            Dictionary with benchmark categories and available datasets
        """
        return {
            "ragas": [
                "fiqa",           # Financial Q&A - Good for Fast RAG (single-hop)
                "pubmed_qa",      # Medical Q&A - Good for domain testing, Deep Research
                "wiki_qa",        # Wikipedia Q&A - Good for Fast RAG (general knowledge)
                "hotpot_qa",      # Multi-hop Q&A - BEST for Deep Research mode
                "msmarco"          # Microsoft MRC - Good for general RAG testing
            ],
            "beir": [
                "fiqa",           # Financial Q&A (different format than RAGAS fiqa)
                "scifact",        # Scientific fact verification - Good for Deep Research
                "nfcorpus",       # Medical IR - Good for domain testing
                "scidocs",        # Scientific documents - Good for Deep Research
                "trec-covid",     # COVID-19 IR - Domain-specific
                "quora",          # Duplicate questions - Not ideal for RAG evaluation
                "dbpedia",        # Entity retrieval - Good for Fast RAG
                "fever",          # Fact verification - Good for Deep Research
                "climate-fever",  # Climate facts - Domain-specific
                "hotpotqa"        # Multi-hop Q&A - BEST for Deep Research mode
            ],
            "other": [
                "squad",          # Stanford Question Answering Dataset
                "natural_questions",  # Google Natural Questions
                "ms_marco"        # Microsoft Machine Reading Comprehension
            ]
        }
    
    def list_datasets(self) -> List[str]: #list available datasetes
        datasets = [
            f.replace(".json", "")
            for f in os.listdir(self.datasets_dir)
            if f.endswith(".json")
        ]
        return datasets
    
    def get_dataset(self, name: str) -> Optional[EvaluationDataset]:# get datasets by name
        filepath = os.path.join(self.datasets_dir, f"{name}.json")
        if os.path.exists(filepath):
            return EvaluationDataset.load(filepath)
        return None
    
    def create_dataset_from_database(
        self,
        db_session: Session,
        max_samples: Optional[int] = 20,
        min_chunk_length: int = 100
    ) -> Optional[EvaluationDataset]:
        """
        Create an evaluation dataset from existing documents in the database.
        This ensures ground truth matches your actual database content.
        
        Args:
            db_session: SQLAlchemy database session
            max_samples: Maximum number of samples to create
            min_chunk_length: Minimum chunk text length to include
            
        Returns:
            EvaluationDataset with samples based on existing chunks
        """
        try:
            from app.models import Chunk, Document
            
            logger.info(
                "Creating evaluation dataset from database",
                extra={"max_samples": max_samples, "min_chunk_length": min_chunk_length}
            )
            
            # Query chunks from database
            chunks = db_session.query(Chunk).join(Document).filter(
                Chunk.text.isnot(None),
                Chunk.text != "",
                Chunk.text.length() >= min_chunk_length
            ).all()
            
            if not chunks:
                logger.warning("No chunks found in database to create evaluation dataset")
                return None
            
            # Randomly sample chunks
            if len(chunks) > max_samples:
                chunks = random.sample(chunks, max_samples)
            else:
                chunks = chunks[:max_samples]
            
            samples = []
            for chunk in chunks:
                # Extract a query from the chunk (use first sentence or key phrase)
                chunk_text = chunk.text.strip()
                
                # Simple query generation: use first sentence or first 50 words
                if '.' in chunk_text:
                    query = chunk_text.split('.')[0].strip()
                    if len(query) < 20:  # If first sentence is too short, use first 50 words
                        words = chunk_text.split()[:50]
                        query = ' '.join(words)
                        if len(query) > 200:
                            query = query[:200] + "..."
                else:
                    words = chunk_text.split()[:50]
                    query = ' '.join(words)
                    if len(query) > 200:
                        query = query[:200] + "..."
                
                # Create ground truth chunk entry
                ground_truth_chunk = {
                    "id": chunk.id,
                    "text": chunk_text,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id
                }
                
                # Get document info for metadata
                doc = db_session.query(Document).filter(Document.id == chunk.document_id).first()
                doc_title = doc.title if doc else f"Document {chunk.document_id}"
                
                sample = EvaluationSample(
                    query=query,
                    ground_truth_chunks=[ground_truth_chunk],
                    ground_truth_answer=None,  # Can be generated later if needed
                    expected_sources=[doc_title] if doc else [],
                    metadata={
                        "source": "database",
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "document_title": doc_title,
                        "chunk_index": chunk.chunk_index
                    }
                )
                samples.append(sample)
            
            dataset = EvaluationDataset(
                name="custom_database",
                samples=samples,
                description=f"Custom evaluation dataset from database ({len(samples)} samples)"
            )
            
            logger.info(
                "Dataset created from database successfully",
                extra={"num_samples": len(samples), "num_chunks_available": len(chunks)}
            )
            return dataset
            
        except Exception as e:
            logger.error(
                "Failed to create dataset from database",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True
            )
            return None