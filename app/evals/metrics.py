from typing import List, Dict, Any, Optional 
import numpy as np 
from dataclasses import dataclass 

from ragas import evaluate 
from ragas.dataset_schema import EvaluationDataset  # Dataset format for RAGAS

from config.logging_config import get_logger

logger = get_logger(__name__)

# Import metrics - handle different RAGAS versions gracefully
try:
    # Try standard RAGAS import (v0.1+)
    from ragas.metrics import (
        faithfulness, 
        answer_relevance, 
        context_precision, 
        context_recall, 
        answer_correctness, 
        answer_similarity
    )
except (ImportError, AttributeError) as e:
    # If import fails, set to None and handle gracefully in code
    logger.warning(
        "Could not import some RAGAS metrics - some evaluation features may be limited",
        extra={"error": str(e)}
    )
    faithfulness = None
    answer_relevance = None
    context_precision = None
    context_recall = None
    answer_correctness = None
    answer_similarity = None

@dataclass 
class RetrievalMetrics:
    precision_at_k: float # of top k results how many are relevant 
    recall_at_k: float # of all relevent documents how many did we find 
    mrr: float #Mean Reciprocal Rank: position of first relevant document 
    ndcg: float #Normalized discounted cumulative gain 
    #higher score for higher positioning 

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg
        }

    
@dataclass 
class GenerationMetrics:
    faithfulness: float
    answer_relevance: float
    context_precision: float 
    context_recall: float 
    answer_correctness: Optional[float] = None 
    answer_similarity: Optional[float] = None 


    def to_dict(self) -> Dict[str, float]:
        result = {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall
        }
        if self.answer_correctness is not None:
            result["answer_correctness"] = self.answer_correctness
        if self.answer_similarity is not None:
            result["answer_similarity"] = self.answer_similarity
        return result


@dataclass
class LatencyMetrics:
    total_time_ms: float 
    retrieval_time_ms: float 
    generation_time_ms: float 
    embedding_time_ms: float 

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_time_ms": self.total_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "embedding_time_ms": self.embedding_time_ms
        }

class RAGEvaluator:
    def __init__(self):
        logger.info("Initializing RAG Evaluator")

    def evaluate_retrieval(self, # to evaluate the retrieval quality
        queries: List[str], 
        retrieved_chunks: List[List[Dict[str, Any]]],
        ground_truth_chunks: List[List[Dict[str, Any]]], 
        k: int = 5 
    ) -> RetrievalMetrics:

        logger.info(
            "Evaluating retrieval quality",
            extra={"num_queries": len(queries), "k": k}
        )
        precisions = []
        recalls = []
        reciprocal_ranks = []
        ndcgs = []

        for query, retrieved, ground_truth in zip(queries, retrieved_chunks, ground_truth_chunks):
            # Extract ground truth text content (benchmark datasets provide text, not database IDs)
            gt_texts = []
            for chunk in ground_truth:
                if isinstance(chunk, dict):
                    # Get text content from ground truth chunk
                    text = chunk.get("text") or chunk.get("content") or str(chunk)
                    if text and text.strip():
                        gt_texts.append(text.lower().strip())
                elif isinstance(chunk, str) and chunk.strip():
                    gt_texts.append(chunk.lower().strip())
            
            # Extract retrieved chunk text content
            retrieved_texts = []
            for chunk in retrieved[:k]:
                if isinstance(chunk, dict):
                    # Get text content from retrieved chunk
                    text = chunk.get("text") or chunk.get("content") or chunk.get("chunk_text") or str(chunk)
                    if text and text.strip():
                        retrieved_texts.append(text.lower().strip())
                elif isinstance(chunk, str) and chunk.strip():
                    retrieved_texts.append(chunk.lower().strip())
            
            # If no ground truth or no retrieved chunks, skip this query
            if not gt_texts or not retrieved_texts:
                precisions.append(0.0)
                recalls.append(0.0)
                reciprocal_ranks.append(0.0)
                ndcgs.append(0.0)
                continue
            
            # Match retrieved chunks to ground truth by text similarity (fuzzy matching)
            # For exact match, we check if retrieved text contains or is contained in ground truth text
            relevant_retrieved = 0
            matched_gt_indices = set()
            for ret_text in retrieved_texts:
                for i, gt_text in enumerate(gt_texts):
                    if i in matched_gt_indices:
                        continue
                    # Check for overlap (simple text matching - can be improved with embeddings)
                    # Match if texts are similar (contain each other or have significant overlap)
                    ret_words = set(ret_text.split())
                    gt_words = set(gt_text.split())
                    overlap_ratio = len(ret_words & gt_words) / max(len(gt_words), 1) if gt_words else 0.0
                    
                    if (gt_text in ret_text or ret_text in gt_text or overlap_ratio > 0.3):
                        relevant_retrieved += 1
                        matched_gt_indices.add(i)
                        break
            precision = relevant_retrieved/k if k > 0 else 0.0 
            precisions.append(precision)

            recall = relevant_retrieved / len(gt_texts) if len(gt_texts) > 0 else 0.0 
            recalls.append(recall)

            # Calculate MRR - find first relevant retrieved chunk
            rr = 0.0 
            for rank, ret_text in enumerate(retrieved_texts, 1):
                # Check if this retrieved text matches any ground truth
                for gt_text in gt_texts:
                    if (gt_text in ret_text or ret_text in gt_text or 
                        len(set(gt_text.split()) & set(ret_text.split())) / max(len(gt_text.split()), 1) > 0.5):
                        rr = 1.0 / rank
                        break
                if rr > 0:
                    break
            reciprocal_ranks.append(rr)
        
            #DCG = sum of(relevance score/log2(position+1)) for each position 
            # IDCG = ideal DCG (if all relevant docs were at top)
            # NDCG = DCG / IDCG (normalized to 0.0-1.0)
            dcg = 0.0
            for rank, ret_text in enumerate(retrieved_texts, 1):
                # Check if this retrieved text matches any ground truth
                is_relevant = False
                for gt_text in gt_texts:
                    if (gt_text in ret_text or ret_text in gt_text or 
                        len(set(gt_text.split()) & set(ret_text.split())) / max(len(gt_text.split()), 1) > 0.5):
                        is_relevant = True
                        break
                if is_relevant:
                    dcg += 1.0 / np.log2(rank + 2)
            
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(1, min(len(gt_texts), k) + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        metrics = RetrievalMetrics(
            precision_at_k = np.mean(precisions), 
            recall_at_k = np.mean(recalls),
            mrr=np.mean(reciprocal_ranks), 
            ndcg=np.mean(ndcgs)
        )

        logger.info(
            "Retrieval evaluation completed",
            extra=metrics.to_dict()
        )
        
        return metrics

    def evaluate_generation(
        self, 
        queries: List[str], 
        answers: List[str], 
        contexts: List[List[str]], 
        ground_truth: Optional[List[str]] = None
    ) -> GenerationMetrics:
        logger.info(
            "Evaluating generation quality",
            extra={"num_queries": len(queries)}
        )

        dataset = {
            "question": queries, 
            "answer": answers, 
            "context": [ctx for ctx in contexts]
        }

        if ground_truth:
            dataset["ground_truth"] = ground_truth

            metrics_to_use = [
                faithfulness,        # Is answer based on context? (no ground truth needed)
                answer_relevance,    # Does answer address question? (no ground truth needed)
                context_precision,   # Are chunks relevant? (no ground truth needed)
                context_recall,      # Did we get all relevant chunks? (no ground truth needed)
                answer_correctness,  # How accurate vs ground truth? (REQUIRES ground truth)
                answer_similarity    # How similar to ground truth? (REQUIRES ground truth)
            ]
        else:
            metrics_to_use = [
                faithfulness,      
                answer_relevance,  
                context_precision, 
                context_recall     
            ]
        
        try:
            result = evaluate(
                dataset=dataset, 
                metrics=metrics_to_use, 
                llm=None, 
                embeddings=None
            )

            metrics = GenerationMetrics(
                faithfulness=result.get("faithfulness", 0.0),
                answer_relevance=result.get("answer_relevance", 0.0),
                context_precision=result.get("context_precision", 0.0),
                context_recall=result.get("context_recall", 0.0),
                answer_correctness=result.get("answer_correctness") if ground_truth else None,
                answer_similarity=result.get("answer_similarity") if ground_truth else None
            )
            logger.info(
                "Generation Evaluation completed", 
                extra=metrics.to_dict()
            )
            return metrics 
        
        except Exception as e: 
            logger.error(
                "Error in generation evaluation",
                extra={"error": str(e)},
                exc_info=True
            )
            return GenerationMetrics(
                faithfulness=0.0,
                answer_relevance=0.0,
                context_precision=0.0,
                context_recall=0.0
            )
    
    def evaluate_latency(
        self, 
        latency_data: List[Dict[str, float]]
    ) -> LatencyMetrics: 
        logger.info("Evaluating latency metrics", extra={"num_samples": len(latency_data)})
        
        total_times = [d.get("total_time_ms", 0.0) for d in latency_data]
        retrieval_times = [d.get("retrieval_time_ms", 0.0) for d in latency_data]
        generation_times = [d.get("generation_time_ms", 0.0) for d in latency_data]
        embedding_times = [d.get("embedding_time_ms", 0.0) for d in latency_data]
        
        metrics = LatencyMetrics(
            total_time_ms=np.mean(total_times),
            retrieval_time_ms=np.mean(retrieval_times),
            generation_time_ms=np.mean(generation_times),
            embedding_time_ms=np.mean(embedding_times)
        )
        
        logger.info("Latency evaluation completed", extra=metrics.to_dict())
        
        return metrics
    




