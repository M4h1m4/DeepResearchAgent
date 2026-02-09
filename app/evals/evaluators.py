#Evaluation Orchestrator for FastRAG and DeepResearch

from typing import List, Dict, Any, Optional 
import time 

from app.evals.metrics import RAGEvaluator, RetrievalMetrics, GenerationMetrics, LatencyMetrics
from app.evals.datasets import EvaluationDataset, EvaluationSample
from app.services.rag_service import RAGService
from app.services.deep_research_service import DeepResearchService
from app.database import get_db
from config.logging_config import get_logger

logger = get_logger(__name__)

class FastRAGEvaluator:
    def __init__(self):
        self.rag_service = RAGService()
        self.metrics_evaluator = RAGEvaluator()
        logger.info("FastRAGEvaluator initialized")

    def evaluate(
        self, 
        dataset: EvaluationDataset, 
        db_Session
    ) -> Dict[str, Any]:
        logger.info(
            "Starting Fast RAG evaluation",
            extra={"dataset_name": dataset.name, "num_samples": len(dataset)}
        )

        queries = [sample.query for sample in dataset.samples]
        answers = []
        retrieved_chunks_list = []
        context_list = []
        latency_data = [] 

        for i, sample in enumerate(dataset.samples):
            start_time = time.time()

            result = self.rag_service.query(
                db=db_session, 
                query=sample.query, 
                top_k=5
            )
            total_time = (time.time() - start_time) * 1000 
            answers.append((result.get("answer", "")))
            retrieved_chunks = result.get("retrieved_chunks", [])
            retrieved_chunks_list.append(retrieved_chunks)
            
            contexts = [
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in retrieved_chunks
            ]
            contexts_list.append(contexts)
            
            latency_data.append({
                "total_time_ms": total_time,
                "retrieval_time_ms": result.get("response_time_ms", 0) * 0.3,  # Estimate
                "generation_time_ms": result.get("response_time_ms", 0) * 0.7,  # Estimate
                "embedding_time_ms": 0.0
            })
        
        # Evaluate retrieval
        ground_truth_chunks = [
            sample.ground_truth_chunks or []
            for sample in dataset.samples
        ]
        retrieval_metrics = self.metrics_evaluator.evaluate_retrieval(
            queries=queries,
            retrieved_chunks=retrieved_chunks_list,
            ground_truth_chunks=ground_truth_chunks,
            k=5
        )
        
        # Evaluate generation
        ground_truth_answers = [
            sample.ground_truth_answer
            for sample in dataset.samples
            if sample.ground_truth_answer
        ] or None
        
        generation_metrics = self.metrics_evaluator.evaluate_generation(
            queries=queries,
            answers=answers,
            contexts=contexts_list,
            ground_truth=ground_truth_answers
        )
        
        # Evaluate latency
        latency_metrics = self.metrics_evaluator.evaluate_latency(latency_data)
        
        results = {
            "dataset_name": dataset.name,
            "num_samples": len(dataset),
            "retrieval_metrics": retrieval_metrics.to_dict(),
            "generation_metrics": generation_metrics.to_dict(),
            "latency_metrics": latency_metrics.to_dict(),
            "samples": [
                {
                    "query": q,
                    "answer": a,
                    "retrieved_chunks_count": len(rc)
                }
                for q, a, rc in zip(queries, answers, retrieved_chunks_list)
            ]
        }
        
        logger.info("Fast RAG evaluation completed", extra={"results": results})
        
        return results
    
class DeepResearchEvaluator:
    logger.info(
            "Starting Deep Research evaluation",
            extra={"dataset_name": dataset.name, "num_samples": len(dataset)}
        )
        
        queries = [sample.query for sample in dataset.samples]
        answers = []
        all_findings = []
        latency_data = []
        
        # Run queries
        for i, sample in enumerate(dataset.samples):
            start_time = time.time()
            
            # Execute deep research
            result = self.deep_research_service.research(sample.query)
            
            total_time = (time.time() - start_time) * 1000
            
            answers.append(result.get("answer", ""))
            
            # Collect all findings
            findings = result.get("findings", [])
            all_findings.append(findings)
            
            latency_data.append({
                "total_time_ms": total_time,
                "retrieval_time_ms": total_time * 0.4,  # Estimate
                "generation_time_ms": total_time * 0.6,  # Estimate
                "embedding_time_ms": 0.0
            })
        
        # Extract contexts from findings
        contexts_list = []
        retrieved_chunks_list = []
        for findings in all_findings:
            contexts = []
            chunks = []
            for finding in findings:
                contexts.append(finding.get("answer", ""))
                chunks.extend(finding.get("retrieved_chunks", []))
            contexts_list.append(contexts)
            retrieved_chunks_list.append(chunks)
        
        # Evaluate generation (no retrieval metrics for deep research)
        ground_truth_answers = [
            sample.ground_truth_answer
            for sample in dataset.samples
            if sample.ground_truth_answer
        ] or None
        
        generation_metrics = self.metrics_evaluator.evaluate_generation(
            queries=queries,
            answers=answers,
            contexts=contexts_list,
            ground_truth=ground_truth_answers
        )
        
        # Evaluate latency
        latency_metrics = self.metrics_evaluator.evaluate_latency(latency_data)
        
        results = {
            "dataset_name": dataset.name,
            "num_samples": len(dataset),
            "generation_metrics": generation_metrics.to_dict(),
            "latency_metrics": latency_metrics.to_dict(),
            "iterations": [len(f) for f in all_findings],
            "samples": [
                {
                    "query": q,
                    "answer": a[:200],  # Truncate for logging
                    "findings_count": len(f)
                }
                for q, a, f in zip(queries, answers, all_findings)
            ]
        }
        
        logger.info("Deep Research evaluation completed", extra={"results": results})
        
        return results