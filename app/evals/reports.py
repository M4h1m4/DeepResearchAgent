from typing import Dict, Any, Optional 
from dataclasses import dataclass 
from datetime import datetime 
import json 
import os 

from config.logging_config import get_logger 

logger = get_logger(__name__)

@dataclass
class EvaluationReport:
    """Evaluation report."""f
    
    mode: str  # "fast_rag" or "deep_research"
    dataset_name: str
    timestamp: str
    metrics: Dict[str, Any]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "summary": self.summary
        }

    @classmethod
    def from_results(
        cls,
        mode: str,
        results: Dict[str, Any]
    ) -> "EvaluationReport":
        metrics = {}
        summary = {}
        
        if "retrieval_metrics" in results:
            metrics["retrieval"] = results["retrieval_metrics"]
            summary["retrieval"] = {
                "precision": results["retrieval_metrics"]["precision_at_k"],
                "recall": results["retrieval_metrics"]["recall_at_k"],
                "mrr": results["retrieval_metrics"]["mrr"]
            }
        
        if "generation_metrics" in results:
            metrics["generation"] = results["generation_metrics"]
            summary["generation"] = {
                "faithfulness": results["generation_metrics"]["faithfulness"],
                "relevance": results["generation_metrics"]["answer_relevance"],
                "context_precision": results["generation_metrics"]["context_precision"],
                "context_recall": results["generation_metrics"]["context_recall"]
            }
        
        if "latency_metrics" in results:
            metrics["latency"] = results["latency_metrics"]
            summary["latency"] = {
                "avg_total_ms": results["latency_metrics"]["total_time_ms"],
                "avg_retrieval_ms": results["latency_metrics"]["retrieval_time_ms"],
                "avg_generation_ms": results["latency_metrics"]["generation_time_ms"]
            }
        
        return cls(
            mode=mode,
            dataset_name=results.get("dataset_name", "unknown"),
            timestamp=datetime.utcnow().isoformat(),
            metrics=metrics,
            summary=summary
        )
    
    def save(self, filepath: str):
        logger.info(f"Saving evaluation report to {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Report saved successfully")
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {self.mode.upper()}")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"\nSummary:")
        
        if "retrieval" in self.summary:
            print(f"\nRetrieval Metrics:")
            print(f"  Precision@K: {self.summary['retrieval']['precision']:.3f}")
            print(f"  Recall@K: {self.summary['retrieval']['recall']:.3f}")
            print(f"  MRR: {self.summary['retrieval']['mrr']:.3f}")
        
        if "generation" in self.summary:
            print(f"\nGeneration Metrics:")
            print(f"  Faithfulness: {self.summary['generation']['faithfulness']:.3f}")
            print(f"  Answer Relevance: {self.summary['generation']['relevance']:.3f}")
            print(f"  Context Precision: {self.summary['generation']['context_precision']:.3f}")
            print(f"  Context Recall: {self.summary['generation']['context_recall']:.3f}")
        
        if "latency" in self.summary:
            print(f"\nLatency Metrics:")
            print(f"  Avg Total Time: {self.summary['latency']['avg_total_ms']:.2f} ms")
            print(f"  Avg Retrieval Time: {self.summary['latency']['avg_retrieval_ms']:.2f} ms")
            print(f"  Avg Generation Time: {self.summary['latency']['avg_generation_ms']:.2f} ms")
        
        print(f"{'='*60}\n")