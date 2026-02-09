#orchestration layer for evaluation 

from typing import List, Dict, Any, Optional 
from pathlib import Path 

from app.evals.evaluators import FastRAGEvaluator, DeepResearchEvaluator
from app.evals.datasets import EvaluationDataset, DatasetManager
from app.evals.reports import EvaluationReport
from app.database import get_db
from config.logging_config import get_logger

logger = get_logger(__name__)

class EvaluationRunner:
    #Orchestrates the evaluation runs 

    def __init__(self):
        self.fast_rag_evaluator = FastRAGEvaluator()
        self.deep_research_evaluator = DeepResearchEvaluator()
        self.dataset_manager = DatasetManager()
        logger.info("EvaluationRunner initialized")

    def run_fast_rag_evaluation(
        self,
        dataset_name: Optional[str] = None, 
        dataset: Optional[EvaluationDataset] = None
    ) -> Dict[str, Any]:
        logger.info("Running Fast RAG evaluation")
        if dataset is None:
            if dataset_name is None:
                raise ValueError("Either dataset name or dataset is provided")
            dataset = self.dataset_manager.get_dataset(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found")
        
        db_gen = get_db()
        db = next(db_gen)
        try:
            results = self.fast_rag_evaluator.evaluate(dataset, db)

            report = EvaluationReport.from_results(
                mode="fast_rag",
                results=results
            )
            logger.info("Fast RAG evaluation completed successfully")
            
            return {
                "results": results,
                "report": report.to_dict()
            }
        finally:
            db.close()

    def run_deep_research_evaluation(
        self, 
        dataset_name: Optional[str] = None, 
        dataset: Optional[EvaluationDataset] = None
    ) -> Dict[str, Any]:
        logger.info("Running Deep Research evaluation")
        
        # Load dataset
        if dataset is None:
            if dataset_name is None:
                raise ValueError("Either dataset_name or dataset must be provided")
            dataset = self.dataset_manager.get_dataset(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found")
        
        # Get database session
        db_gen = get_db()
        db = next(db_gen)
        
        try:
            # Run evaluation
            results = self.deep_research_evaluator.evaluate(dataset, db)
            
            # Generate report
            report = EvaluationReport.from_results(
                mode="deep_research",
                results=results
            )
            
            logger.info("Deep Research evaluation completed successfully")
            
            return {
                "results": results,
                "report": report.to_dict()
            }
        finally:
            db.close()

    def compare_modes(
        self,
        dataset_name: Optional[str] = None,
        dataset: Optional[EvaluationDataset] = None
    ) -> Dict[str, Any]:
        logger.info("Running mode comparison")
        
        # Load dataset
        if dataset is None:
            if dataset_name is None:
                raise ValueError("Either dataset_name or dataset must be provided")
            dataset = self.dataset_manager.get_dataset(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found")
        
        # Run both evaluations
        fast_rag_results = self.run_fast_rag_evaluation(dataset=dataset)
        deep_research_results = self.run_deep_research_evaluation(dataset=dataset)
        
        # Compare
        comparison = {
            "dataset_name": dataset.name,
            "num_samples": len(dataset),
            "fast_rag": {
                "generation_metrics": fast_rag_results["results"]["generation_metrics"],
                "latency_metrics": fast_rag_results["results"]["latency_metrics"]
            },
            "deep_research": {
                "generation_metrics": deep_research_results["results"]["generation_metrics"],
                "latency_metrics": deep_research_results["results"]["latency_metrics"]
            },
            "comparison": {
                "latency_ratio": (
                    deep_research_results["results"]["latency_metrics"]["total_time_ms"] /
                    fast_rag_results["results"]["latency_metrics"]["total_time_ms"]
                ),
                "faithfulness_diff": (
                    deep_research_results["results"]["generation_metrics"]["faithfulness"] -
                    fast_rag_results["results"]["generation_metrics"]["faithfulness"]
                ),
                "relevance_diff": (
                    deep_research_results["results"]["generation_metrics"]["answer_relevance"] -
                    fast_rag_results["results"]["generation_metrics"]["answer_relevance"]
                )
            }
        }
        
        logger.info("Mode comparison completed", extra={"comparison": comparison})
        
        return comparison