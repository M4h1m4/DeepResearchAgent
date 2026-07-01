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
                raise ValueError("Either dataset_name or dataset must be provided")
            
            # Check if user wants to create dataset from database
            if dataset_name == "custom_database" or dataset_name == "database":
                logger.info("Creating evaluation dataset from database")
                db_gen = get_db()
                db = next(db_gen)
                try:
                    dataset = self.dataset_manager.create_dataset_from_database(
                        db_session=db,
                        max_samples=10
                    )
                    if dataset is None:
                        raise ValueError("Failed to create dataset from database. Make sure you have documents uploaded.")
                finally:
                    db.close()
            else:
                # Try to load as benchmark dataset first (RAGAS)
                available_benchmarks = self.dataset_manager.list_available_benchmarks()
                if dataset_name in available_benchmarks.get("ragas", []):
                    logger.info(f"Loading RAGAS benchmark dataset: {dataset_name}")
                    try:
                        dataset = self.dataset_manager.load_ragas_dataset(
                            dataset_name=dataset_name,
                            max_samples=10  # Limit to 10 for faster testing
                        )
                        if dataset is None:
                            raise ValueError(f"Failed to load RAGAS dataset '{dataset_name}'. Check server logs for details.")
                    except Exception as e:
                        logger.error(f"Error loading RAGAS dataset {dataset_name}: {str(e)}", exc_info=True)
                        raise ValueError(f"Failed to load RAGAS dataset '{dataset_name}': {str(e)}")
                elif dataset_name in available_benchmarks.get("beir", []):
                    logger.info(f"Loading BEIR benchmark dataset: {dataset_name}")
                    try:
                        dataset = self.dataset_manager.load_beir_dataset(
                            dataset_name=dataset_name,
                            max_samples=10
                        )
                        if dataset is None:
                            raise ValueError(f"Failed to load BEIR dataset '{dataset_name}'. Check server logs for details.")
                    except Exception as e:
                        logger.error(f"Error loading BEIR dataset {dataset_name}: {str(e)}", exc_info=True)
                        raise ValueError(f"Failed to load BEIR dataset '{dataset_name}': {str(e)}")
                else:
                    # Try loading from saved datasets
                    dataset = self.dataset_manager.get_dataset(dataset_name)
                
                if dataset is None:
                    raise ValueError(f"Dataset '{dataset_name}' not found. Available benchmarks: {available_benchmarks}. Use 'custom_database' to create from your documents.")
        
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
            
            # Try to load as benchmark dataset first (RAGAS)
            available_benchmarks = self.dataset_manager.list_available_benchmarks()
            if dataset_name in available_benchmarks.get("ragas", []):
                logger.info(f"Loading RAGAS benchmark dataset: {dataset_name}")
                dataset = self.dataset_manager.load_ragas_dataset(
                    dataset_name=dataset_name,
                    max_samples=10  # Limit to 50 for faster testing
                )
            elif dataset_name in available_benchmarks.get("beir", []):
                logger.info(f"Loading BEIR benchmark dataset: {dataset_name}")
                dataset = self.dataset_manager.load_beir_dataset(
                    dataset_name=dataset_name,
                    max_samples=10
                )
            else:
                # Try loading from saved datasets
                dataset = self.dataset_manager.get_dataset(dataset_name)
            
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found. Available benchmarks: {available_benchmarks}")
        
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
            
            # Try to load as benchmark dataset first (RAGAS)
            available_benchmarks = self.dataset_manager.list_available_benchmarks()
            if dataset_name in available_benchmarks.get("ragas", []):
                logger.info(f"Loading RAGAS benchmark dataset: {dataset_name}")
                dataset = self.dataset_manager.load_ragas_dataset(
                    dataset_name=dataset_name,
                    max_samples=10  # Limit to 50 for faster testing
                )
            elif dataset_name in available_benchmarks.get("beir", []):
                logger.info(f"Loading BEIR benchmark dataset: {dataset_name}")
                dataset = self.dataset_manager.load_beir_dataset(
                    dataset_name=dataset_name,
                    max_samples=10
                )
            else:
                # Try loading from saved datasets
                dataset = self.dataset_manager.get_dataset(dataset_name)
            
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found. Available benchmarks: {available_benchmarks}")
        
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