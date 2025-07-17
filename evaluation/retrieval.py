"""
Evaluation framework for retrieval performance in RAG systems.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from app.models import Document, QueryResponse
from app.services.rag_service import RAGService

from evaluation.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class RetrievalEvaluator(BaseEvaluator):
    """Evaluator for assessing retrieval performance of RAG systems."""
    
    def __init__(
        self,
        rag_service: RAGService,
        name: str = "retrieval_evaluator",
        save_dir: Optional[str] = None
    ):
        super().__init__(name=name, save_dir=save_dir)
        self.rag_service = rag_service
    
    def evaluate(
        self,
        queries: List[Dict[str, Any]],
        configuration_name: str = "default",
        k: int = 5,
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate retrieval performance on a set of queries with known relevant documents.
        
        Args:
            queries: List of query objects, each containing:
                     - query: the query text
                     - relevant_docs: list of document IDs that are relevant
                     - (optional) metadata: additional information about the query
            configuration_name: RAG configuration name to use
            k: Number of top documents to retrieve for evaluation
            
        Returns:
            List of EvaluationResult objects with metrics
        """
        results = []
        all_precision = []
        all_recall = []
        all_f1 = []
        all_mrr = []
        all_latencies = []
        
        for query_obj in queries:
            query = query_obj["query"]
            relevant_docs = set(query_obj.get("relevant_docs", []))
            query_metadata = query_obj.get("metadata", {})
            
            # Skip queries with no relevant documents
            if not relevant_docs:
                logger.warning(f"No relevant documents for query: {query}")
                continue
            
            # Measure retrieval performance
            start_time = time.time()
            response = self.rag_service.query(
                query=query,
                configuration_name=configuration_name,
                k=k,
                **kwargs
            )
            end_time = time.time()
            latency = end_time - start_time
            all_latencies.append(latency)
            
            # Extract retrieved document IDs
            retrieved_docs = []
            if hasattr(response, "sources"):
                retrieved_docs = [doc.id for doc in response.sources if hasattr(doc, "id")]
            
            # Calculate precision, recall, F1
            retrieved_set = set(retrieved_docs)
            true_positives = len(relevant_docs.intersection(retrieved_set))
            precision = true_positives / len(retrieved_set) if retrieved_set else 0
            recall = true_positives / len(relevant_docs) if relevant_docs else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            
            # Calculate Mean Reciprocal Rank (MRR)
            rank = 0
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    rank = i + 1
                    break
            
            mrr = 1 / rank if rank > 0 else 0
            all_mrr.append(mrr)
            
            # Record individual query results
            results.append(
                EvaluationResult(
                    metric_name="per_query_metrics",
                    value={
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "mrr": mrr,
                        "latency_seconds": latency,
                    },
                    metadata={
                        "query": query,
                        "retrieved_docs": retrieved_docs,
                        "relevant_docs": list(relevant_docs),
                        **query_metadata
                    }
                )
            )
        
        # Calculate aggregate metrics
        results.append(
            EvaluationResult(
                metric_name="precision",
                value=float(np.mean(all_precision)),
                metadata={"k": k, "configuration_name": configuration_name}
            )
        )
        
        results.append(
            EvaluationResult(
                metric_name="recall",
                value=float(np.mean(all_recall)),
                metadata={"k": k, "configuration_name": configuration_name}
            )
        )
        
        results.append(
            EvaluationResult(
                metric_name="f1",
                value=float(np.mean(all_f1)),
                metadata={"k": k, "configuration_name": configuration_name}
            )
        )
        
        results.append(
            EvaluationResult(
                metric_name="mrr",
                value=float(np.mean(all_mrr)),
                metadata={"k": k, "configuration_name": configuration_name}
            )
        )
        
        results.append(
            EvaluationResult(
                metric_name="avg_latency",
                value=float(np.mean(all_latencies)),
                metadata={"k": k, "configuration_name": configuration_name}
            )
        )
        
        return results


class SearchQualityEvaluator(BaseEvaluator):
    """Evaluator for assessing search quality with different retrieval strategies."""
    
    def __init__(
        self,
        rag_service: RAGService,
        name: str = "search_quality_evaluator",
        save_dir: Optional[str] = None
    ):
        super().__init__(name=name, save_dir=save_dir)
        self.rag_service = rag_service
    
    def evaluate_hybrid_retrieval(
        self,
        queries: List[str],
        configuration_name: str = "default",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        k: int = 5,
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Compare different retrieval strategies (vector, BM25, hybrid) for each query.
        
        Args:
            queries: List of query strings to evaluate
            configuration_name: RAG configuration name to use
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            k: Number of documents to retrieve
            
        Returns:
            List of EvaluationResult objects with metrics
        """
        results = []
        
        # Ensure configuration exists
        config = self.rag_service.get_configuration(configuration_name)
        
        for query in queries:
            # Vector-only search
            config_override_vector = config.copy()
            config_override_vector.retrieval_options.vector_search_weight = 1.0
            config_override_vector.retrieval_options.bm25_search_weight = 0.0
            
            response_vector = self.rag_service.query(
                query=query,
                configuration_name=configuration_name,
                k=k,
                config_override=config_override_vector,
                **kwargs
            )
            
            # BM25-only search
            config_override_bm25 = config.copy()
            config_override_bm25.retrieval_options.vector_search_weight = 0.0
            config_override_bm25.retrieval_options.bm25_search_weight = 1.0
            
            response_bm25 = self.rag_service.query(
                query=query,
                configuration_name=configuration_name,
                k=k,
                config_override=config_override_bm25,
                **kwargs
            )
            
            # Hybrid search
            config_override_hybrid = config.copy()
            config_override_hybrid.retrieval_options.vector_search_weight = vector_weight
            config_override_hybrid.retrieval_options.bm25_search_weight = bm25_weight
            
            response_hybrid = self.rag_service.query(
                query=query,
                configuration_name=configuration_name,
                k=k,
                config_override=config_override_hybrid,
                **kwargs
            )
            
            # Record results
            vector_docs = [self._get_doc_summary(doc) for doc in response_vector.sources]
            bm25_docs = [self._get_doc_summary(doc) for doc in response_bm25.sources]
            hybrid_docs = [self._get_doc_summary(doc) for doc in response_hybrid.sources]
            
            # Calculate overlap between strategies
            vector_ids = set(doc.get("id") for doc in vector_docs)
            bm25_ids = set(doc.get("id") for doc in bm25_docs)
            hybrid_ids = set(doc.get("id") for doc in hybrid_docs)
            
            overlap_vector_bm25 = len(vector_ids.intersection(bm25_ids))
            overlap_vector_hybrid = len(vector_ids.intersection(hybrid_ids))
            overlap_bm25_hybrid = len(bm25_ids.intersection(hybrid_ids))
            
            results.append(
                EvaluationResult(
                    metric_name="retrieval_strategy_comparison",
                    value={
                        "overlap_vector_bm25": overlap_vector_bm25,
                        "overlap_vector_hybrid": overlap_vector_hybrid,
                        "overlap_bm25_hybrid": overlap_bm25_hybrid,
                        "unique_to_vector": len(vector_ids - bm25_ids.union(hybrid_ids)),
                        "unique_to_bm25": len(bm25_ids - vector_ids.union(hybrid_ids)),
                        "unique_to_hybrid": len(hybrid_ids - vector_ids.union(bm25_ids))
                    },
                    metadata={
                        "query": query,
                        "vector_docs": vector_docs,
                        "bm25_docs": bm25_docs,
                        "hybrid_docs": hybrid_docs,
                        "vector_weight": vector_weight,
                        "bm25_weight": bm25_weight,
                        "configuration_name": configuration_name,
                        "k": k
                    }
                )
            )
        
        return results
    
    @staticmethod
    def _get_doc_summary(doc: Document) -> Dict[str, Any]:
        """Create a summary of a document for inclusion in results."""
        return {
            "id": doc.id if hasattr(doc, "id") else None,
            "score": doc.score if hasattr(doc, "score") else None,
            "title": doc.metadata.get("title", "") if hasattr(doc, "metadata") else "",
            "filename": doc.filename if hasattr(doc, "filename") else "",
        }
