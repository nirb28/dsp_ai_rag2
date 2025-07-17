"""
Evaluation framework for query response quality in RAG systems.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.models import QueryResponse
from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingService

from evaluation.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class QueryEvaluator(BaseEvaluator):
    """Evaluator for assessing query response quality."""
    
    def __init__(
        self,
        rag_service: RAGService,
        name: str = "query_evaluator",
        save_dir: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        super().__init__(name=name, save_dir=save_dir)
        self.rag_service = rag_service
        self.embedding_service = embedding_service
    
    def evaluate(
        self,
        eval_data: List[Dict[str, Any]],
        configuration_name: str = "default",
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate query response quality on a set of evaluation data.
        
        Args:
            eval_data: List of evaluation objects, each containing:
                       - query: the query text
                       - expected_answer: the expected answer (ground truth)
                       - (optional) metadata: additional information about the query
            configuration_name: RAG configuration name to use
            
        Returns:
            List of EvaluationResult objects with metrics
        """
        results = []
        
        for item in eval_data:
            query = item["query"]
            expected_answer = item["expected_answer"]
            query_metadata = item.get("metadata", {})
            
            # Get response from RAG system
            start_time = time.time()
            response = self.rag_service.query(
                query=query,
                configuration_name=configuration_name,
                **kwargs
            )
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract generated answer
            generated_answer = response.answer if hasattr(response, "answer") else ""
            
            # Calculate metrics
            metrics = self._calculate_metrics(generated_answer, expected_answer)
            
            # Record individual query results
            results.append(
                EvaluationResult(
                    metric_name="query_response_metrics",
                    value={
                        **metrics,
                        "latency_seconds": latency,
                    },
                    metadata={
                        "query": query,
                        "expected_answer": expected_answer,
                        "generated_answer": generated_answer,
                        "configuration_name": configuration_name,
                        **query_metadata
                    }
                )
            )
        
        # Calculate aggregate metrics
        accuracy_scores = [
            result.value["answer_correctness"]
            for result in results
            if "answer_correctness" in result.value
        ]
        
        relevance_scores = [
            result.value["answer_relevance"]
            for result in results
            if "answer_relevance" in result.value
        ]
        
        completeness_scores = [
            result.value["answer_completeness"]
            for result in results
            if "answer_completeness" in result.value
        ]
        
        similarity_scores = [
            result.value["semantic_similarity"]
            for result in results
            if "semantic_similarity" in result.value
        ]
        
        if accuracy_scores:
            results.append(
                EvaluationResult(
                    metric_name="avg_answer_correctness",
                    value=float(np.mean(accuracy_scores)),
                    metadata={"configuration_name": configuration_name}
                )
            )
        
        if relevance_scores:
            results.append(
                EvaluationResult(
                    metric_name="avg_answer_relevance",
                    value=float(np.mean(relevance_scores)),
                    metadata={"configuration_name": configuration_name}
                )
            )
        
        if completeness_scores:
            results.append(
                EvaluationResult(
                    metric_name="avg_answer_completeness",
                    value=float(np.mean(completeness_scores)),
                    metadata={"configuration_name": configuration_name}
                )
            )
        
        if similarity_scores:
            results.append(
                EvaluationResult(
                    metric_name="avg_semantic_similarity",
                    value=float(np.mean(similarity_scores)),
                    metadata={"configuration_name": configuration_name}
                )
            )
        
        return results
    
    def _calculate_metrics(
        self, 
        generated_answer: str, 
        expected_answer: str
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics for a single query response.
        
        Args:
            generated_answer: The answer generated by the RAG system
            expected_answer: The expected ground truth answer
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Exact match (binary)
        metrics["exact_match"] = 1.0 if self._normalize_text(generated_answer) == self._normalize_text(expected_answer) else 0.0
        
        # Token overlap (F1 score)
        metrics["token_overlap"] = self._calculate_token_overlap(generated_answer, expected_answer)
        
        # Contains key information (check if all keywords from expected answer are in generated)
        metrics["contains_key_info"] = self._calculate_keyword_coverage(generated_answer, expected_answer)
        
        # Answer relevance (based on keyword density and structural similarity)
        metrics["answer_relevance"] = self._calculate_relevance(generated_answer, expected_answer)
        
        # Answer completeness (based on length ratio and keyword coverage)
        metrics["answer_completeness"] = self._calculate_completeness(generated_answer, expected_answer)
        
        # Answer correctness (combining other metrics)
        metrics["answer_correctness"] = (
            0.3 * metrics["token_overlap"] +
            0.4 * metrics["contains_key_info"] +
            0.3 * metrics["answer_relevance"]
        )
        
        # Semantic similarity (if embedding service is available)
        if self.embedding_service:
            try:
                metrics["semantic_similarity"] = self._calculate_semantic_similarity(generated_answer, expected_answer)
            except Exception as e:
                logger.error(f"Error calculating semantic similarity: {e}")
        
        return metrics
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def _calculate_token_overlap(self, generated: str, expected: str) -> float:
        """Calculate F1 score for token overlap."""
        generated_tokens = set(self._normalize_text(generated).split())
        expected_tokens = set(self._normalize_text(expected).split())
        
        if not expected_tokens or not generated_tokens:
            return 0.0
        
        common_tokens = generated_tokens.intersection(expected_tokens)
        precision = len(common_tokens) / len(generated_tokens)
        recall = len(common_tokens) / len(expected_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_keyword_coverage(self, generated: str, expected: str) -> float:
        """Calculate coverage of keywords from expected answer in generated answer."""
        # Extract "important" words (non-stopwords) from expected answer
        expected_text = self._normalize_text(expected)
        generated_text = self._normalize_text(generated)
        
        # Simple approach: use words with 4+ characters as "keywords"
        keywords = [word for word in expected_text.split() if len(word) >= 4]
        
        if not keywords:
            return 1.0  # No keywords to match
            
        # Count matches
        matches = sum(1 for keyword in keywords if keyword in generated_text)
        return matches / len(keywords)
    
    def _calculate_relevance(self, generated: str, expected: str) -> float:
        """Calculate relevance of generated answer to expected answer."""
        # Combine token overlap and keyword coverage
        token_overlap = self._calculate_token_overlap(generated, expected)
        keyword_coverage = self._calculate_keyword_coverage(generated, expected)
        
        # Weight more heavily toward keyword coverage
        return 0.4 * token_overlap + 0.6 * keyword_coverage
    
    def _calculate_completeness(self, generated: str, expected: str) -> float:
        """Calculate completeness of generated answer relative to expected answer."""
        # Normalize texts
        gen_norm = self._normalize_text(generated)
        exp_norm = self._normalize_text(expected)
        
        # Length comparison (penalize if too short, but don't reward for excessive length)
        gen_words = len(gen_norm.split())
        exp_words = len(exp_norm.split())
        
        length_ratio = min(gen_words / max(1, exp_words), 1.5) / 1.5
        
        # Combine with keyword coverage
        keyword_coverage = self._calculate_keyword_coverage(generated, expected)
        
        return 0.4 * length_ratio + 0.6 * keyword_coverage
    
    def _calculate_semantic_similarity(self, generated: str, expected: str) -> float:
        """Calculate semantic similarity using embeddings."""
        if not self.embedding_service:
            return 0.0
            
        # Get embeddings
        generated_embedding = self.embedding_service.get_embeddings([generated])[0]
        expected_embedding = self.embedding_service.get_embeddings([expected])[0]
        
        # Calculate cosine similarity
        dot_product = np.dot(generated_embedding, expected_embedding)
        norm_generated = np.linalg.norm(generated_embedding)
        norm_expected = np.linalg.norm(expected_embedding)
        
        similarity = dot_product / (norm_generated * norm_expected)
        return float(similarity)


class QuerySetBenchmark(BaseEvaluator):
    """
    Benchmark multiple configurations against a standard query set.
    """
    
    def __init__(
        self,
        rag_service: RAGService,
        name: str = "query_benchmark",
        save_dir: Optional[str] = None,
        max_workers: int = 4
    ):
        super().__init__(name=name, save_dir=save_dir)
        self.rag_service = rag_service
        self.max_workers = max_workers
    
    def evaluate(
        self,
        query_set: List[str],
        configurations: List[str],
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Compare multiple configurations on the same set of queries.
        
        Args:
            query_set: List of query strings to benchmark
            configurations: List of configuration names to compare
            
        Returns:
            List of EvaluationResult objects with metrics
        """
        results = []
        
        for config_name in configurations:
            start_time = time.time()
            
            # Process queries with multithreading for faster benchmarking
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_query = {
                    executor.submit(
                        self._run_query, config_name, query, **kwargs
                    ): query for query in query_set
                }
                
                query_results = []
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        query_results.append(future.result())
                    except Exception as e:
                        logger.error(f"Query '{query}' with config '{config_name}' failed: {e}")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            latencies = [qr["latency"] for qr in query_results]
            avg_latency = np.mean(latencies) if latencies else 0
            p95_latency = np.percentile(latencies, 95) if latencies else 0
            
            results.append(
                EvaluationResult(
                    metric_name="configuration_benchmark",
                    value={
                        "total_time": total_time,
                        "avg_query_time": avg_latency,
                        "p95_query_time": p95_latency,
                        "queries_per_second": len(query_results) / total_time if total_time > 0 else 0,
                        "successful_queries": len(query_results)
                    },
                    metadata={
                        "configuration_name": config_name,
                        "total_queries": len(query_set),
                        "query_details": [
                            {
                                "query": qr["query"],
                                "latency": qr["latency"],
                                "sources_count": qr["sources_count"]
                            }
                            for qr in query_results
                        ]
                    }
                )
            )
        
        return results
    
    def _run_query(
        self,
        configuration_name: str,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a single query and return results with timing."""
        start_time = time.time()
        response = self.rag_service.query(
            query=query,
            configuration_name=configuration_name,
            **kwargs
        )
        end_time = time.time()
        
        sources_count = len(response.sources) if hasattr(response, "sources") else 0
        
        return {
            "query": query,
            "latency": end_time - start_time,
            "sources_count": sources_count
        }
