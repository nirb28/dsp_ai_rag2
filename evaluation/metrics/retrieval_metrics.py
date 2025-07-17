"""
Metrics for evaluating retrieval performance in RAG systems.
"""
import numpy as np
from typing import List, Dict, Any, Set, Union, Optional


def precision_at_k(relevant_docs: Union[List, Set], retrieved_docs: Union[List, Set], k: Optional[int] = None) -> float:
    """
    Calculate precision@k.
    
    Args:
        relevant_docs: Set or list of relevant document IDs
        retrieved_docs: Set or list of retrieved document IDs
        k: Number of retrieved documents to consider (defaults to all)
        
    Returns:
        Precision value between 0 and 1
    """
    if isinstance(relevant_docs, list):
        relevant_docs = set(relevant_docs)
    
    if isinstance(retrieved_docs, list):
        retrieved_docs = set(retrieved_docs)
        
    if k is not None and k < len(retrieved_docs):
        # If k is specified, consider only top k retrieved docs
        # Note: This assumes retrieved_docs is already in relevance order
        retrieved_docs = set(list(retrieved_docs)[:k])
    
    if not retrieved_docs:
        return 0.0
        
    true_positives = len(relevant_docs.intersection(retrieved_docs))
    return true_positives / len(retrieved_docs)


def recall_at_k(relevant_docs: Union[List, Set], retrieved_docs: Union[List, Set], k: Optional[int] = None) -> float:
    """
    Calculate recall@k.
    
    Args:
        relevant_docs: Set or list of relevant document IDs
        retrieved_docs: Set or list of retrieved document IDs
        k: Number of retrieved documents to consider (defaults to all)
        
    Returns:
        Recall value between 0 and 1
    """
    if isinstance(relevant_docs, list):
        relevant_docs = set(relevant_docs)
    
    if isinstance(retrieved_docs, list):
        retrieved_docs = set(retrieved_docs)
        
    if k is not None and k < len(retrieved_docs):
        # If k is specified, consider only top k retrieved docs
        retrieved_docs = set(list(retrieved_docs)[:k])
    
    if not relevant_docs:
        return 1.0  # All relevant docs are retrieved if there are no relevant docs
        
    true_positives = len(relevant_docs.intersection(retrieved_docs))
    return true_positives / len(relevant_docs)


def f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score value between 0 and 1
    """
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)


def mean_reciprocal_rank(relevant_docs: Union[List, Set], retrieved_docs: List) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        relevant_docs: Set or list of relevant document IDs
        retrieved_docs: List of retrieved document IDs in rank order
        
    Returns:
        MRR value between 0 and 1
    """
    if isinstance(relevant_docs, list):
        relevant_docs = set(relevant_docs)
    
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    # Find the rank of the first relevant document
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    
    return 0.0


def normalized_discounted_cumulative_gain(
    relevant_docs: Union[List, Set], 
    retrieved_docs: List, 
    relevance_scores: Optional[Dict[str, float]] = None,
    k: Optional[int] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        relevant_docs: Set or list of relevant document IDs
        retrieved_docs: List of retrieved document IDs in rank order
        relevance_scores: Optional dict mapping document IDs to relevance scores
                         (defaults to binary relevance if not provided)
        k: Number of documents to consider (defaults to all)
        
    Returns:
        NDCG value between 0 and 1
    """
    if isinstance(relevant_docs, list):
        relevant_docs = set(relevant_docs)
    
    if not relevant_docs or not retrieved_docs:
        return 0.0
        
    # Limit to top k if specified
    if k is not None:
        retrieved_docs = retrieved_docs[:k]
        
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            # Get relevance score (binary or provided score)
            rel = relevance_scores.get(doc_id, 1.0) if relevance_scores else 1.0
            # Use log base 2 as per standard definition
            dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed
    
    # Calculate ideal DCG (IDCG)
    # For binary relevance, this is the DCG if all relevant docs were retrieved first
    if not relevance_scores:
        # Binary relevance
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), len(retrieved_docs))))
    else:
        # Get all relevance scores for relevant documents
        rel_scores = [relevance_scores.get(doc_id, 1.0) for doc_id in relevant_docs]
        # Sort in descending order
        rel_scores.sort(reverse=True)
        # Take only as many as the retrieved docs
        rel_scores = rel_scores[:len(retrieved_docs)]
        # Calculate IDCG
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_scores))
    
    if idcg == 0:
        return 0.0
        
    return dcg / idcg


def mean_average_precision(queries_relevant_retrieved: List[Dict[str, Union[List, Set]]]) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.
    
    Args:
        queries_relevant_retrieved: List of dicts, each with 'relevant' and 'retrieved' keys
        
    Returns:
        MAP value between 0 and 1
    """
    if not queries_relevant_retrieved:
        return 0.0
    
    average_precisions = []
    
    for query_data in queries_relevant_retrieved:
        relevant = query_data['relevant']
        retrieved = query_data['retrieved']
        
        if not relevant or not retrieved:
            continue
            
        # Convert to sets if they are lists
        if isinstance(relevant, list):
            relevant = set(relevant)
        
        # Calculate precision at each relevant document
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                relevant_count += 1
                # Precision at this point (i+1 docs retrieved)
                precisions.append(relevant_count / (i + 1))
        
        # Average precision for this query
        if precisions:
            avg_precision = sum(precisions) / len(relevant)
            average_precisions.append(avg_precision)
    
    # Mean of average precisions across all queries
    if not average_precisions:
        return 0.0
        
    return sum(average_precisions) / len(average_precisions)


def hit_rate(queries_relevant_retrieved: List[Dict[str, Union[List, Set]]]) -> float:
    """
    Calculate hit rate (fraction of queries that retrieved at least one relevant document).
    
    Args:
        queries_relevant_retrieved: List of dicts, each with 'relevant' and 'retrieved' keys
        
    Returns:
        Hit rate value between 0 and 1
    """
    if not queries_relevant_retrieved:
        return 0.0
    
    hits = 0
    
    for query_data in queries_relevant_retrieved:
        relevant = query_data['relevant']
        retrieved = query_data['retrieved']
        
        if not relevant:
            continue
            
        # Convert to sets if they are lists
        if isinstance(relevant, list):
            relevant = set(relevant)
        if isinstance(retrieved, list):
            retrieved = set(retrieved)
        
        # Check if at least one relevant document was retrieved
        if relevant.intersection(retrieved):
            hits += 1
    
    return hits / len(queries_relevant_retrieved)
