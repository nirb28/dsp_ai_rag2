import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(result_lists: List[List[Dict[str, Any]]], 
                           k: int = 60,
                           key_field: str = 'content',
                           score_field: str = 'similarity_score') -> List[Dict[str, Any]]:
    """
    Implement Reciprocal Rank Fusion to combine multiple search result lists.
    
    Args:
        result_lists: List of document lists, each from a different retriever
        k: Constant in the RRF formula (default=60)
        key_field: Field to use as unique identifier for documents
        score_field: Field that contains the score/rank
        
    Returns:
        Combined and reranked list of documents
    """
    # Dictionary to hold document fusion scores
    doc_scores = {}
    
    # Process each result list
    for result_list in result_lists:
        # Create a lookup to efficiently retrieve actual scores later
        doc_lookup = {doc[key_field]: doc for doc in result_list}
        
        # Sort by score (assuming higher is better)
        sorted_docs = sorted(result_list, key=lambda x: x.get(score_field, 0), reverse=True)
        
        # Calculate RRF score for each document
        for rank, doc in enumerate(sorted_docs, 1):
            doc_key = doc[key_field]
            
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    'document': doc_lookup[doc_key],
                    'rrf_score': 0
                }
            
            # RRF formula: 1 / (rank + k)
            doc_scores[doc_key]['rrf_score'] += 1 / (rank + k)
    
    # Sort by RRF score and return only the documents
    sorted_results = sorted(doc_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
    
    # Copy RRF score to each document
    for result in sorted_results:
        result['document']['rrf_score'] = result['rrf_score']
    
    return [item['document'] for item in sorted_results]

def simple_fusion(result_lists: List[List[Dict[str, Any]]],
                  key_field: str = 'content',
                  score_field: str = 'similarity_score') -> List[Dict[str, Any]]:
    """
    Simple fusion method that combines results by normalizing scores and averaging.
    
    Args:
        result_lists: List of document lists, each from a different retriever
        key_field: Field to use as unique identifier for documents
        score_field: Field that contains the score/rank
        
    Returns:
        Combined list of documents with averaged scores
    """
    all_docs = {}
    
    # Process each result list
    for result_list in result_lists:
        # Skip empty lists
        if not result_list:
            continue
            
        # Get max and min scores for normalization
        try:
            scores = [doc.get(score_field, 0) for doc in result_list]
            max_score = max(scores) if scores else 1
            min_score = min(scores) if scores else 0
            score_range = max_score - min_score if max_score != min_score else 1
        except Exception as e:
            logger.warning(f"Error calculating score range: {e}. Using default normalization.")
            max_score = 1
            min_score = 0
            score_range = 1
        
        # Process each document
        for doc in result_list:
            doc_key = doc[key_field]
            
            # Normalize the score to [0, 1]
            normalized_score = (doc.get(score_field, 0) - min_score) / score_range if score_range else 0
            
            if doc_key not in all_docs:
                all_docs[doc_key] = {
                    'document': doc.copy(),
                    'scores': [normalized_score],
                    'count': 1
                }
            else:
                all_docs[doc_key]['scores'].append(normalized_score)
                all_docs[doc_key]['count'] += 1
    
    # Calculate average scores
    for doc_key, doc_data in all_docs.items():
        avg_score = sum(doc_data['scores']) / len(doc_data['scores'])
        doc_data['document']['avg_score'] = avg_score
        doc_data['document'][score_field] = avg_score  # Replace original score
    
    # Sort by average score
    sorted_results = sorted(all_docs.values(), key=lambda x: x['document'][score_field], reverse=True)
    
    return [item['document'] for item in sorted_results]
