#!/usr/bin/env python
"""
Example script demonstrating how to use the model server reranking in the RAG pipeline.
"""

import sys
import os
import asyncio
import json
import logging
import requests
from pathlib import Path

# Add the project root to Python path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import RerankerConfig, RerankerModel
from app.services.reranker_service import RerankerService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function demonstrating how to use model server reranking."""
    
    # Define a reranker configuration with model server
    config = RerankerConfig(
        enabled=True,
        model=RerankerModel.LOCAL_MODEL_SERVER,  # Use model server for reranking
        model_name="ms-marco-minilm",     # This model name should match one of the downloaded models
        server_url="http://localhost:9001", # URL of the model server
        top_n=5,                          # Number of documents to rerank
        score_threshold=0.2               # Minimum score to include a document
    )
    
    # Create a mock document collection for demonstration
    documents = [
        {
            "content": "Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time.",
            "metadata": {"source": "doc1", "page": 1},
            "similarity_score": 0.78
        },
        {
            "content": "Python is a popular programming language used for various purposes like web development, data analysis, and artificial intelligence.",
            "metadata": {"source": "doc2", "page": 1},
            "similarity_score": 0.65
        },
        {
            "content": "The weather forecast predicts rain for tomorrow afternoon.",
            "metadata": {"source": "doc3", "page": 1},
            "similarity_score": 0.45
        },
        {
            "content": "Deep learning is a subset of machine learning that uses neural networks with many layers.",
            "metadata": {"source": "doc4", "page": 1},
            "similarity_score": 0.72
        },
        {
            "content": "Natural language processing is concerned with the interactions between computers and human language.",
            "metadata": {"source": "doc5", "page": 1},
            "similarity_score": 0.59
        }
    ]
    
    # Create reranker service with the configured model
    reranker_service = RerankerService(config)
    
    # Sample query
    query = "What is machine learning?"
    
    logger.info(f"Original documents with initial similarity scores:")
    for i, doc in enumerate(documents):
        logger.info(f"{i+1}. Score: {doc['similarity_score']:.4f} - {doc['content'][:50]}...")
    
    # Rerank the documents based on relevance to the query
    if reranker_service.config.enabled:
        logger.info(f"\nReranking documents using model server ({config.model_name})...")
        reranked_docs = await reranker_service.rerank(query, documents)
        
        logger.info(f"\nReranked documents:")
        for i, doc in enumerate(reranked_docs):
            logger.info(f"{i+1}. Score: {doc['similarity_score']:.4f} - {doc['content'][:50]}...")
    else:
        logger.warning("Reranking is not enabled in the configuration.")

if __name__ == "__main__":
    asyncio.run(main())
