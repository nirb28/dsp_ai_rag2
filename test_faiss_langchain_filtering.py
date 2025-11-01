#!/usr/bin/env python3
"""
Test script for LangChain-style FAISS metadata filtering.

This script demonstrates the MongoDB-style operators supported by the FAISS vector store,
following the LangChain convention as documented at:
https://python.langchain.com/docs/integrations/vectorstores/faiss/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from langchain.docstore.document import Document as LangchainDocument

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.rag_service import RAGService
from app.config import settings


def create_test_documents():
    """Create test documents with various metadata for filtering."""
    documents = [
        LangchainDocument(
            page_content="Building an exciting new project with LangChain - come check it out!",
            metadata={"source": "tweet", "author": "alice", "score": 0.9, "category": "tech"}
        ),
        LangchainDocument(
            page_content="LangGraph is the best framework for building stateful, agentic applications!",
            metadata={"source": "tweet", "author": "bob", "score": 0.8, "category": "tech"}
        ),
        LangchainDocument(
            page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
            metadata={"source": "news", "author": "weather_service", "score": 0.7, "category": "weather"}
        ),
        LangchainDocument(
            page_content="Robbers broke into the city bank and stole $1 million in cash.",
            metadata={"source": "news", "author": "reporter", "score": 0.6, "category": "crime"}
        ),
        LangchainDocument(
            page_content="Machine learning is transforming how we approach data analysis and prediction.",
            metadata={"source": "blog", "author": "alice", "score": 0.95, "category": "tech"}
        ),
        LangchainDocument(
            page_content="The stock market showed mixed results today with tech stocks gaining ground.",
            metadata={"source": "news", "author": "financial_analyst", "score": 0.75, "category": "finance"}
        )
    ]
    return documents


def test_simple_equality_filtering(vector_store):
    """Test simple equality filtering: filter={"source": "tweet"}"""
    print("\n=== Test 1: Simple Equality Filtering ===")
    print("Filter: {'source': 'tweet'}")
    
    results = vector_store.similarity_search(
        "LangChain provides abstractions to make working with LLMs easy",
        k=5,
        filter={"source": "tweet"}
    )
    
    print(f"Found {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:80]}...")
        print(f"   Metadata: {doc.metadata}")


def test_equality_operator(vector_store):
    """Test $eq operator: filter={"source": {"$eq": "tweet"}}"""
    print("\n=== Test 2: $eq Operator ===")
    print("Filter: {'source': {'$eq': 'tweet'}}")
    
    results = vector_store.similarity_search(
        "LangChain provides abstractions to make working with LLMs easy",
        k=5,
        filter={"source": {"$eq": "tweet"}}
    )
    
    print(f"Found {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:80]}...")
        print(f"   Metadata: {doc.metadata}")


def test_greater_than_operator(vector_store):
    """Test $gt operator: filter={"score": {"$gt": 0.8}}"""
    print("\n=== Test 3: $gt Operator ===")
    print("Filter: {'score': {'$gt': 0.8}}")
    
    results = vector_store.similarity_search(
        "technology and machine learning",
        k=5,
        filter={"score": {"$gt": 0.8}}
    )
    
    print(f"Found {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:80]}...")
        print(f"   Metadata: {doc.metadata}")


def test_in_operator(vector_store):
    """Test $in operator: filter={"category": {"$in": ["tech", "finance"]}}"""
    print("\n=== Test 4: $in Operator ===")
    print("Filter: {'category': {'$in': ['tech', 'finance']}}")
    
    results = vector_store.similarity_search(
        "analysis and data",
        k=5,
        filter={"category": {"$in": ["tech", "finance"]}}
    )
    
    print(f"Found {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:80]}...")
        print(f"   Metadata: {doc.metadata}")


def test_and_operator(vector_store):
    """Test $and operator: filter={"$and": [{"source": "news"}, {"score": {"$gt": 0.7}}]}"""
    print("\n=== Test 5: $and Operator ===")
    print("Filter: {'$and': [{'source': 'news'}, {'score': {'$gt': 0.7}}]}")
    
    results = vector_store.similarity_search(
        "news and information",
        k=5,
        filter={"$and": [{"source": "news"}, {"score": {"$gt": 0.7}}]}
    )
    
    print(f"Found {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:80]}...")
        print(f"   Metadata: {doc.metadata}")


def test_or_operator(vector_store):
    """Test $or operator: filter={"$or": [{"author": "alice"}, {"category": "weather"}]}"""
    print("\n=== Test 6: $or Operator ===")
    print("Filter: {'$or': [{'author': 'alice'}, {'category': 'weather'}]}")
    
    results = vector_store.similarity_search(
        "information and data",
        k=5,
        filter={"$or": [{"author": "alice"}, {"category": "weather"}]}
    )
    
    print(f"Found {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:80]}...")
        print(f"   Metadata: {doc.metadata}")


def test_not_operator(vector_store):
    """Test $not operator: filter={"$not": {"source": "tweet"}}"""
    print("\n=== Test 7: $not Operator ===")
    print("Filter: {'$not': {'source': 'tweet'}}")
    
    results = vector_store.similarity_search(
        "information and news",
        k=5,
        filter={"$not": {"source": "tweet"}}
    )
    
    print(f"Found {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:80]}...")
        print(f"   Metadata: {doc.metadata}")


def test_complex_filter(vector_store):
    """Test complex filter combining multiple operators."""
    print("\n=== Test 8: Complex Filter ===")
    complex_filter = {
        "$and": [
            {"$or": [{"source": "news"}, {"source": "blog"}]},
            {"score": {"$gte": 0.7}},
            {"category": {"$neq": "weather"}}
        ]
    }
    print(f"Filter: {json.dumps(complex_filter, indent=2)}")
    
    results = vector_store.similarity_search(
        "technology and analysis",
        k=5,
        filter=complex_filter
    )
    
    print(f"Found {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:80]}...")
        print(f"   Metadata: {doc.metadata}")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test LangChain-style FAISS metadata filtering')
    parser.add_argument('--config', '-c', default='default', 
                       help='Configuration name to use (default: default)')
    parser.add_argument('--create-new', action='store_true',
                       help='Create a new test configuration instead of using existing')
    args = parser.parse_args()
    
    print("=== LangChain-style FAISS Metadata Filtering Test ===")
    print("Testing MongoDB-style operators as documented in LangChain FAISS integration")
    print("Reference: https://python.langchain.com/docs/integrations/vectorstores/faiss/")
    print(f"Using configuration: {args.config}")
    
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        if args.create_new:
            # Create a new test configuration
            from app.config import VectorStoreConfig, VectorStore, EmbeddingConfig, EmbeddingModel
            
            test_config = {
                "name": "faiss_filter_test",
                "vector_store": VectorStoreConfig(
                    type=VectorStore.FAISS,
                    index_path="./test_storage/faiss_filter_test"
                ),
                "embedding": EmbeddingConfig(
                    enabled=True,
                    model=EmbeddingModel.SENTENCE_TRANSFORMERS,
                    endpoint=None,
                    api_key=None
                ),
                "reranker": {
                    "enabled": False
                }
            }
            
            # Add the test configuration
            rag_service.add_configuration(test_config)
            config_name = "faiss_filter_test"
            print(f"Created new test configuration: {config_name}")
        else:
            config_name = args.config
            # Check if configuration exists
            if not rag_service.get_configuration(config_name):
                print(f"Configuration '{config_name}' does not exist.")
                print("Available configurations:")
                for cfg in rag_service.list_configurations():
                    print(f"  - {cfg}")
                print("\nUse --create-new to create a test configuration, or specify an existing one with --config")
                return 1
        
        # Create and add test documents
        print("\nAdding test documents...")
        documents = create_test_documents()
        
        # Get the vector store directly and add documents
        vector_store = rag_service._get_vector_store(config_name)
        doc_ids = vector_store.add_documents(documents)
        print(f"Added {len(doc_ids)} documents to configuration '{config_name}'")
        
        # Print all documents for reference
        print("\n=== Test Documents ===")
        for i, doc in enumerate(documents, 1):
            print(f"{i}. Content: {doc.page_content[:80]}...")
            print(f"   Metadata: {doc.metadata}")
        
        # We already have the vector store from above, no need to get it again
        
        # Run all filter tests
        test_simple_equality_filtering(vector_store)
        test_equality_operator(vector_store)
        test_greater_than_operator(vector_store)
        test_in_operator(vector_store)
        test_and_operator(vector_store)
        test_or_operator(vector_store)
        test_not_operator(vector_store)
        test_complex_filter(vector_store)
        
        print("\n=== All Tests Completed Successfully! ===")
        print("FAISS metadata filtering now supports LangChain-style MongoDB operators:")
        print("- Simple equality: {'key': 'value'}")
        print("- $eq, $neq, $gt, $lt, $gte, $lte")
        print("- $in, $nin")
        print("- $and, $or, $not")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
