#!/usr/bin/env python
"""
Test script for demonstrating reranking and context injection with the RAG API.
This script:
1. Configures a collection with reranking and context injection
2. Makes queries with context items simulating a conversation history
3. Compares results with and without these features

Usage:
    python reranking_and_context.py
"""

import asyncio
import json
import time
import requests
import sys
from typing import List, Dict, Any, Optional

# API URL - update if your service runs on a different port
BASE_URL = "http://localhost:8000"
# API route prefix
API_PREFIX = "/api/v1"

async def main():
    configuration_name = "rag_enhanced_test"
    print(f"Testing reranking and context injection with configuration: {configuration_name}")
    
    # Step 1: Apply a configuration preset with features enabled
    print("\n1. Setting up collection with reranking and context injection...")
    try:
        # First, let's check if the collection exists
        configurations = requests.get(f"{BASE_URL}/configurations").json()
        collection_exists = any(c["name"] == configuration_name for c in configurations.get("configurations", []))
        
        # Configure the collection
        config = {
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "batch_size": 32
            },
            "vector_store": {
                "type": "faiss",
                "index_type": "l2"
            },
            "chunking": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "add_document_metadata": True
            },
            "generation": {
                "model": "llama3-8b-8192",
                "temperature": 0.7,
                "prompt_template": "Answer the question based on the following context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                "model_kwargs": {
                    "max_tokens": 500
                }
            },
            "reranking": {
                "enabled": True,
                "model": "cross-encoder",
                "top_n": 5,
                "score_threshold": 0.5
            },
            "context_injection": {
                "enabled": True,
                "prefix": "Based on our conversation: ",
                "position": "before",
                "separator": " ",
                "max_items": 5
            }
        }
        
        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/configure_collection",
            json={"configuration_name": configuration_name, "config": config}
        ).json()
        
        print(f"Configuration set: {response.get('message')}")
        
        # If collection is new, upload a sample document
        if not collection_exists:
            print("Creating sample document for testing...")
            with open("examples/sample_content.txt", "w") as f:
                f.write("""
                # AI and Machine Learning
                Artificial intelligence (AI) is intelligence demonstrated by machines.
                
                ## Large Language Models
                Large Language Models (LLMs) are AI systems trained on vast amounts of text data.
                They can generate human-like text, answer questions, and perform various language tasks.
                
                ## Retrieval Augmented Generation
                Retrieval Augmented Generation (RAG) is a technique that enhances language models
                by retrieving relevant information from external sources before generating responses.
                
                ## Reranking in Search
                Reranking is the process of taking search results and reordering them based on
                additional criteria to improve relevance. This is especially useful in RAG systems.
                
                ## Context Window and Injection
                Context window refers to the amount of text a language model can process at once.
                Context injection involves adding specific contextual information to guide the model's responses.
                """)
                
            # Upload the sample content
            files = {'file': open('examples/sample_content.txt', 'rb')}
            data = {'configuration_name': configuration_name}
            upload_response = requests.post(f"{BASE_URL}{API_PREFIX}/upload", files=files, data=data).json()
            print(f"Uploaded document: {upload_response.get('message')}")
            time.sleep(2)  # Give time for processing to complete
    except Exception as e:
        print(f"Error during setup: {e}")
        return
    
    # Step 2: Make queries with and without context
    print("\n2. Testing queries with context injection...")
    
    # Define some conversation history as context items
    context_items = [
        {"content": "Tell me about AI and machine learning", "role": "user"},
        {"content": "Artificial Intelligence refers to the simulation of human intelligence in machines. Machine learning is a subset of AI that allows systems to learn from data.", "role": "assistant"},
        {"content": "What about large language models?", "role": "user"}
    ]
    
    # Query without context injection
    print("\nQuery WITHOUT context injection:")
    query_no_context = "How do they relate to context windows?"
    response_no_context = requests.post(
        f"{BASE_URL}{API_PREFIX}/query",
        json={"query": query_no_context, "configuration_name": configuration_name}
    ).json()
    
    print(f"Query: {query_no_context}")
    print(f"Answer: {response_no_context.get('answer')}")
    
    # Query with context injection
    print("\nQuery WITH context injection:")
    query_with_context = "How do they relate to context windows?"
    response_with_context = requests.post(
        f"{BASE_URL}{API_PREFIX}/query",
        json={
            "query": query_with_context,
            "configuration_name": configuration_name,
            "context_items": context_items
        }
    ).json()
    
    print(f"Query: {query_with_context}")
    print(f"Answer: {response_with_context.get('answer')}")
    
    # Step 3: Demonstrate reranking impact
    print("\n3. Testing effect of reranking...")
    
    # Temporarily disable reranking for comparison
    config["reranking"]["enabled"] = False
    response = requests.post(
        f"{BASE_URL}/configure",
        json={"configuration_name": configuration_name, "config": config}
    ).json()
    
    # Query without reranking
    print("\nQuery WITHOUT reranking:")
    query = "What is retrieval augmented generation?"
    response_no_reranking = requests.post(
        f"{BASE_URL}{API_PREFIX}/query",
        json={"query": query, "configuration_name": configuration_name}
    ).json()
    
    print(f"Query: {query}")
    print(f"Answer: {response_no_reranking.get('answer')}")
    
    # Re-enable reranking
    config["reranking"]["enabled"] = True
    response = requests.post(
        f"{BASE_URL}/configure",
        json={"configuration_name": configuration_name, "config": config}
    ).json()
    
    # Query with reranking
    print("\nQuery WITH reranking:")
    response_with_reranking = requests.post(
        f"{BASE_URL}{API_PREFIX}/query",
        json={"query": query, "configuration_name": configuration_name}
    ).json()
    
    print(f"Query: {query}")
    print(f"Answer: {response_with_reranking.get('answer')}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(main())
