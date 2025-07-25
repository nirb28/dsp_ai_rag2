#!/usr/bin/env python3
"""
NetworkX Graph Store POC Example

This script demonstrates how to use the NetworkX graph store as an alternative
to traditional vector stores in the RAG system.

The graph store creates relationships between documents based on:
- Shared entities (proper nouns, names, etc.)
- Shared keywords (significant terms)
- Graph structure (centrality, connectivity)

Usage:
    python examples/networkx_graph_poc.py
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.rag_service import RAGService
from app.config import RAGConfig, VectorStoreConfig, EmbeddingConfig, GenerationConfig, RerankerConfig
from app.models import QueryRequest, DocumentUploadRequest


async def create_networkx_configuration():
    """Create a RAG configuration that uses NetworkX graph store."""
    
    config = RAGConfig(
        vector_store=VectorStoreConfig(
            type="networkx",  # Use NetworkX graph store
            index_path="./storage/networkx_poc",
            dimension=384  # Not used by NetworkX but required for config validation
        ),
        embedding=EmbeddingConfig(
            enabled=False,  # NetworkX doesn't require embeddings for basic functionality
            model="sentence-transformers/all-MiniLM-L6-v2",
            endpoint="http://localhost:9001"
        ),
        generation=GenerationConfig(
            enabled=True,
            model="meta/llama-3.3-70b-instruct",
            api_key="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",
            endpoint="https://integrate.api.nvidia.com",
            max_tokens=1000,
            temperature=0.7
        ),
        reranker=RerankerConfig(
            enabled=False,  # Disable reranking for simplicity
            model="none"
        ),
        retrieval={"k": 5, "similarity_threshold": 0.3}
    )
    
    return config


def create_sample_documents():
    """Create sample documents to demonstrate graph relationships."""
    
    documents = [
        {
            "content": """
            Machine Learning is a subset of Artificial Intelligence that enables computers to learn 
            and make decisions without being explicitly programmed. It involves algorithms that can 
            identify patterns in data and make predictions. Popular techniques include Neural Networks, 
            Decision Trees, and Support Vector Machines. Companies like Google, Microsoft, and Amazon 
            use machine learning extensively in their products.
            """,
            "filename": "ml_intro.txt",
            "metadata": {"topic": "machine_learning", "difficulty": "beginner"}
        },
        {
            "content": """
            Neural Networks are computing systems inspired by biological neural networks. They consist 
            of interconnected nodes called neurons that process information. Deep Learning is a subset 
            of machine learning that uses multi-layered neural networks. TensorFlow and PyTorch are 
            popular frameworks for building neural networks. Google developed TensorFlow while Facebook 
            created PyTorch.
            """,
            "filename": "neural_networks.txt",
            "metadata": {"topic": "deep_learning", "difficulty": "intermediate"}
        },
        {
            "content": """
            Natural Language Processing (NLP) is a branch of Artificial Intelligence that helps computers 
            understand, interpret, and manipulate human language. It combines computational linguistics 
            with machine learning and deep learning models. Applications include chatbots, translation 
            services, and sentiment analysis. OpenAI's GPT models and Google's BERT are breakthrough 
            NLP technologies.
            """,
            "filename": "nlp_overview.txt",
            "metadata": {"topic": "nlp", "difficulty": "intermediate"}
        },
        {
            "content": """
            Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, 
            and systems to extract knowledge from structured and unstructured data. It combines statistics, 
            machine learning, and domain expertise. Python and R are popular programming languages for 
            data science. Tools like Jupyter Notebooks, Pandas, and Scikit-learn are commonly used.
            """,
            "filename": "data_science.txt",
            "metadata": {"topic": "data_science", "difficulty": "beginner"}
        },
        {
            "content": """
            Computer Vision is a field of Artificial Intelligence that trains computers to interpret 
            and understand visual information from the world. It uses machine learning algorithms to 
            identify objects, faces, and scenes in images and videos. Convolutional Neural Networks 
            are particularly effective for computer vision tasks. Applications include autonomous 
            vehicles, medical imaging, and facial recognition systems.
            """,
            "filename": "computer_vision.txt",
            "metadata": {"topic": "computer_vision", "difficulty": "advanced"}
        }
    ]
    
    return documents


async def demonstrate_graph_store():
    """Demonstrate the NetworkX graph store functionality."""
    
    print("NetworkX Graph Store POC")
    print("=" * 50)
    
    # Initialize RAG service
    rag_service = RAGService()
    
    # Create NetworkX configuration
    print("\nCreating NetworkX configuration...")
    # config = await create_networkx_configuration()
    config = rag_service.get_configuration("networkx_poc")
    rag_service.set_configuration("networkx_poc", config)
    print("Configuration created successfully!")
    
    # Upload sample documents
    print("\nUploading sample documents...")
    documents = create_sample_documents()
    
    for i, doc in enumerate(documents, 1):
        print(f"   Uploading document {i}/5: {doc['filename']}")
        result = await rag_service.upload_text_content(
            content=doc["content"],
            filename=doc["filename"],
            configuration_name="networkx_poc",
            metadata=doc["metadata"]
        )
        print(f"   ✅ Document uploaded with ID: {result.id}")
    
    # Get graph statistics
    print("\n📊 Graph Statistics:")
    vector_store = rag_service._get_vector_store("networkx_poc")
    if hasattr(vector_store, 'get_graph_stats'):
        stats = vector_store.get_graph_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # Demonstrate graph-based queries
    print("\n🔍 Testing Graph-Based Queries:")
    print("-" * 30)
    
    test_queries = [
        "What is machine learning?",
        "Tell me about neural networks",
        "How does Google use AI?",
        "What programming languages are used in data science?",
        "Explain computer vision applications"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        # Perform retrieval
        try:
            documents, metadata = await rag_service.retrieve(
                query=query,
                configuration_name="networkx_poc",
                k=3
            )
            
            print(f"   📋 Found {len(documents)} relevant documents:")
            for j, doc in enumerate(documents, 1):
                score = doc.get('similarity_score', 0)
                filename = doc.get('metadata', {}).get('filename', 'Unknown')
                topic = doc.get('metadata', {}).get('topic', 'Unknown')
                print(f"      {j}. {filename} (topic: {topic}, score: {score:.3f})")
                
                # Show a snippet of the content
                content_snippet = doc['content'][:100].replace('\n', ' ').strip()
                print(f"         Preview: {content_snippet}...")
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Demonstrate query with generation
    print("\n💬 Testing Query with Generation:")
    print("-" * 35)
    
    if config.generation.api_key:
        try:
            query = "What are the main applications of artificial intelligence?"
            print(f"🔍 Query: {query}")
            
            response = await rag_service.query(
                query=query,
                configuration_name="networkx_poc",
                k=3
            )
            
            print(f"\n📋 Retrieved {len(response.documents)} documents")
            print(f"💡 Generated Response:")
            print(f"   {response.response}")
            
        except Exception as e:
            print(f"❌ Generation error: {str(e)}")
            print("Tip: Set GROQ_API_KEY environment variable for generation")
    else:
        print("Skipping generation (no API key provided)")
        print("Tip: Set GROQ_API_KEY environment variable for generation")
    
    print("\nNetworkX Graph Store POC completed!")
    print("\nKey Benefits of Graph Store:")
    print("   - Captures semantic relationships between documents")
    print("   - Enables discovery through graph traversal")
    print("   - Provides interpretable similarity scoring")
    print("   - Supports complex queries through graph algorithms")
    print("   - No dependency on vector embeddings")


if __name__ == "__main__":
    asyncio.run(demonstrate_graph_store())
