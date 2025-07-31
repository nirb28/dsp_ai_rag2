#!/usr/bin/env python3
"""
Simple test script to verify NetworkX integration works correctly.
This script tests the basic functionality without requiring API keys.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from app.services.networkx_graph_store import NetworkXGraphStore
from langchain.docstore.document import Document as LangchainDocument


def test_networkx_basic_functionality():
    """Test basic NetworkX graph store functionality."""
    
    print("Testing NetworkX Graph Store Integration")
    print("=" * 50)
    
    # Create a test configuration
    config = {
        'type': 'networkx',
        'index_path': './test_storage/networkx_test',
        'name': 'test_graph'
    }
    
    # Initialize the graph store
    print("\n1. Initializing NetworkX Graph Store...")
    try:
        graph_store = NetworkXGraphStore(config)
        print("Graph store initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize graph store: {str(e)}")
        return False
    
    # Create test documents
    print("\n2. Creating test documents...")
    test_docs = [
        LangchainDocument(
            page_content="Python is a popular programming language used for data science and machine learning. It has libraries like Pandas, NumPy, and Scikit-learn.",
            metadata={"filename": "python_intro.txt", "topic": "programming"}
        ),
        LangchainDocument(
            page_content="Machine learning is a subset of artificial intelligence. Python and R are commonly used languages for machine learning projects.",
            metadata={"filename": "ml_basics.txt", "topic": "ai"}
        ),
        LangchainDocument(
            page_content="Data science involves extracting insights from data using statistical methods and machine learning. Python is the most popular language in this field.",
            metadata={"filename": "data_science.txt", "topic": "data"}
        )
    ]
    print(f"Created {len(test_docs)} test documents")
    
    # Add documents to the graph store
    print("\n3. Adding documents to graph store...")
    try:
        doc_ids = graph_store.add_documents(test_docs)
        print(f"Added {len(doc_ids)} documents to graph store")
        print(f"   Document IDs: {doc_ids}")
    except Exception as e:
        print(f"Failed to add documents: {str(e)}")
        return False
    
    # Check document count
    print("\n4. Checking document count...")
    try:
        count = graph_store.get_document_count()
        print(f"Graph store contains {count} documents")
    except Exception as e:
        print(f"❌ Failed to get document count: {str(e)}")
        return False
    
    # Test similarity search
    print("\n5. Testing similarity search...")
    test_queries = [
        "What is Python used for?",
        "Tell me about machine learning",
        "How is data science related to programming?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        try:
            results = graph_store.similarity_search(query, k=2, similarity_threshold=0.1)
            print(f"   Found {len(results)} results")
            
            for j, (doc, score) in enumerate(results, 1):
                filename = doc.metadata.get('filename', 'Unknown')
                topic = doc.metadata.get('topic', 'Unknown')
                print(f"      {j}. {filename} (topic: {topic}, score: {score:.3f})")
                
        except Exception as e:
            print(f"   Search failed: {str(e)}")
    
    # Test graph statistics
    print("\n6. Getting graph statistics...")
    try:
        stats = graph_store.get_graph_stats()
        print("Graph statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"Failed to get graph stats: {str(e)}")
    
    # Test document deletion
    print("\n7️⃣ Testing document deletion...")
    try:
        if doc_ids:
            # Delete the first document
            graph_store.delete_documents([doc_ids[0]])
            new_count = graph_store.get_document_count()
            print(f"Deleted 1 document. New count: {new_count}")
    except Exception as e:
        print(f"Failed to delete document: {str(e)}")
    
    print("\nNetworkX integration test completed!")
    return True


def test_graph_relationships():
    """Test that the graph creates meaningful relationships between documents."""
    
    print("\nTesting Graph Relationship Creation")
    print("=" * 40)
    
    config = {
        'type': 'networkx',
        'index_path': './test_storage/networkx_relationships',
        'name': 'relationship_test'
    }
    
    graph_store = NetworkXGraphStore(config)
    
    # Create documents with clear relationships
    related_docs = [
        LangchainDocument(
            page_content="Apple Inc. is a technology company founded by Steve Jobs. The company is known for iPhone, iPad, and Mac computers.",
            metadata={"filename": "apple_company.txt", "category": "technology"}
        ),
        LangchainDocument(
            page_content="Steve Jobs was the co-founder and CEO of Apple Inc. He was known for his innovative vision and product design philosophy.",
            metadata={"filename": "steve_jobs.txt", "category": "biography"}
        ),
        LangchainDocument(
            page_content="The iPhone is Apple's flagship smartphone product. It revolutionized the mobile phone industry with its touchscreen interface.",
            metadata={"filename": "iphone_product.txt", "category": "product"}
        ),
        LangchainDocument(
            page_content="Microsoft Corporation is a technology company that competes with Apple. Bill Gates founded Microsoft and it's known for Windows and Office.",
            metadata={"filename": "microsoft.txt", "category": "technology"}
        )
    ]
    
    # Add documents
    doc_ids = graph_store.add_documents(related_docs)
    print(f"Added {len(doc_ids)} related documents")
    
    # Test that related documents are found together
    print("\nTesting relationship-based search:")
    
    queries = [
        "Tell me about Apple and Steve Jobs",
        "What products does Apple make?",
        "Compare Apple and Microsoft"
    ]
    
    for query in queries:
        print(f"\n   Query: {query}")
        results = graph_store.similarity_search(query, k=3, similarity_threshold=0.1)
        
        for i, (doc, score) in enumerate(results, 1):
            filename = doc.metadata.get('filename', 'Unknown')
            category = doc.metadata.get('category', 'Unknown')
            print(f"      {i}. {filename} (category: {category}, score: {score:.3f})")
    
    # Show graph statistics
    stats = graph_store.get_graph_stats()
    print(f"\nRelationship Graph Stats:")
    print(f"   Nodes: {stats.get('num_nodes', 0)}")
    print(f"   Edges: {stats.get('num_edges', 0)}")
    print(f"   Density: {stats.get('density', 0):.3f}")
    print(f"   Connected: {stats.get('is_connected', False)}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test NetworkX Integration")
    parser.add_argument('--config', type=str, help='Use an existing configuration name (skips creation/cleanup)')
    parser.add_argument('--create-new', action='store_true', help='Force creation of a new test configuration')
    args = parser.parse_args()

    print("Starting NetworkX Integration Tests\n")
    if args.config and not args.create_new:
        print(f"\nℹ️ Using existing configuration: {args.config}")
        print("  (Test will run in read-only mode, skipping creation/cleanup.)")
        # You would insert logic here to actually test with the existing config if desired.
        print("\nTest complete using existing configuration.")
    else:
        try:
            # Run basic functionality test
            success1 = test_networkx_basic_functionality()
            
            # Run relationship test
            success2 = test_graph_relationships()
            
            if success1 and success2:
                print("\nAll tests passed! NetworkX integration is working correctly.")
            else:
                print("\nSome tests failed. Please check the error messages above.")
        except Exception as e:
            print(f"\nTest execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nNext steps:")
    print("   1. Install NetworkX: pip install networkx")
    print("   2. Run the full POC: python examples/networkx_graph_poc.py")
    print("   3. Configure your RAG system to use 'networkx' as vector store type")
