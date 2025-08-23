"""
Elasticsearch Multi-Search Example for DSP AI RAG2 Project

This example demonstrates how to use Elasticsearch with three different search strategies:
1. Fulltext search - Traditional BM25-based text search
2. Vector search - Embedding-based similarity search using custom embeddings  
3. Semantic search - Elasticsearch's built-in semantic search with ELSER model
4. Hybrid search - Combination of all three search strategies

Prerequisites:
- Elasticsearch server running on localhost:9200 (or configure different URL)
- Optional: ELSER model deployed for semantic search
- RAG configuration with Elasticsearch vector store
"""

import asyncio
import json
import logging
from pathlib import Path
from langchain.docstore.document import Document as LangchainDocument

from app.config import RAGConfig, VectorStore
from app.services.rag_service import RAGService
from app.services.elasticsearch_vector_store import ElasticsearchSearchType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to parse data, learn from it, and make predictions or decisions.",
        "metadata": {"category": "technology", "topic": "machine_learning", "difficulty": "beginner"}
    },
    {
        "content": "Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized areas like computer vision and natural language processing.",
        "metadata": {"category": "technology", "topic": "deep_learning", "difficulty": "intermediate"}
    },
    {
        "content": "Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms that can understand, interpret, and generate human language.",
        "metadata": {"category": "technology", "topic": "nlp", "difficulty": "intermediate"}
    },
    {
        "content": "Computer vision is an interdisciplinary field that deals with how computers can be made to gain high-level understanding from digital images or videos. It seeks to automate tasks that the human visual system can do.",
        "metadata": {"category": "technology", "topic": "computer_vision", "difficulty": "intermediate"}
    },
    {
        "content": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward. It's inspired by behaviorist psychology.",
        "metadata": {"category": "technology", "topic": "reinforcement_learning", "difficulty": "advanced"}
    },
    {
        "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
        "metadata": {"category": "technology", "topic": "data_science", "difficulty": "beginner"}
    },
    {
        "content": "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for rapid application development.",
        "metadata": {"category": "programming", "topic": "python", "difficulty": "beginner"}
    },
    {
        "content": "Neural networks are computing systems inspired by biological neural networks. They consist of nodes (neurons) connected by edges (synapses) that can transmit signals between them.",
        "metadata": {"category": "technology", "topic": "neural_networks", "difficulty": "intermediate"}
    }
]

def create_elasticsearch_config():
    """Create a RAG configuration for Elasticsearch with sample settings."""
    config_data = {
        "vector_store": {
            "type": "elasticsearch",
            "es_url": "http://localhost:9200",
            "es_index_name": "rag_multi_search_demo",
            "es_user": None,  # Set if authentication is required
            "es_password": None,  # Set if authentication is required
            "es_api_key": None,  # Alternative to username/password
            "es_use_index_suffix": False
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 32
        },
        "generation": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 500
        }
    }
    
    return RAGConfig(**config_data)

async def demonstrate_search_types():
    """Demonstrate all four search types with sample queries."""
    try:
        # Create configuration
        config = create_elasticsearch_config()
        
        # Initialize RAG service
        rag_service = RAGService(config, "elasticsearch_demo")
        
        # Add sample documents
        logger.info("Adding sample documents to Elasticsearch...")
        documents = [
            LangchainDocument(page_content=doc["content"], metadata=doc["metadata"])
            for doc in SAMPLE_DOCUMENTS
        ]
        
        document_ids = rag_service.add_documents(documents)
        logger.info(f"Added {len(document_ids)} documents to Elasticsearch")
        
        # Test queries
        test_queries = [
            "What is machine learning and AI?",
            "deep neural networks computer vision",
            "Python programming language features",
            "reinforcement learning algorithms"
        ]
        
        # Demonstrate each search type
        for query in test_queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"Query: '{query}'")
            logger.info(f"{'='*60}")
            
            # 1. Fulltext Search (BM25)
            logger.info("\n--- FULLTEXT SEARCH (BM25) ---")
            fulltext_results = await demonstrate_fulltext_search(rag_service, query)
            
            # 2. Vector Search (Embeddings)
            logger.info("\n--- VECTOR SEARCH (Embeddings) ---")
            vector_results = await demonstrate_vector_search(rag_service, query)
            
            # 3. Semantic Search (ELSER)
            logger.info("\n--- SEMANTIC SEARCH (ELSER) ---")
            semantic_results = await demonstrate_semantic_search(rag_service, query)
            
            # 4. Hybrid Search (Combined)
            logger.info("\n--- HYBRID SEARCH (Combined) ---")
            hybrid_results = await demonstrate_hybrid_search(rag_service, query)
            
            # Compare results
            logger.info("\n--- SEARCH COMPARISON ---")
            compare_search_results(query, {
                "fulltext": fulltext_results,
                "vector": vector_results, 
                "semantic": semantic_results,
                "hybrid": hybrid_results
            })
        
        # Demonstrate filtering
        logger.info(f"\n{'='*60}")
        logger.info("Demonstrating metadata filtering")
        logger.info(f"{'='*60}")
        await demonstrate_filtering(rag_service)
        
        logger.info("\nElasticsearch multi-search demonstration completed!")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        raise

async def demonstrate_fulltext_search(rag_service: RAGService, query: str):
    """Demonstrate fulltext search using BM25."""
    try:
        # Get vector store and cast to ElasticsearchVectorStore
        vector_store = rag_service._get_vector_store()
        
        # Perform fulltext search
        results = vector_store.similarity_search(
            query=query,
            k=3,
            search_type=ElasticsearchSearchType.FULLTEXT
        )
        
        logger.info("Fulltext search results:")
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"{i}. Score: {score:.3f}")
            logger.info(f"   Content: {doc.page_content[:100]}...")
            logger.info(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in fulltext search: {str(e)}")
        return []

async def demonstrate_vector_search(rag_service: RAGService, query: str):
    """Demonstrate vector search using embeddings."""
    try:
        # Get vector store and cast to ElasticsearchVectorStore  
        vector_store = rag_service._get_vector_store()
        
        # Perform vector search
        results = vector_store.similarity_search(
            query=query,
            k=3,
            search_type=ElasticsearchSearchType.VECTOR
        )
        
        logger.info("Vector search results:")
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"{i}. Score: {score:.3f}")
            logger.info(f"   Content: {doc.page_content[:100]}...")
            logger.info(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        return []

async def demonstrate_semantic_search(rag_service: RAGService, query: str):
    """Demonstrate semantic search using ELSER."""
    try:
        # Get vector store and cast to ElasticsearchVectorStore
        vector_store = rag_service._get_vector_store()
        
        # Perform semantic search
        results = vector_store.similarity_search(
            query=query,
            k=3,
            search_type=ElasticsearchSearchType.SEMANTIC
        )
        
        logger.info("Semantic search results:")
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"{i}. Score: {score:.3f}")
            logger.info(f"   Content: {doc.page_content[:100]}...")
            logger.info(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.warning(f"Semantic search not available: {str(e)}")
        logger.info("Note: Semantic search requires ELSER model deployment in Elasticsearch")
        return []

async def demonstrate_hybrid_search(rag_service: RAGService, query: str):
    """Demonstrate hybrid search combining all strategies."""
    try:
        # Get vector store and cast to ElasticsearchVectorStore
        vector_store = rag_service._get_vector_store()
        
        # Perform hybrid search
        results = vector_store.similarity_search(
            query=query,
            k=3,
            search_type=ElasticsearchSearchType.HYBRID
        )
        
        logger.info("Hybrid search results:")
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"{i}. Score: {score:.3f}")
            logger.info(f"   Content: {doc.page_content[:100]}...")
            logger.info(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        return []

async def demonstrate_filtering(rag_service: RAGService):
    """Demonstrate metadata filtering with different search types."""
    try:
        vector_store = rag_service._get_vector_store()
        
        # Filter by category
        logger.info("\nSearching for 'learning' in technology category:")
        results = vector_store.similarity_search(
            query="learning algorithms",
            k=5,
            filter={"category": "technology"},
            search_type=ElasticsearchSearchType.VECTOR
        )
        
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"{i}. {doc.metadata.get('topic', 'N/A')} (score: {score:.3f})")
        
        # Filter by difficulty
        logger.info("\nSearching for beginner-level content:")
        results = vector_store.similarity_search(
            query="programming and data",
            k=5,
            filter={"difficulty": "beginner"},
            search_type=ElasticsearchSearchType.FULLTEXT
        )
        
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"{i}. {doc.metadata.get('topic', 'N/A')} (score: {score:.3f})")
            
    except Exception as e:
        logger.error(f"Error in filtering demonstration: {str(e)}")

def compare_search_results(query: str, results_dict: dict):
    """Compare results from different search strategies."""
    logger.info(f"Result comparison for query: '{query}'")
    logger.info("-" * 50)
    
    for search_type, results in results_dict.items():
        if results:
            top_topic = results[0][1] if results else "N/A"
            logger.info(f"{search_type.upper():12}: {len(results)} results, top score: {top_topic:.3f}")
        else:
            logger.info(f"{search_type.upper():12}: No results")

if __name__ == "__main__":
    print("Elasticsearch Multi-Search Demonstration")
    print("========================================")
    print()
    print("This script demonstrates:")
    print("1. Fulltext search using BM25 algorithm")
    print("2. Vector search using embeddings")
    print("3. Semantic search using ELSER (if available)")
    print("4. Hybrid search combining all strategies")
    print("5. Metadata filtering with all search types")
    print()
    print("Prerequisites:")
    print("- Elasticsearch server running on localhost:9200")
    print("- Optional: ELSER model for semantic search")
    print()
    
    # Run the demonstration
    asyncio.run(demonstrate_search_types())
