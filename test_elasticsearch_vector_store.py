#!/usr/bin/env python3
"""
Test script for Elasticsearch multi-search vector store functionality.

This script validates the enhanced Elasticsearch vector store implementation by:
1. Creating a test configuration with Elasticsearch vector store
2. Adding sample documents to the store
3. Testing all search types: fulltext, vector, semantic, and hybrid
4. Performing searches with various metadata filters
5. Testing document deletion and counting
6. Validating MongoDB-style filtering capabilities
7. Comparing performance of different search strategies

Usage:
    # Create new configuration with username/password auth
    python test_elasticsearch_vector_store.py --config test-elasticsearch --create-new
    
    # Create new configuration with API key auth
    python test_elasticsearch_vector_store.py --config test-elasticsearch --create-new --use-api-key
    
    # Create configuration without index suffix (use exact index name)
    python test_elasticsearch_vector_store.py --config test-elasticsearch --create-new --no-index-suffix
    
    # Test existing configuration
    python test_elasticsearch_vector_store.py --config test-elasticsearch

Prerequisites:
- Elasticsearch server running on localhost:9200 (or configure elasticsearch_url)
- Required dependencies installed (elasticsearch, langchain-community)
- For API key auth: Set your API key in the create_test_configuration function
- Optional: ELSER model deployed for semantic search testing
"""

import asyncio
import argparse
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from app.config import RAGConfig, VectorStoreConfig, EmbeddingConfig, VectorStore, EmbeddingModel
from app.services.rag_service import RAGService
from app.services.elasticsearch_vector_store import ElasticsearchSearchType
from langchain.docstore.document import Document as LangchainDocument

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_configuration(config_name: str, use_api_key: bool = False, use_index_suffix: bool = True) -> RAGConfig:
    """Create a test configuration for Elasticsearch vector store."""
    logger.info(f"Creating test configuration: {config_name}")
    
    # Configure authentication based on preference
    if use_api_key:
        # API key authentication - set your API key here
        es_api_key = None  # Set your API key here, e.g., "your-api-key"
        es_api_key_id = None  # Optional: set API key ID for identification
        es_user = None
        es_password = None
        auth_method = "API key"
    else:
        # Username/password authentication
        es_api_key = None
        es_api_key_id = None
        es_user = None  # Set if authentication is required
        es_password = None  # Set if authentication is required
        auth_method = "username/password"
    
    logger.info(f"Using {auth_method} authentication for Elasticsearch")
    logger.info(f"Index suffix enabled: {use_index_suffix}")
    
    config = RAGConfig(
        vector_store=VectorStoreConfig(
            type=VectorStore.ELASTICSEARCH,
            dimension=384,  # Dimension for sentence-transformers/all-MiniLM-L6-v2
            es_url="http://localhost:9200",
            es_index_name="test-documents",
            es_user=es_user,
            es_password=es_password,
            es_api_key=es_api_key,
            es_api_key_id=es_api_key_id,
            es_use_index_suffix=use_index_suffix
        ),
        embedding=EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM,
            batch_size=32
        ),
        retrieval_k=5,
        similarity_threshold=0.7
    )
    
    return config

def create_sample_documents() -> List[LangchainDocument]:
    """Create sample documents for testing."""
    documents = [
        LangchainDocument(
            page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            metadata={"source": "ml_basics", "category": "technology", "level": "beginner", "topic": "machine_learning"}
        ),
        LangchainDocument(
            page_content="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            metadata={"source": "dl_guide", "category": "technology", "level": "intermediate", "topic": "deep_learning"}
        ),
        LangchainDocument(
            page_content="Natural language processing enables computers to understand, interpret, and generate human language.",
            metadata={"source": "nlp_intro", "category": "technology", "level": "intermediate", "topic": "nlp"}
        ),
        LangchainDocument(
            page_content="Python is a popular programming language for data science and machine learning applications.",
            metadata={"source": "python_guide", "category": "programming", "level": "beginner", "topic": "python"}
        ),
        LangchainDocument(
            page_content="Data visualization helps in understanding patterns and insights from complex datasets.",
            metadata={"source": "viz_tutorial", "category": "data_science", "level": "beginner", "topic": "visualization"}
        ),
        LangchainDocument(
            page_content="Cloud computing provides scalable infrastructure for machine learning workloads.",
            metadata={"source": "cloud_ml", "category": "technology", "level": "advanced", "topic": "cloud"}
        )
    ]
    
    return documents

async def test_elasticsearch_vector_store(config_name: str, create_new: bool = False, use_api_key: bool = False, use_index_suffix: bool = True):
    """Test Elasticsearch vector store functionality."""
    
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        if create_new:
            # Create and save test configuration
            config = create_test_configuration(config_name, use_api_key, use_index_suffix)
            await rag_service.create_configuration(config_name, config)
            logger.info(f"Created new configuration: {config_name}")
        
        # Load configuration
        config = await rag_service.get_configuration(config_name)
        if not config:
            raise ValueError(f"Configuration '{config_name}' not found. Use --create-new to create it.")
        
        logger.info(f"Using configuration: {config_name}")
        logger.info(f"Vector store type: {config.vector_store.type}")
        logger.info(f"Elasticsearch URL: {config.vector_store.es_url}")
        logger.info(f"Elasticsearch index: {config.vector_store.es_index_name}")
        
        # Test 1: Add documents
        logger.info("\n=== Test 1: Adding Documents ===")
        sample_docs = create_sample_documents()
        
        result = await rag_service.add_documents(config_name, sample_docs)
        logger.info(f"Added {len(result['document_ids'])} documents")
        logger.info(f"Document IDs: {result['document_ids'][:3]}...")  # Show first 3 IDs
        
        # Test 2: Basic similarity search
        logger.info("\n=== Test 2: Basic Similarity Search ===")
        query = "What is machine learning?"
        
        search_result = await rag_service.retrieve(config_name, query, k=3)
        logger.info(f"Query: '{query}'")
        logger.info(f"Found {len(search_result['documents'])} documents:")
        
        for i, doc in enumerate(search_result['documents'][:3]):
            logger.info(f"  {i+1}. Score: {doc.get('similarity_score', 'N/A'):.3f}")
            logger.info(f"     Content: {doc['content'][:100]}...")
            logger.info(f"     Metadata: {doc['metadata']}")
        
        # Test 3: Filtered search - category filter
        logger.info("\n=== Test 3: Filtered Search (Category) ===")
        filter_metadata = {"category": "technology"}
        
        filtered_result = await rag_service.retrieve(
            config_name, 
            query, 
            k=5, 
            filter_metadata=filter_metadata
        )
        logger.info(f"Query: '{query}' with filter: {filter_metadata}")
        logger.info(f"Found {len(filtered_result['documents'])} documents:")
        
        for i, doc in enumerate(filtered_result['documents']):
            logger.info(f"  {i+1}. Score: {doc.get('similarity_score', 'N/A'):.3f}")
            logger.info(f"     Category: {doc['metadata'].get('category')}")
            logger.info(f"     Content: {doc['content'][:80]}...")
        
        # Test 4: Complex filter with MongoDB-style operators
        logger.info("\n=== Test 4: Complex MongoDB-style Filtering ===")
        complex_filter = {
            "$and": [
                {"category": {"$in": ["technology", "programming"]}},
                {"level": {"$neq": "advanced"}}
            ]
        }
        
        complex_result = await rag_service.retrieve(
            config_name,
            "programming and algorithms",
            k=5,
            filter_metadata=complex_filter
        )
        logger.info(f"Query: 'programming and algorithms' with complex filter: {json.dumps(complex_filter, indent=2)}")
        logger.info(f"Found {len(complex_result['documents'])} documents:")
        
        for i, doc in enumerate(complex_result['documents']):
            logger.info(f"  {i+1}. Score: {doc.get('similarity_score', 'N/A'):.3f}")
            logger.info(f"     Category: {doc['metadata'].get('category')}, Level: {doc['metadata'].get('level')}")
            logger.info(f"     Content: {doc['content'][:80]}...")
        
        # Test 5: Document count
        logger.info("\n=== Test 5: Document Count ===")
        count_result = await rag_service.get_document_count(config_name)
        logger.info(f"Total documents in store: {count_result['count']}")
        
        # Test 6: Different similarity thresholds
        logger.info("\n=== Test 6: Similarity Threshold Testing ===")
        thresholds = [0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            threshold_result = await rag_service.retrieve(
                config_name,
                "artificial intelligence and neural networks",
                k=10,
                similarity_threshold=threshold
            )
            logger.info(f"Threshold {threshold}: {len(threshold_result['documents'])} documents")
        
        # Test 7: Test with different query types
        logger.info("\n=== Test 7: Different Query Types ===")
        test_queries = [
            "Python programming language",
            "data visualization techniques",
            "cloud infrastructure for ML",
            "beginner guide to AI"
        ]
        
        for query in test_queries:
            query_result = await rag_service.retrieve(config_name, query, k=2)
            logger.info(f"Query: '{query}' -> {len(query_result['documents'])} results")
            if query_result['documents']:
                top_doc = query_result['documents'][0]
                logger.info(f"  Top result: {top_doc['content'][:60]}... (Score: {top_doc.get('similarity_score', 'N/A'):.3f})")
        
        # Test 8: Multi-Search Type Testing (New Feature)
        logger.info("\n=== Test 8: Multi-Search Type Testing ===")
        await test_search_types(rag_service, config_name)
        
        # Test 9: Search Performance Comparison
        logger.info("\n=== Test 9: Search Performance Comparison ===")
        await compare_search_performance(rag_service, config_name)
        
        logger.info("\n=== All Tests Completed Successfully! ===")
        logger.info("Elasticsearch multi-search vector store is working correctly.")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

async def test_search_types(rag_service: RAGService, config_name: str):
    """Test all search types: fulltext, vector, semantic, and hybrid."""
    try:
        # Get the vector store directly to test search types
        vector_store = rag_service._get_vector_store()
        
        test_query = "machine learning algorithms and neural networks"
        
        logger.info(f"Testing search types with query: '{test_query}'")
        
        # Test 1: Fulltext Search (BM25)
        logger.info("\n--- Fulltext Search (BM25) ---")
        try:
            fulltext_results = vector_store.similarity_search(
                query=test_query,
                k=3,
                search_type=ElasticsearchSearchType.FULLTEXT
            )
            logger.info(f"Fulltext search returned {len(fulltext_results)} results:")
            for i, (doc, score) in enumerate(fulltext_results[:2]):
                logger.info(f"  {i+1}. Score: {score:.3f} | Topic: {doc.metadata.get('topic', 'N/A')}")
                logger.info(f"     Content: {doc.page_content[:80]}...")
        except Exception as e:
            logger.error(f"Fulltext search failed: {str(e)}")
        
        # Test 2: Vector Search (Embeddings)
        logger.info("\n--- Vector Search (Embeddings) ---")
        try:
            vector_results = vector_store.similarity_search(
                query=test_query,
                k=3,
                search_type=ElasticsearchSearchType.VECTOR
            )
            logger.info(f"Vector search returned {len(vector_results)} results:")
            for i, (doc, score) in enumerate(vector_results[:2]):
                logger.info(f"  {i+1}. Score: {score:.3f} | Topic: {doc.metadata.get('topic', 'N/A')}")
                logger.info(f"     Content: {doc.page_content[:80]}...")
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
        
        # Test 3: Semantic Search (ELSER)
        logger.info("\n--- Semantic Search (ELSER) ---")
        try:
            semantic_results = vector_store.similarity_search(
                query=test_query,
                k=3,
                search_type=ElasticsearchSearchType.SEMANTIC
            )
            logger.info(f"Semantic search returned {len(semantic_results)} results:")
            for i, (doc, score) in enumerate(semantic_results[:2]):
                logger.info(f"  {i+1}. Score: {score:.3f} | Topic: {doc.metadata.get('topic', 'N/A')}")
                logger.info(f"     Content: {doc.page_content[:80]}...")
        except Exception as e:
            logger.warning(f"Semantic search not available: {str(e)}")
            logger.info("Note: Semantic search requires ELSER model deployment")
        
        # Test 4: Hybrid Search (Combined)
        logger.info("\n--- Hybrid Search (Combined) ---")
        try:
            hybrid_results = vector_store.similarity_search(
                query=test_query,
                k=3,
                search_type=ElasticsearchSearchType.HYBRID
            )
            logger.info(f"Hybrid search returned {len(hybrid_results)} results:")
            for i, (doc, score) in enumerate(hybrid_results[:2]):
                logger.info(f"  {i+1}. Score: {score:.3f} | Topic: {doc.metadata.get('topic', 'N/A')}")
                logger.info(f"     Content: {doc.page_content[:80]}...")
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
        
        # Test search types with metadata filtering
        logger.info("\n--- Search Types with Filtering ---")
        metadata_filter = {"category": "technology"}
        
        for search_type in [ElasticsearchSearchType.FULLTEXT, ElasticsearchSearchType.VECTOR, ElasticsearchSearchType.HYBRID]:
            try:
                filtered_results = vector_store.similarity_search(
                    query="artificial intelligence",
                    k=3,
                    filter=metadata_filter,
                    search_type=search_type
                )
                logger.info(f"{search_type.value} with filter: {len(filtered_results)} results")
            except Exception as e:
                logger.error(f"{search_type.value} with filter failed: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error testing search types: {str(e)}")

async def compare_search_performance(rag_service: RAGService, config_name: str):
    """Compare performance and results of different search strategies."""
    try:
        vector_store = rag_service._get_vector_store()
        
        test_queries = [
            "machine learning algorithms",
            "Python programming language",
            "data visualization techniques",
            "cloud computing infrastructure"
        ]
        
        logger.info("Comparing search strategies across multiple queries:")
        
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            logger.info("-" * 40)
            
            search_results = {}
            
            # Test each search type
            for search_type in [ElasticsearchSearchType.FULLTEXT, ElasticsearchSearchType.VECTOR, ElasticsearchSearchType.HYBRID]:
                try:
                    results = vector_store.similarity_search(
                        query=query,
                        k=3,
                        search_type=search_type
                    )
                    search_results[search_type.value] = results
                    
                    if results:
                        top_score = results[0][1]
                        top_topic = results[0][0].metadata.get('topic', 'N/A')
                        logger.info(f"  {search_type.value:10}: {len(results)} results, top: {top_topic} ({top_score:.3f})")
                    else:
                        logger.info(f"  {search_type.value:10}: No results")
                        
                except Exception as e:
                    logger.error(f"  {search_type.value:10}: Error - {str(e)}")
            
            # Analyze result overlap
            if len(search_results) >= 2:
                analyze_result_overlap(search_results)
                
    except Exception as e:
        logger.error(f"Error in performance comparison: {str(e)}")

def analyze_result_overlap(search_results: Dict[str, List]):
    """Analyze overlap between different search result sets."""
    try:
        # Get content from each search type for comparison
        content_sets = {}
        for search_type, results in search_results.items():
            if results:
                content_sets[search_type] = set(doc.page_content[:100] for doc, _ in results)
        
        if len(content_sets) >= 2:
            search_types = list(content_sets.keys())
            for i in range(len(search_types)):
                for j in range(i + 1, len(search_types)):
                    type_a, type_b = search_types[i], search_types[j]
                    overlap = len(content_sets[type_a] & content_sets[type_b])
                    total_unique = len(content_sets[type_a] | content_sets[type_b])
                    
                    if total_unique > 0:
                        overlap_percentage = (overlap / total_unique) * 100
                        logger.info(f"    {type_a}-{type_b} overlap: {overlap_percentage:.1f}% ({overlap}/{total_unique})")
                        
    except Exception as e:
        logger.error(f"Error analyzing result overlap: {str(e)}")

def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test Elasticsearch vector store functionality")
    parser.add_argument("--config", required=True, help="Configuration name to use/create")
    parser.add_argument("--create-new", action="store_true", help="Create a new test configuration")
    parser.add_argument("--use-api-key", action="store_true", help="Use API key authentication instead of username/password")
    parser.add_argument("--no-index-suffix", action="store_true", help="Don't append configuration name to index name")
    
    args = parser.parse_args()
    
    # Determine index suffix behavior (default is True, --no-index-suffix makes it False)
    use_index_suffix = not args.no_index_suffix
    
    logger.info("Starting Elasticsearch Vector Store Test")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Create new: {args.create_new}")
    logger.info(f"Use API key: {args.use_api_key}")
    logger.info(f"Use index suffix: {use_index_suffix}")
    
    # Run the async test
    asyncio.run(test_elasticsearch_vector_store(args.config, args.create_new, args.use_api_key, use_index_suffix))

if __name__ == "__main__":
    main()
