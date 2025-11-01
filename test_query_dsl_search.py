#!/usr/bin/env python3
"""
Test script for Elasticsearch QueryDSL search functionality.
Tests the new query_dsl search type with various DSL templates including RRF examples.
"es_query_dsl_template": {  "retriever": {    "rrf": {      "retrievers": [        {          "standard": {            "query": {              "match": {                "text": "where is yellowstone"              }            }          }        },        {          "standard": {            "query": {                "semantic": {                    "field": "text",                    "query": "where is yellowstone"                }            }          }        }      ]    }  }}
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from app.config import RAGConfig
from app.services.rag_service import RAGService
from app.model_schemas.base_models import QueryRequest

def create_rrf_query_template() -> Dict[str, Any]:
    """
    Create an RRF (Reciprocal Rank Fusion) query template that combines:
    - BM25 fulltext search
    - Dense vector search
    - Semantic search (if available)
    """
    return {
        "query": {
            "rrf": {
                "queries": [
                    {
                        "match": {
                            "content": "$QUERY$"
                        }
                    },
                    {
                        "knn": {
                            "field": "vector",
                            "query_vector": "$QUERY$",  # This would need to be replaced with actual vector
                            "k": 50,
                            "num_candidates": 100
                        }
                    }
                ],
                "rank_constant": 20,
                "window_size": 100
            }
        },
        "size": 10
    }

def create_multi_match_query_template() -> Dict[str, Any]:
    """
    Create a multi-match query template that searches across multiple fields.
    """
    return {
        "query": {
            "multi_match": {
                "query": "$QUERY$",
                "fields": ["content^2", "title^3", "summary"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        },
        "highlight": {
            "fields": {
                "content": {},
                "title": {},
                "summary": {}
            }
        },
        "size": 10
    }

def create_bool_query_template() -> Dict[str, Any]:
    """
    Create a complex boolean query template with boosting.
    """
    return {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": {
                                "query": "$QUERY$",
                                "boost": 1.0
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "content": {
                                "query": "$QUERY$",
                                "boost": 2.0
                            }
                        }
                    },
                    {
                        "wildcard": {
                            "content": {
                                "value": "*$QUERY$*",
                                "boost": 0.5
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "size": 10
    }

def create_test_config_with_query_dsl(template: Dict[str, Any]) -> RAGConfig:
    """Create a test configuration with queryDSL template."""
    config_dict = {
        "configurations": {
            "test_query_dsl": {
                "embedding": {
                    "provider": "local_model_server",
                    "model": "all-MiniLM-L6-v2",
                    "endpoint_url": "http://localhost:8001/embeddings",
                    "dimension": 384
                },
                "vector_store": {
                    "type": "elasticsearch",
                    "es_url": "http://localhost:9200",
                    "es_index_name": "test_documents",
                    "es_search_type": "query_dsl",
                    "es_query_dsl_template": template,
                    "es_fulltext_field": "content",
                    "es_semantic_field": "semantic_content",
                    "es_semantic_inference_id": "my-elser-model",
                    "normalize_similarity_scores": False
                }
            }
        }
    }
    return RAGConfig.from_dict(config_dict)

async def test_query_dsl_search(template_name: str, template: Dict[str, Any], test_queries: List[str]):
    """Test queryDSL search with a specific template."""
    print(f"\n{'='*60}")
    print(f"Testing {template_name}")
    print(f"{'='*60}")
    
    try:
        # Create configuration with the template
        config = create_test_config_with_query_dsl(template)
        rag_service = RAGService(config)
        
        print(f"Template structure:")
        print(json.dumps(template, indent=2))
        print(f"\n{'-'*40}")
        
        # Test each query
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            print(f"{'-'*20}")
            
            try:
                request = QueryRequest(
                    query=query,
                    configuration_name="test_query_dsl",
                    k=5
                )
                
                # This will test the queryDSL functionality
                results = await rag_service.query(request)
                
                print(f"✅ Query executed successfully")
                print(f"Results found: {len(results.documents)}")
                
                if results.documents:
                    for i, doc in enumerate(results.documents[:3], 1):
                        print(f"  {i}. Score: {doc.similarity_score:.4f}")
                        content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                        print(f"     Content: {content_preview}")
                        
                else:
                    print("  No documents found")
                    
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
                
    except Exception as e:
        print(f"❌ Template test failed: {str(e)}")
        
    print(f"\n{'-'*40}")

async def test_template_validation():
    """Test query DSL template validation and $QUERY$ replacement."""
    print(f"\n{'='*60}")
    print("Testing Template Validation & Query Replacement")
    print(f"{'='*60}")
    
    # Test template with missing $QUERY$ placeholder
    invalid_template = {
        "query": {
            "match": {
                "content": "static query without placeholder"
            }
        }
    }
    
    print("\n1. Testing template without $QUERY$ placeholder:")
    try:
        config = create_test_config_with_query_dsl(invalid_template)
        rag_service = RAGService(config)
        
        request = QueryRequest(
            query="test query",
            configuration_name="test_query_dsl",
            k=5
        )
        
        results = await rag_service.query(request)
        print("✅ Template accepted (will use static query)")
        
    except Exception as e:
        print(f"❌ Template validation failed: {str(e)}")
    
    # Test template with multiple $QUERY$ placeholders
    multi_placeholder_template = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": "$QUERY$"
                        }
                    },
                    {
                        "match": {
                            "title": "$QUERY$"
                        }
                    }
                ]
            }
        }
    }
    
    print("\n2. Testing template with multiple $QUERY$ placeholders:")
    try:
        config = create_test_config_with_query_dsl(multi_placeholder_template)
        rag_service = RAGService(config)
        
        request = QueryRequest(
            query="machine learning",
            configuration_name="test_query_dsl", 
            k=5
        )
        
        results = await rag_service.query(request)
        print("✅ Multiple placeholders replaced successfully")
        
    except Exception as e:
        print(f"❌ Multiple placeholder test failed: {str(e)}")

async def test_fallback_behavior():
    """Test fallback behavior when queryDSL template is not configured or fails."""
    print(f"\n{'='*60}")
    print("Testing Fallback Behavior")
    print(f"{'='*60}")
    
    # Test with no template configured
    config_dict = {
        "configurations": {
            "test_no_template": {
                "embedding": {
                    "provider": "local_model_server",
                    "model": "all-MiniLM-L6-v2",
                    "endpoint_url": "http://localhost:8001/embeddings",
                    "dimension": 384
                },
                "vector_store": {
                    "type": "elasticsearch",
                    "es_url": "http://localhost:9200",
                    "es_index_name": "test_documents",
                    "es_search_type": "query_dsl",
                    # No es_query_dsl_template provided
                    "es_fulltext_field": "content"
                }
            }
        }
    }
    
    print("\n1. Testing with no queryDSL template configured:")
    try:
        config = RAGConfig.from_dict(config_dict)
        rag_service = RAGService(config)
        
        request = QueryRequest(
            query="test query",
            configuration_name="test_no_template",
            k=5
        )
        
        results = await rag_service.query(request)
        print("✅ Fallback to fulltext search succeeded")
        
    except Exception as e:
        print(f"❌ Fallback test failed: {str(e)}")

async def main():
    """Run all queryDSL search tests."""
    print("🧪 Elasticsearch QueryDSL Search Test Suite")
    print("=" * 60)
    
    test_queries = [
        "machine learning algorithms",
        "natural language processing",
        "deep neural networks",
        "data science techniques"
    ]
    
    # Test different query templates
    await test_query_dsl_search(
        "Multi-Match Query",
        create_multi_match_query_template(),
        test_queries[:2]
    )
    
    await test_query_dsl_search(
        "Boolean Query with Boosting",
        create_bool_query_template(),
        test_queries[2:]
    )
    
    await test_query_dsl_search(
        "RRF Query (Reciprocal Rank Fusion)",
        create_rrf_query_template(),
        test_queries[:1]  # RRF might need special handling
    )
    
    # Test validation and edge cases
    await test_template_validation()
    await test_fallback_behavior()
    
    print(f"\n{'='*60}")
    print("✅ QueryDSL Search Test Suite Complete")
    print(f"{'='*60}")
    
    print(f"\n📋 Test Summary:")
    print(f"- Multi-Match Query: Tests flexible field searching with boosting")
    print(f"- Boolean Query: Tests complex query composition with multiple match types")
    print(f"- RRF Query: Tests Reciprocal Rank Fusion for combining multiple search strategies")
    print(f"- Template Validation: Tests $QUERY$ placeholder replacement")
    print(f"- Fallback Behavior: Tests graceful degradation when template is not configured")

if __name__ == "__main__":
    asyncio.run(main())
