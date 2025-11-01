#!/usr/bin/env python3
"""
Test script for Query Expansion Metadata Feature

This script tests the new include_metadata functionality for query expansion.
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any
import os

# Configuration
BASE_URL = "http://localhost:9000/api/v1"
existing_llm_config = "nvidia-llama3-8b"
existing_rag_config = "batch_ml_ai_basics_test"

async def test_query_expansion_metadata():
    """Test the query expansion metadata feature."""
    
    llm_config_name = existing_llm_config if existing_llm_config else "test-groq-llama3"
    async with aiohttp.ClientSession() as session:
        print("Testing Query Expansion Metadata Feature")
        print("=" * 50)
        
        if existing_llm_config:
            print(f"\nℹ️ Using existing LLM configuration: {existing_llm_config}")
        else:
            # 1. Create LLM configuration for testing
            print("\nCreating test LLM configuration...")
            llm_config = {
                "name": llm_config_name,
                "provider": "groq",
                "model": "llama3-8b-8192",
                "endpoint": "https://api.groq.com/openai/v1/chat/completions",
                "api_key": GROQ_API_KEY,
                "system_prompt": "You are a helpful assistant that generates query variations for information retrieval.",
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 0.9,
                "timeout": 30
            }
            try:
                async with session.post(f"{BASE_URL}/llm-configs", json=llm_config) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Created LLM configuration: {result['name']}")
                    else:
                        error = await response.text()
                        print(f"❌ Failed to create LLM configuration: {error}")
                        return
            except Exception as e:
                print(f"❌ Error creating LLM configuration: {str(e)}")
                return
        # 2. Test query with metadata enabled
        print("\n🔍 Step 2: Testing query with metadata enabled...")
        
        test_query = "What is machine learning?"
        
        query_payload = {
            "query": test_query,
            "configuration_name": existing_rag_config,
            "k": 3,
            "query_expansion": {
                "enabled": True,
                "strategy": "multi_query",
                "llm_config_name": llm_config_name,
                "num_queries": 4,
                "include_metadata": True
            }
        }
        
        try:
            async with session.post(f"{BASE_URL}/query", json=query_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Query completed in {result['processing_time']:.2f}s")
                    print(f"📄 Found {len(result['sources'])} sources")
                    
                    # Check if metadata is present
                    if 'query_expansion_metadata' in result and result['query_expansion_metadata']:
                        metadata = result['query_expansion_metadata']
                        print(f"\n📊 Query Expansion Metadata Found!")
                        print(f"  Original Query: {metadata.get('original_query', 'N/A')}")
                        print(f"  Strategy: {metadata.get('strategy', 'N/A')}")
                        print(f"  LLM Config: {metadata.get('llm_config_name', 'N/A')}")
                        print(f"  LLM Provider: {metadata.get('llm_provider', 'N/A')}")
                        print(f"  Requested Queries: {metadata.get('requested_num_queries', 'N/A')}")
                        print(f"  Actual Queries: {metadata.get('actual_num_queries', 'N/A')}")
                        print(f"  Processing Time: {metadata.get('processing_time_seconds', 0):.3f}s")
                        print(f"  Expansion Successful: {metadata.get('expansion_successful', False)}")
                        
                        if 'expanded_queries' in metadata:
                            print(f"  Expanded Queries:")
                            for i, query in enumerate(metadata['expanded_queries']):
                                original_marker = " (ORIGINAL)" if query == test_query else ""
                                print(f"    {i+1}. {query}{original_marker}")
                        
                        if 'query_results_summary' in metadata:
                            print(f"  Query Results Summary:")
                            for i, query_result in enumerate(metadata['query_results_summary']):
                                original_marker = " (ORIGINAL)" if query_result.get('is_original') else ""
                                print(f"    Query {i+1}{original_marker}: '{query_result['query']}'")
                                print(f"      Results: {query_result['results_count']}, Top Score: {query_result['top_similarity_score']:.3f}")
                        
                        print(f"  Total Unique Results: {metadata.get('total_unique_results', 'N/A')}")
                        
                        if metadata.get('error_message'):
                            print(f"  Error: {metadata['error_message']}")
                    else:
                        print("❌ No query expansion metadata found!")
                        
                else:
                    error = await response.text()
                    print(f"❌ Query failed: {error}")
        except Exception as e:
            print(f"❌ Error during query: {str(e)}")
        
        # 3. Test retrieve with metadata enabled
        print("\n🔍 Step 3: Testing retrieve with metadata enabled...")
        
        retrieve_payload = {
            "query": test_query,
            "configuration_name": existing_rag_config,
            "k": 3,
            "query_expansion": {
                "enabled": True,
                "strategy": "fusion",
                "llm_config_name": llm_config_name,
                "num_queries": 3,
                "include_metadata": True
            }
        }
        
        try:
            async with session.post(f"{BASE_URL}/retrieve", json=retrieve_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Retrieve completed in {result['processing_time']:.2f}s")
                    print(f"📄 Retrieved {len(result['documents'])} documents")
                    
                    # Check if metadata is present
                    if 'query_expansion_metadata' in result and result['query_expansion_metadata']:
                        metadata = result['query_expansion_metadata']
                        print(f"\n📊 Query Expansion Metadata Found!")
                        print(f"  Strategy: {metadata.get('strategy', 'N/A')}")
                        print(f"  Expansion Successful: {metadata.get('expansion_successful', False)}")
                        print(f"  Processing Time: {metadata.get('processing_time_seconds', 0):.3f}s")
                        
                        if 'query_results_summary' in metadata:
                            print(f"  Query Results Summary:")
                            for i, query_result in enumerate(metadata['query_results_summary']):
                                original_marker = " (ORIGINAL)" if query_result.get('is_original') else ""
                                print(f"    Query {i+1}{original_marker}: '{query_result['query']}'")
                                print(f"      Results: {query_result['results_count']}, Top Score: {query_result['top_similarity_score']:.3f}")
                    else:
                        print("❌ No query expansion metadata found!")
                        
                else:
                    error = await response.text()
                    print(f"❌ Retrieve failed: {error}")
        except Exception as e:
            print(f"❌ Error during retrieve: {str(e)}")
        
        # 4. Test without metadata (should not include metadata)
        print("\n🔍 Step 4: Testing query without metadata...")
        
        query_payload_no_metadata = {
            "query": test_query,
            "configuration_name": existing_rag_config,
            "k": 3,
            "query_expansion": {
                "enabled": True,
                "strategy": "fusion",
                "llm_config_name": llm_config_name,
                "num_queries": 3,
                "include_metadata": False  # Explicitly disabled
            }
        }
        
        try:
            async with session.post(f"{BASE_URL}/query", json=query_payload_no_metadata) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Query completed in {result['processing_time']:.2f}s")
                    
                    # Check if metadata is NOT present
                    if 'query_expansion_metadata' not in result or result['query_expansion_metadata'] is None:
                        print("✅ Correctly excluded metadata when include_metadata=False")
                    else:
                        print("❌ Metadata was included when it should have been excluded!")
                        
                else:
                    error = await response.text()
                    print(f"❌ Query failed: {error}")
        except Exception as e:
            print(f"❌ Error during query: {str(e)}")
        
        print(f"\n{'='*50}")
        print("✅ Query expansion metadata testing completed!")
        print("\n💡 Summary:")
        print("  - Added include_metadata field to QueryExpansionRequest")
        print("  - Enhanced query expansion service to collect metadata")
        print("  - Updated query() and retrieve() methods to return metadata")
        print("  - Added query_expansion_metadata to response models")
        print("  - Metadata includes: strategy, LLM config, processing time, queries, results summary")


if __name__ == "__main__":
    # Check if required dependencies are available
    try:
        import aiohttp
    except ImportError:
        print("❌ Error: aiohttp is required. Install with: pip install aiohttp")
        exit(1)
    
    # Run the test
    asyncio.run(test_query_expansion_metadata())
