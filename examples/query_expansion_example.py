#!/usr/bin/env python3
"""
Query Expansion Example for DSP AI RAG2 Project

This example demonstrates how to use the query expansion feature with different strategies
and LLM configurations.
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any, Optional

# Configuration
BASE_URL = "http://localhost:9000"  # Adjust to your RAG service URL
GROQ_API_KEY = "your-groq-api-key-here"  # Replace with your actual API key


async def create_llm_configuration(session: aiohttp.ClientSession, config: Dict[str, Any]) -> bool:
    """Create an LLM configuration for query expansion."""
    try:
        async with session.post(f"{BASE_URL}/llm-configs", json=config) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Created LLM configuration: {result['name']}")
                return True
            else:
                error = await response.text()
                print(f"❌ Failed to create LLM configuration: {error}")
                return False
    except Exception as e:
        print(f"❌ Error creating LLM configuration: {str(e)}")
        return False


async def test_query_expansion(session: aiohttp.ClientSession, query: str, expansion_config: Dict[str, Any]) -> None:
    """Test query expansion with different strategies."""
    print(f"\n🔍 Testing query: '{query}'")
    print(f"📋 Expansion config: {json.dumps(expansion_config, indent=2)}")
    
    # Test with /query endpoint
    query_payload = {
        "query": query,
        "configuration_name": "default",
        "k": 5,
        "query_expansion": expansion_config
    }
    
    try:
        async with session.post(f"{BASE_URL}/query", json=query_payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Query completed in {result['processing_time']:.2f}s")
                print(f"📄 Found {len(result['sources'])} sources")
                
                # Show sources with query information
                for i, source in enumerate(result['sources'][:3]):  # Show first 3 sources
                    print(f"  Source {i+1}:")
                    print(f"    Score: {source['similarity_score']:.3f}")
                    if 'source_query' in source and source['source_query']:
                        print(f"    From query: '{source['source_query']}'")
                    print(f"    Content: {source['content'][:100]}...")
                    print()
                
                print(f"🤖 Answer: {result['answer'][:200]}...")
            else:
                error = await response.text()
                print(f"❌ Query failed: {error}")
    except Exception as e:
        print(f"❌ Error during query: {str(e)}")
    
    # Test with /retrieve endpoint
    retrieve_payload = {
        "query": query,
        "configuration_name": "default",
        "k": 5,
        "query_expansion": expansion_config
    }
    
    try:
        async with session.post(f"{BASE_URL}/retrieve", json=retrieve_payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"🔍 Retrieve completed in {result['processing_time']:.2f}s")
                print(f"📄 Retrieved {len(result['documents'])} documents")
                
                # Show documents with query information
                for i, doc in enumerate(result['documents'][:2]):  # Show first 2 documents
                    print(f"  Document {i+1}:")
                    print(f"    Score: {doc['similarity_score']:.3f}")
                    if 'source_query' in doc and doc['source_query']:
                        print(f"    From query: '{doc['source_query']}'")
                    print(f"    Content: {doc['content'][:100]}...")
                    print()
            else:
                error = await response.text()
                print(f"❌ Retrieve failed: {error}")
    except Exception as e:
        print(f"❌ Error during retrieve: {str(e)}")


async def main():
    """Main example function."""
    print("🚀 Query Expansion Example for DSP AI RAG2")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # 1. Create LLM configurations
        print("\n📝 Step 1: Creating LLM configurations...")
        
        # Groq configuration
        groq_config = {
            "name": "groq-llama3",
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
        
        # OpenAI-compatible configuration (for local models)
        local_config = {
            "name": "local-llama",
            "provider": "openai_compatible",
            "model": "llama3",
            "endpoint": "http://localhost:8000/v1/chat/completions",
            "api_key": None,  # No API key needed for local
            "system_prompt": "Generate query variations for better document retrieval.",
            "temperature": 0.5,
            "max_tokens": 256,
            "top_p": 0.9,
            "timeout": 30
        }
        
        # Create configurations
        await create_llm_configuration(session, groq_config)
        await create_llm_configuration(session, local_config)
        
        # 2. Test different query expansion strategies
        print("\n🧪 Step 2: Testing query expansion strategies...")
        
        test_queries = [
            "What is machine learning?",
            "How does neural network training work?",
            "Explain transformer architecture"
        ]
        
        # Test fusion strategy
        fusion_config = {
            "enabled": True,
            "strategy": "fusion",
            "llm_config_name": "groq-llama3",
            "num_queries": 3
        }
        
        # Test multi-query strategy
        multi_query_config = {
            "enabled": True,
            "strategy": "multi_query",
            "llm_config_name": "groq-llama3",
            "num_queries": 4
        }
        
        for query in test_queries:
            print(f"\n{'='*60}")
            
            # Test fusion strategy
            print(f"\n🔄 Testing FUSION strategy:")
            await test_query_expansion(session, query, fusion_config)
            
            # Test multi-query strategy
            print(f"\n🔄 Testing MULTI-QUERY strategy:")
            await test_query_expansion(session, query, multi_query_config)
            
            # Test without expansion for comparison
            print(f"\n🔄 Testing WITHOUT expansion (baseline):")
            no_expansion_config = {"enabled": False}
            await test_query_expansion(session, query, no_expansion_config)
        
        # 3. List all LLM configurations
        print(f"\n{'='*60}")
        print("\n📋 Step 3: Listing all LLM configurations...")
        try:
            async with session.get(f"{BASE_URL}/llm-configs") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Found {result['total_count']} LLM configurations:")
                    for config in result['configurations']:
                        print(f"  - {config['name']} ({config['provider']}, {config['model']})")
                else:
                    error = await response.text()
                    print(f"❌ Failed to list configurations: {error}")
        except Exception as e:
            print(f"❌ Error listing configurations: {str(e)}")
    
    print(f"\n{'='*60}")
    print("✅ Query expansion example completed!")
    print("\n💡 Tips:")
    print("  - Use 'fusion' strategy for semantically similar query variations")
    print("  - Use 'multi_query' strategy for exploring different aspects")
    print("  - Adjust num_queries based on your performance requirements")
    print("  - Results are automatically merged and deduplicated")
    print("  - Original query is always included in the search")


if __name__ == "__main__":
    # Check if required dependencies are available
    try:
        import aiohttp
    except ImportError:
        print("❌ Error: aiohttp is required. Install with: pip install aiohttp")
        exit(1)
    
    # Run the example
    asyncio.run(main())
