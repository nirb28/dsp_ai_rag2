#!/usr/bin/env python3
"""
Test script for Neo4j Knowledge Graph functionality

This script tests the new knowledge graph implementation to ensure it works correctly
with the existing RAG service architecture.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from langchain.docstore.document import Document as LangchainDocument
from app.services.rag_service import RAGService
from app.config import LLMConfig, LLMProvider, RAGConfig


async def test_knowledge_graph_basic():
    """Test basic knowledge graph functionality."""
    print("🧪 Testing Knowledge Graph Basic Functionality")
    print("=" * 50)
    
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        # Create LLM configuration for graph extraction
        print("1. Creating LLM configuration...")
        llm_config_name = "test-kg-llm"
        
        llm_config_obj = LLMConfig(
            provider=LLMProvider.GROQ,
            model="llama3-8b-8192",
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key=os.getenv("GROQ_API_KEY", "test-key"),
            system_prompt="Extract entities and relationships from text.",
            temperature=0.1,
            max_tokens=512
        )
        rag_service.set_llm_configuration(llm_config_name, llm_config_obj)
        print(f"✅ Created LLM config: {llm_config_name}")
        
        # Create knowledge graph configuration
        print("2. Creating knowledge graph configuration...")
        config_name = "test_kg_config"
        
        config_data = {
            "vector_store": {
                "type": "neo4j_knowledge_graph",
                "neo4j_uri": os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
                "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
                "neo4j_password": os.getenv("NEO4J_PASSWORD", "password"),
                "neo4j_database": "neo4j",
                "kg_llm_config_name": llm_config_name
            },
            "embedding": {
                "enabled": False,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "generation": {
                "model": "llama3-8b-8192",
                "provider": "groq",
                "endpoint": "https://api.groq.com/openai/v1/chat/completions",
                "api_key": os.getenv("GROQ_API_KEY", "test-key"),
                "temperature": 0.7,
                "max_tokens": 512
            }
        }
        
        config = RAGConfig(**config_data)
        rag_service.set_configuration(config_name, config)
        print(f"✅ Created configuration: {config_name}")
        
        # Test document addition
        print("3. Testing document addition...")
        test_docs = [
            LangchainDocument(
                page_content="John Smith works at Microsoft Corporation in Seattle. He is a software engineer.",
                metadata={"id": "doc1", "title": "Employee Info", "category": "hr"}
            ),
            LangchainDocument(
                page_content="Microsoft Corporation is a technology company founded by Bill Gates and Paul Allen.",
                metadata={"id": "doc2", "title": "Company Info", "category": "business"}
            )
        ]
        
        doc_ids = []
        for i, doc in enumerate(test_docs):
            doc_result = rag_service.upload_text_content(
                content=doc.page_content,
                filename=doc.metadata.get('title', f'test_doc_{i}.txt'),
                configuration_name=config_name,
                metadata=doc.metadata,
                process_immediately=True
            )
            doc_ids.append(doc_result['document_id'])
        print(f"✅ Added documents: {doc_ids}")
        
        # Test vector store access
        print("4. Testing vector store access...")
        vector_store = rag_service.vector_store_manager.get_vector_store(
            config_name,
            rag_service.configurations[config_name].vector_store,
            rag_service.configurations[config_name].embedding.dict()
        )
        
        # Test document count
        doc_count = vector_store.get_document_count()
        print(f"✅ Document count: {doc_count}")
        
        # Test graph stats
        print("5. Testing graph statistics...")
        stats = vector_store.get_graph_stats()
        print(f"✅ Graph stats: {json.dumps(stats, indent=2)}")
        
        # Test entity finding
        print("6. Testing entity finding...")
        entities = vector_store.find_entities(limit=5)
        print(f"✅ Found {len(entities)} entities")
        for entity in entities[:3]:
            print(f"   - {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
        
        # Test similarity search
        print("7. Testing similarity search...")
        search_results = vector_store.similarity_search("Who works at Microsoft?", k=2)
        print(f"✅ Search returned {len(search_results)} results")
        for result in search_results:
            print(f"   - Score: {result.get('score', 0):.3f}, Content: {result.get('content', '')[:100]}...")
        
        # Test query functionality
        print("8. Testing RAG query...")
        query_result = rag_service.query(config_name, "Tell me about Microsoft Corporation", k=2)
        print(f"✅ Query result: {query_result['answer'][:200]}...")
        print(f"   Sources: {len(query_result['sources'])}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_configuration_validation():
    """Test configuration validation for knowledge graph."""
    print("\n🧪 Testing Configuration Validation")
    print("=" * 40)
    
    try:
        rag_service = RAGService()
        
        # Test missing LLM config name
        print("1. Testing missing LLM config name...")
        try:
            config_data = {
                "vector_store": {
                    "type": "neo4j_knowledge_graph",
                    "neo4j_uri": "neo4j://localhost:7687",
                    "neo4j_user": "neo4j",
                    "neo4j_password": "password",
                    "neo4j_database": "neo4j"
                    # Missing kg_llm_config_name
                },
                "embedding": {"enabled": False},
                "generation": {"model": "test", "provider": "groq"}
            }
            
            config = RAGConfig(**config_data)
            rag_service.set_configuration("test_invalid", config)
            print("❌ Should have failed with missing LLM config")
            return False
            
        except ValueError as e:
            if "kg_llm_config_name is required" in str(e):
                print("✅ Correctly caught missing LLM config error")
            else:
                print(f"❌ Unexpected error: {e}")
                return False
        
        print("✅ Configuration validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Knowledge Graph Tests")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY not set. Using test key.")
    
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    print(f"Neo4j URI: {neo4j_uri}")
    print()
    
    async def run_tests():
        # Run basic functionality test
        basic_test_passed = await test_knowledge_graph_basic()
        
        # Run configuration validation test
        config_test_passed = await test_configuration_validation()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        print(f"   Basic functionality: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
        print(f"   Configuration validation: {'✅ PASSED' if config_test_passed else '❌ FAILED'}")
        
        if basic_test_passed and config_test_passed:
            print("\n🎉 All tests passed! Knowledge graph implementation is working.")
        else:
            print("\n⚠️  Some tests failed. Please check the implementation.")
    
    # Run the tests
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
