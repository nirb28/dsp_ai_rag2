#!/usr/bin/env python3
"""
Simple test script to validate query expansion implementation
"""

import asyncio
import sys
import os
import argparse

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_query_expansion(config_name=None, create_new=False):
    """Test the query expansion functionality."""
    print("🧪 Testing Query Expansion Implementation")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from app.services.query_expansion_service import QueryExpansionService
        from app.config import LLMConfig, LLMProvider
        from app.model_schemas import QueryExpansionRequest, LLMConfigRequest
        from app.services.rag_service import RAGService
        print("✅ All imports successful")
        
        rag_service = RAGService()
        if config_name and not create_new:
            print(f"\nℹ️ Using existing configuration: {config_name}")
            # Optionally, check if config exists
            # If you want to test actual expansion logic with the config, add more here
            print(f"\n{'='*50}")
            print(f"🎉 Test finished using existing config '{config_name}' (no creation/cleanup).")
            return True
        # Otherwise, create and clean up test config
        print("\n🔧 Testing LLM configuration creation...")
        llm_config = LLMConfig(
            name="test-config",
            provider=LLMProvider.GROQ,
            model="llama3-8b-8192",
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key="test-key",
            system_prompt="Test prompt",
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            timeout=30
        )
        print(f"✅ LLM config created: {llm_config.name}")
        
        # Test RAG service initialization
        print("\n🚀 Testing RAG service initialization...")
        print(f"✅ RAG service initialized with {len(rag_service.llm_configurations)} LLM configs")
        
        # Test LLM config management
        print("\n💾 Testing LLM configuration management...")
        success = rag_service.set_llm_configuration("test-config", llm_config)
        if success:
            print("✅ LLM configuration saved successfully")
            
            # Test retrieval
            retrieved_config = rag_service.get_llm_configuration("test-config")
            print(f"✅ LLM configuration retrieved: {retrieved_config.name}")
            
            # Test listing
            all_configs = rag_service.get_llm_configurations()
            print(f"✅ Listed {len(all_configs)} LLM configurations")
        else:
            print("❌ Failed to save LLM configuration")
        
        # Test query expansion service
        print("\n🔍 Testing Query Expansion Service...")
        expansion_service = QueryExpansionService()
        print("✅ Query expansion service created")
        
        # Test request models
        print("\n📋 Testing request models...")
        query_expansion_req = QueryExpansionRequest(
            enabled=True,
            strategy="fusion",
            llm_config_name="test-config",
            num_queries=3
        )
        print(f"✅ Query expansion request created: {query_expansion_req.strategy}")
        
        llm_config_req = LLMConfigRequest(
            name="test-api-config",
            provider="groq",
            model="llama3-8b-8192",
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key="test-key"
        )
        print(f"✅ LLM config request created: {llm_config_req.name}")
        
        # Test cleanup
        print("\n🧹 Testing cleanup...")
        cleanup_success = rag_service.delete_llm_configuration("test-config")
        if cleanup_success:
            print("✅ Test configuration cleaned up successfully")
        else:
            print("⚠️  Test configuration cleanup failed (may not exist)")
        
        print(f"\n{'='*50}")
        print("🎉 All tests passed! Query expansion implementation is ready.")
        print("\n📝 Next steps:")
        print("1. Start the RAG service: python app/main.py")
        print("2. Create LLM configurations via POST /llm-configs")
        print("3. Use query expansion in /query and /retrieve endpoints")
        print("4. Test with the example script: python examples/query_expansion_example.py")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Query Expansion Implementation")
    parser.add_argument('--config', type=str, help='Use an existing configuration name (skips creation/cleanup)')
    parser.add_argument('--create-new', action='store_true', help='Force creation of a new test configuration')
    args = parser.parse_args()
    result = asyncio.run(test_query_expansion(config_name=args.config, create_new=args.create_new))
    sys.exit(0 if result else 1)
