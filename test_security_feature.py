"""
Test script for the security feature implementation.

This script tests:
1. Security configuration creation
2. JWT Bearer token authentication
3. Metadata filter extraction from JWT claims
4. Query and retrieve endpoint security validation
"""

import asyncio
import json
import jwt
import httpx
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import argparse
import sys


class SecurityFeatureTester:
    """Test class for security feature functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", config_name: Optional[str] = None, jwt_secret: Optional[str] = None, use_existing_config: bool = False):
        self.base_url = base_url
        self.use_existing_config = use_existing_config
        self.test_config_name = config_name if config_name else "security_test_config"
        self.jwt_secret = jwt_secret if jwt_secret else "test_secret_key_for_security_testing_123"
        
    def create_jwt_token(
        self, 
        subject: str = "test_user",
        metadata_filter: Optional[Dict[str, Any]] = None,
        expires_in_minutes: int = 30
    ) -> str:
        """Create a JWT token for testing."""
        payload = {
            "sub": subject,
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes),
            "iss": "test_issuer",
            "aud": "test_audience"
        }
        
        if metadata_filter:
            payload["metadata_filter"] = metadata_filter
            
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    async def create_test_configuration(self) -> bool:
        """Create a test configuration with security enabled."""
        print("📋 Creating test configuration with security enabled...")
        
        config = {
            "chunking": {
                "strategy": "recursive_text",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "vector_store": {
                "type": "faiss",
                "index_path": f"./storage/{self.test_config_name}_faiss_index",
                "dimension": 384
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32
            },
            "generation": {
                "model": "llama3-8b-8192",
                "provider": "groq",
                "endpoint": "https://api.groq.com/openai/v1/chat/completions",
                "api_key": "${GROQ_API_KEY}",
                "temperature": 0.7,
                "max_tokens": 512
            },
            "retrieval_k": 5,
            "similarity_threshold": 0.7,
            "reranking": {
                "enabled": False
            },
            "security": {
                "enabled": True,
                "type": "jwt_bearer",
                "jwt_secret_key": self.jwt_secret,
                "jwt_algorithm": "HS256",
                "jwt_issuer": "test_issuer",
                "jwt_audience": "test_audience",
                "jwt_require_exp": True,
                "jwt_require_iat": True,
                "jwt_leeway": 0
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/configurations",
                json={
                    "configuration_name": self.test_config_name,
                    "config": config
                }
            )
            
            if response.status_code == 200:
                print("✅ Test configuration created successfully")
                return True
            else:
                print(f"❌ Failed to create test configuration: {response.status_code} - {response.text}")
                return False
    
    async def upload_test_document(self, authorization_header: Optional[str] = None) -> bool:
        """Upload a test document to the configuration."""
        print("📄 Uploading test document...")
        
        test_content = """
        This is a test document for security feature testing.
        
        Machine Learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. It includes supervised learning, unsupervised learning, 
        and reinforcement learning approaches.
        
        Deep Learning is a specialized area of machine learning that uses neural networks 
        with multiple layers to model and understand complex patterns in data.
        
        Natural Language Processing (NLP) is a field that combines computational linguistics 
        with machine learning to help computers understand and process human language.
        """
        
        headers = {}
        if authorization_header:
            headers["Authorization"] = authorization_header
        
        # Create a temporary file-like object
        files = {
            "file": ("test_document.txt", test_content, "text/plain")
        }
        
        data = {
            "configuration_name": self.test_config_name,
            "metadata": json.dumps({
                "source": "test_upload",
                "category": "ai_ml",
                "level": "beginner"
            }),
            "process_immediately": "true"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/upload",
                files=files,
                data=data,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Test document uploaded successfully")
                return True
            else:
                print(f"❌ Failed to upload test document: {response.status_code} - {response.text}")
                return False
    
    async def test_query_without_auth(self) -> bool:
        """Test query endpoint without authentication (should fail)."""
        print("\n🔒 Testing query without authentication...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/query",
                json={
                    "query": "What is machine learning?",
                    "configuration_name": self.test_config_name,
                    "k": 3
                }
            )
            
            if response.status_code == 401:
                print("✅ Query correctly rejected without authentication")
                return True
            else:
                print(f"❌ Query should have been rejected but got: {response.status_code} - {response.text}")
                return False
    
    async def test_query_with_valid_auth(self) -> bool:
        """Test query endpoint with valid JWT token."""
        print("\n🔓 Testing query with valid JWT token...")
        
        # Create JWT token with metadata filter
        metadata_filter = {"category": "ai_ml"}
        token = self.create_jwt_token(
            subject="test_user",
            metadata_filter=metadata_filter
        )
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/query",
                json={
                    "query": "What is machine learning?",
                    "configuration_name": self.test_config_name,
                    "k": 3
                },
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Query successful with valid JWT token")
                print(f"   📊 Found {len(result.get('sources', []))} sources")
                print(f"   ⏱️ Processing time: {result.get('processing_time', 0):.3f}s")
                return True
            else:
                print(f"❌ Query failed with valid token: {response.status_code} - {response.text}")
                return False
    
    async def test_retrieve_with_metadata_filter(self) -> bool:
        """Test retrieve endpoint with JWT metadata filter."""
        print("\n🔍 Testing retrieve with JWT metadata filter...")
        
        # Create JWT token with specific metadata filter
        metadata_filter = {
            "$and": [
                {"source": "test_upload"},
                {"level": "beginner"}
            ]
        }
        token = self.create_jwt_token(
            subject="filtered_user",
            metadata_filter=metadata_filter
        )
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/retrieve",
                json={
                    "query": "deep learning neural networks",
                    "configuration_name": self.test_config_name,
                    "k": 5,
                    "include_metadata": True
                },
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Retrieve successful with JWT metadata filter")
                print(f"   📊 Found {len(result.get('documents', []))} documents")
                print(f"   ⏱️ Processing time: {result.get('processing_time', 0):.3f}s")
                
                # Check if metadata filtering was applied
                documents = result.get('documents', [])
                if documents:
                    print("   📋 Document metadata:")
                    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
                        metadata = doc.get('metadata', {})
                        print(f"      {i+1}. Source: {metadata.get('source', 'N/A')}, Level: {metadata.get('level', 'N/A')}")
                
                return True
            else:
                print(f"❌ Retrieve failed: {response.status_code} - {response.text}")
                return False
    
    async def test_invalid_jwt_token(self) -> bool:
        """Test with invalid JWT token."""
        print("\n❌ Testing with invalid JWT token...")
        
        invalid_token = "invalid.jwt.token"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/query",
                json={
                    "query": "What is machine learning?",
                    "configuration_name": self.test_config_name,
                    "k": 3
                },
                headers={"Authorization": f"Bearer {invalid_token}"}
            )
            
            if response.status_code == 401:
                print("✅ Invalid JWT token correctly rejected")
                return True
            else:
                print(f"❌ Invalid token should have been rejected but got: {response.status_code}")
                return False
    
    async def test_expired_jwt_token(self) -> bool:
        """Test with expired JWT token."""
        print("\n⏰ Testing with expired JWT token...")
        
        # Create expired token (expired 1 minute ago)
        payload = {
            "sub": "test_user",
            "iat": datetime.now(timezone.utc) - timedelta(minutes=2),
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
            "iss": "test_issuer",
            "aud": "test_audience"
        }
        
        expired_token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/query",
                json={
                    "query": "What is machine learning?",
                    "configuration_name": self.test_config_name,
                    "k": 3
                },
                headers={"Authorization": f"Bearer {expired_token}"}
            )
            
            if response.status_code == 401:
                print("✅ Expired JWT token correctly rejected")
                return True
            else:
                print(f"❌ Expired token should have been rejected but got: {response.status_code}")
                return False
    
    async def cleanup_test_configuration(self) -> bool:
        """Clean up the test configuration."""
        print("\n🧹 Cleaning up test configuration...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{self.base_url}/configurations/{self.test_config_name}"
            )
            
            if response.status_code == 200:
                print("✅ Test configuration cleaned up successfully")
                return True
            else:
                print(f"⚠️ Failed to clean up test configuration: {response.status_code}")
                return False
    
    async def run_all_tests(self) -> None:
        """Run all security feature tests."""
        print("🔐 Starting Security Feature Tests")
        print("=" * 50)
        
        test_results = []
        create_and_cleanup = not self.use_existing_config
        
        if create_and_cleanup:
            # Test 1: Create configuration
            result = await self.create_test_configuration()
            test_results.append(("Create Configuration", result))
            if not result:
                print("❌ Cannot continue tests without configuration")
                return
        else:
            print(f"\nℹ️ Using existing configuration: {self.test_config_name}")
        
        # Test 2: Upload test document
        valid_token = self.create_jwt_token()
        result = await self.upload_test_document(f"Bearer {valid_token}")
        test_results.append(("Upload Document", result))
        
        # Test 3: Query without auth
        result = await self.test_query_without_auth()
        test_results.append(("Query Without Auth", result))
        
        # Test 4: Query with valid auth
        result = await self.test_query_with_valid_auth()
        test_results.append(("Query With Valid Auth", result))
        
        # Test 5: Retrieve with metadata filter
        result = await self.test_retrieve_with_metadata_filter()
        test_results.append(("Retrieve With Metadata Filter", result))
        
        # Test 6: Invalid JWT token
        result = await self.test_invalid_jwt_token()
        test_results.append(("Invalid JWT Token", result))
        
        # Test 7: Expired JWT token
        result = await self.test_expired_jwt_token()
        test_results.append(("Expired JWT Token", result))
        
        if create_and_cleanup:
            # Cleanup
            await self.cleanup_test_configuration()
        
        # Print summary
        print("\n" + "=" * 50)
        print("🔐 Security Feature Test Summary")
        print("=" * 50)
        
        passed = 0
        total = len(test_results)
        
        for test_name, success in test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
            if success:
                passed += 1
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All security feature tests passed!")
        else:
            print("⚠️ Some tests failed. Check the implementation.")


async def main():
    """Main function to run the security tests."""
    parser = argparse.ArgumentParser(description="Test Security Feature for RAG2")
    parser.add_argument('--config', action='store_true', help='Use an existing configuration name (skips creation/cleanup)')
    parser.add_argument('--base-url', type=str, default="http://localhost:9000/api/v1", help='Base URL for the API')
    parser.add_argument('--jwt-secret', type=str, help='JWT secret to use for token creation (default is test secret)')
    parser.add_argument('--create-new', type=str, help='Force creation of a new test configuration')
    args = parser.parse_args()

    use_existing_config = args.config is not None and not args.create_new
    config_name = args.config if args.config else None
    config_name = "secure_jwt_demo"
    tester = SecurityFeatureTester(
        base_url=args.base_url,
        config_name=config_name,
        jwt_secret=args.jwt_secret,
        use_existing_config=use_existing_config
    )
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
