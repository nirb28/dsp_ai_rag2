"""
Example script demonstrating how to use the security features of the RAG API.

This script shows:
1. How to create a configuration with security enabled
2. How to generate JWT tokens with metadata filters
3. How to make authenticated API calls
4. How to handle authentication errors
"""

import asyncio
import json
import jwt
import httpx
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional


class RAGSecurityExample:
    """Example class demonstrating RAG API security usage."""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.jwt_secret = "example-secret-key-for-demo-purposes-only"
        self.config_name = "secure_demo_config"
    
    def create_jwt_token(
        self, 
        user_id: str,
        department: str = "engineering",
        access_level: str = "public",
        expires_in_hours: int = 1
    ) -> str:
        """Create a JWT token with user information and metadata filters."""
        
        # Define metadata filter based on user's department and access level
        metadata_filter = {
            "department": department,
            "access_level": access_level
        }
        
        # Create JWT payload
        payload = {
            "sub": user_id,  # Subject (user identifier)
            "iat": datetime.now(timezone.utc),  # Issued at
            "exp": datetime.now(timezone.utc) + timedelta(hours=expires_in_hours),  # Expiration
            "iss": "demo-auth-service",  # Issuer
            "aud": "rag-api",  # Audience
            "metadata_filter": metadata_filter  # Custom claim for document filtering
        }
        
        # Sign and return the token
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    async def create_secure_configuration(self) -> bool:
        """Create a RAG configuration with security enabled."""
        print("🔧 Creating secure configuration...")
        
        config = {
            "configuration_name": self.config_name,
            "config": {
                "chunking": {
                    "strategy": "recursive_text",
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                },
                "vector_store": {
                    "type": "faiss",
                    "index_path": f"./storage/{self.config_name}_index",
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
                    "jwt_issuer": "demo-auth-service",
                    "jwt_audience": "rag-api",
                    "jwt_require_exp": True,
                    "jwt_require_iat": True,
                    "jwt_leeway": 10
                }
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/configurations", json=config)
            
            if response.status_code == 200:
                print("✅ Secure configuration created successfully")
                return True
            else:
                print(f"❌ Failed to create configuration: {response.status_code} - {response.text}")
                return False
    
    async def upload_sample_documents(self) -> bool:
        """Upload sample documents with different metadata for testing."""
        print("📄 Uploading sample documents...")
        
        # Create a token for uploading (admin access)
        admin_token = self.create_jwt_token("admin", "admin", "all")
        
        documents = [
            {
                "content": """
                Machine Learning Basics for Engineering Teams
                
                Machine learning is a subset of artificial intelligence that enables computers 
                to learn and make decisions from data without being explicitly programmed.
                
                Key concepts include:
                - Supervised learning: Learning from labeled examples
                - Unsupervised learning: Finding patterns in unlabeled data
                - Neural networks: Computing systems inspired by biological neural networks
                """,
                "filename": "ml_engineering_guide.txt",
                "metadata": {
                    "department": "engineering",
                    "access_level": "public",
                    "topic": "machine_learning",
                    "author": "Engineering Team"
                }
            },
            {
                "content": """
                Advanced AI Research - Internal Document
                
                This document contains proprietary research on advanced AI algorithms
                developed by our research team. This information is confidential and
                should only be accessed by authorized research personnel.
                
                Topics covered:
                - Novel neural architectures
                - Proprietary training techniques
                - Performance benchmarks
                - Future research directions
                """,
                "filename": "ai_research_internal.txt",
                "metadata": {
                    "department": "research",
                    "access_level": "internal",
                    "topic": "ai_research",
                    "author": "Research Team"
                }
            },
            {
                "content": """
                Marketing Guide: AI Product Positioning
                
                This guide helps the marketing team understand how to position
                our AI products in the market. It includes customer personas,
                competitive analysis, and messaging strategies.
                
                Key points:
                - Target audience identification
                - Value proposition development
                - Competitive differentiation
                - Go-to-market strategy
                """,
                "filename": "marketing_ai_guide.txt",
                "metadata": {
                    "department": "marketing",
                    "access_level": "public",
                    "topic": "marketing",
                    "author": "Marketing Team"
                }
            }
        ]
        
        for doc in documents:
            files = {"file": (doc["filename"], doc["content"], "text/plain")}
            data = {
                "configuration_name": self.config_name,
                "metadata": json.dumps(doc["metadata"]),
                "process_immediately": "true"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/upload",
                    files=files,
                    data=data,
                    headers={"Authorization": f"Bearer {admin_token}"}
                )
                
                if response.status_code == 200:
                    print(f"✅ Uploaded: {doc['filename']}")
                else:
                    print(f"❌ Failed to upload {doc['filename']}: {response.status_code}")
                    return False
        
        return True
    
    async def demonstrate_user_access(self, user_id: str, department: str, access_level: str):
        """Demonstrate how different users see different documents based on their access."""
        print(f"\n👤 Testing access for user: {user_id} (dept: {department}, level: {access_level})")
        
        # Create token for this user
        token = self.create_jwt_token(user_id, department, access_level)
        
        # Query the system
        query_request = {
            "query": "Tell me about artificial intelligence and machine learning",
            "configuration_name": self.config_name,
            "k": 10,
            "include_metadata": True
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/query",
                json=query_request,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                sources = result.get("sources", [])
                print(f"✅ Query successful - Found {len(sources)} accessible documents:")
                
                for i, source in enumerate(sources, 1):
                    metadata = source.get("metadata", {})
                    print(f"   {i}. {metadata.get('topic', 'Unknown')} "
                          f"(dept: {metadata.get('department', 'N/A')}, "
                          f"level: {metadata.get('access_level', 'N/A')})")
                
                print(f"   💬 Answer: {result.get('answer', 'No answer')[:100]}...")
                
            else:
                print(f"❌ Query failed: {response.status_code} - {response.text}")
    
    async def demonstrate_unauthorized_access(self):
        """Demonstrate what happens when no authentication is provided."""
        print(f"\n🚫 Testing unauthorized access (no token)...")
        
        query_request = {
            "query": "Tell me about AI",
            "configuration_name": self.config_name
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/query", json=query_request)
            
            if response.status_code == 401:
                print("✅ Correctly rejected unauthorized request")
                print(f"   Response: {response.json().get('detail', 'No details')}")
            else:
                print(f"❌ Expected 401 but got: {response.status_code}")
    
    async def demonstrate_expired_token(self):
        """Demonstrate what happens with an expired token."""
        print(f"\n⏰ Testing expired token...")
        
        # Create an expired token
        payload = {
            "sub": "test_user",
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),  # Expired 1 hour ago
            "iss": "demo-auth-service",
            "aud": "rag-api"
        }
        expired_token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        query_request = {
            "query": "Tell me about AI",
            "configuration_name": self.config_name
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/query",
                json=query_request,
                headers={"Authorization": f"Bearer {expired_token}"}
            )
            
            if response.status_code == 401:
                print("✅ Correctly rejected expired token")
                print(f"   Response: {response.json().get('detail', 'No details')}")
            else:
                print(f"❌ Expected 401 but got: {response.status_code}")
    
    async def cleanup(self):
        """Clean up the test configuration."""
        print(f"\n🧹 Cleaning up configuration: {self.config_name}")
        
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{self.base_url}/configurations/{self.config_name}")
            
            if response.status_code == 200:
                print("✅ Configuration cleaned up successfully")
            else:
                print(f"⚠️ Failed to clean up: {response.status_code}")
    
    async def run_demo(self):
        """Run the complete security demonstration."""
        print("🔐 RAG API Security Feature Demonstration")
        print("=" * 50)
        
        try:
            # Setup
            if not await self.create_secure_configuration():
                return
            
            if not await self.upload_sample_documents():
                return
            
            # Demonstrate different user access patterns
            await self.demonstrate_user_access("alice", "engineering", "public")
            await self.demonstrate_user_access("bob", "research", "internal")
            await self.demonstrate_user_access("charlie", "marketing", "public")
            
            # Demonstrate security enforcement
            await self.demonstrate_unauthorized_access()
            await self.demonstrate_expired_token()
            
            print(f"\n🎉 Security demonstration completed successfully!")
            print(f"\nKey takeaways:")
            print(f"- Users only see documents matching their metadata filters")
            print(f"- Unauthorized requests are properly rejected")
            print(f"- Expired tokens are handled securely")
            print(f"- Different departments have access to different content")
            
        finally:
            # Cleanup
            await self.cleanup()


async def main():
    """Main function to run the security demonstration."""
    demo = RAGSecurityExample()
    await demo.run_demo()


if __name__ == "__main__":
    print("Starting RAG API Security Demonstration...")
    print("Make sure the RAG API server is running on http://localhost:8000")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
