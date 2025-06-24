"""
Example usage of the RAG as a Service API.

This script demonstrates how to:
1. Set up a collection with custom configuration
2. Upload documents
3. Query the RAG system
4. Use different configuration presets
"""

import requests
import json
import time
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

def check_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print("❌ API health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False

def create_sample_document():
    """Create a sample document for testing."""
    content = """
    Artificial Intelligence and Machine Learning

    Artificial Intelligence (AI) is a broad field of computer science focused on creating 
    systems that can perform tasks that typically require human intelligence. These tasks 
    include learning, reasoning, problem-solving, perception, and language understanding.

    Machine Learning (ML) is a subset of AI that focuses on the development of algorithms 
    and statistical models that enable computer systems to improve their performance on 
    a specific task through experience, without being explicitly programmed.

    Deep Learning is a subset of machine learning that uses neural networks with multiple 
    layers (deep neural networks) to model and understand complex patterns in data. It has 
    been particularly successful in areas such as image recognition, natural language 
    processing, and speech recognition.

    Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
    between computers and humans through natural language. The ultimate objective of NLP 
    is to read, decipher, understand, and make sense of human language in a valuable way.

    Applications of AI include:
    - Autonomous vehicles
    - Medical diagnosis
    - Financial trading
    - Recommendation systems
    - Virtual assistants
    - Game playing (like chess and Go)
    - Image and speech recognition
    """
    
    sample_file = Path("sample_ai_document.txt")
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    return sample_file

def upload_document(file_path, collection_name="default", metadata=None):
    """Upload a document to the RAG system."""
    print(f"📄 Uploading document: {file_path}")
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "collection_name": collection_name,
            "process_immediately": True
        }
        
        if metadata:
            data["metadata"] = json.dumps(metadata)
        
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Document uploaded successfully!")
        print(f"   Document ID: {result['document_id']}")
        print(f"   Status: {result['status']}")
        return result
    else:
        print(f"❌ Upload failed: {response.text}")
        return None

def query_documents(query, collection_name="default", k=5):
    """Query the RAG system."""
    print(f"🔍 Querying: '{query}'")
    
    payload = {
        "query": query,
        "collection_name": collection_name,
        "k": k,
        "include_metadata": True
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Query successful!")
        print(f"   Answer: {result['answer']}")
        print(f"   Sources found: {len(result['sources'])}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        
        # Show sources
        for i, source in enumerate(result['sources'], 1):
            print(f"   Source {i}: {source['metadata'].get('filename', 'Unknown')} "
                  f"(similarity: {source['similarity_score']:.3f})")
        
        return result
    else:
        print(f"❌ Query failed: {response.text}")
        return None

def configure_collection(collection_name, config):
    """Configure a collection with custom settings."""
    print(f"⚙️ Configuring collection: {collection_name}")
    
    payload = {
        "collection_name": collection_name,
        "config": config
    }
    
    response = requests.post(f"{BASE_URL}/configure", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Configuration applied successfully!")
        return result
    else:
        print(f"❌ Configuration failed: {response.text}")
        return None

def apply_preset(preset_name, collection_name):
    """Apply a configuration preset."""
    print(f"🎛️ Applying preset '{preset_name}' to collection '{collection_name}'")
    
    response = requests.post(f"{BASE_URL}/configure/preset/{preset_name}?collection_name={collection_name}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Preset applied successfully!")
        return result
    else:
        print(f"❌ Preset application failed: {response.text}")
        return None

def list_collections():
    """List all collections."""
    print("📋 Listing collections...")
    
    response = requests.get(f"{BASE_URL}/collections")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Found {result['total_count']} collections:")
        
        for collection in result['collections']:
            print(f"   - {collection['name']}: {collection['document_count']} documents")
        
        return result
    else:
        print(f"❌ Failed to list collections: {response.text}")
        return None

def get_presets():
    """Get available configuration presets."""
    print("🎯 Getting available presets...")
    
    response = requests.get(f"{BASE_URL}/presets")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Available presets:")
        
        for name, description in result['description'].items():
            print(f"   - {name}: {description}")
        
        return result
    else:
        print(f"❌ Failed to get presets: {response.text}")
        return None

def main():
    """Main example workflow."""
    print("🚀 RAG as a Service - Example Usage\n")
    
    # Check if API is running
    if not check_health():
        return
    
    # Create sample document
    sample_file = create_sample_document()
    
    try:
        # Example 1: Basic usage with default configuration
        print("\n" + "="*50)
        print("Example 1: Basic Usage")
        print("="*50)
        
        # Upload document
        upload_result = upload_document(
            sample_file, 
            collection_name="basic_example",
            metadata={"topic": "AI/ML", "author": "Example"}
        )
        
        if upload_result:
            # Wait a moment for processing
            time.sleep(2)
            
            # Query the document
            query_documents("What is machine learning?", "basic_example")
            query_documents("What are applications of AI?", "basic_example")
        
        # Example 2: Using configuration presets
        print("\n" + "="*50)
        print("Example 2: Configuration Presets")
        print("="*50)
        
        # Get available presets
        get_presets()
        
        # Apply fast processing preset
        apply_preset("fast_processing", "fast_collection")
        
        # Upload document to fast collection
        upload_document(sample_file, "fast_collection")
        time.sleep(2)
        
        # Query with fast configuration
        query_documents("Explain deep learning", "fast_collection")
        
        # Example 3: Custom configuration
        print("\n" + "="*50)
        print("Example 3: Custom Configuration")
        print("="*50)
        
        # Define custom configuration
        custom_config = {
            "chunking": {
                "strategy": "recursive_text",
                "chunk_size": 800,
                "chunk_overlap": 100
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 16
            },
            "generation": {
                "model": "llama3-8b-8192",
                "temperature": 0.5,
                "max_tokens": 512
            },
            "retrieval_k": 3,
            "similarity_threshold": 0.8
        }
        
        # Apply custom configuration
        configure_collection("custom_collection", custom_config)
        
        # Upload document with custom config
        upload_document(sample_file, "custom_collection")
        time.sleep(2)
        
        # Query with custom configuration
        query_documents("What is natural language processing?", "custom_collection")
        
        # Example 4: Collection management
        print("\n" + "="*50)
        print("Example 4: Collection Management")
        print("="*50)
        
        # List all collections
        list_collections()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
    
    finally:
        # Cleanup
        if sample_file.exists():
            sample_file.unlink()
            print(f"🧹 Cleaned up sample file: {sample_file}")

if __name__ == "__main__":
    main()
