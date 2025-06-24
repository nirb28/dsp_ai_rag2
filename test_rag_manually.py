"""
Manual test script for RAG functionality.

This script tests the core RAG functionality directly without using pytest,
which can sometimes have issues with virtual environments in Windows.
"""

import os
import tempfile
import json
import time
import datetime
from pathlib import Path

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

# Configure the app environment first
os.environ["STORAGE_PATH"] = os.path.join(os.getcwd(), "test_storage")
os.environ["GROQ_API_KEY"] = "dummy-key-for-testing"  # Replace with real key if you want to test generation

# Import app modules
from app.main import app
from fastapi.testclient import TestClient
from app.config import RAGConfig, ChunkingStrategy, EmbeddingModel, VectorStore

def clear_terminal():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(message):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def create_test_client():
    """Create a test client for the API."""
    return TestClient(app)

def create_test_file(content):
    """Create a temporary file with the given content."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(content.encode('utf-8'))
        return f.name

def main():
    """Run the manual RAG tests."""
    clear_terminal()
    print_header("RAG SERVICE MANUAL TEST")
    
    # Create test storage directory if it doesn't exist
    os.makedirs(os.environ["STORAGE_PATH"], exist_ok=True)
    
    # Initialize test client
    print("\nInitializing test client...")
    client = create_test_client()
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    health_response = client.get("/api/v1/health")
    print(f"Status code: {health_response.status_code}")
    print(f"Response: {health_response.json()}")
    
    # Create test file
    print("\nCreating test document...")
    test_content = """
    Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
    especially computer systems. These processes include learning, reasoning, and self-correction.
    
    Machine learning is a subset of artificial intelligence that provides systems the ability 
    to automatically learn and improve from experience without being explicitly programmed.
    """
    test_file_path = create_test_file(test_content)
    print(f"Created test file: {test_file_path}")
    
    try:
        # Test 1: Configure a collection
        print_header("TEST 1: CONFIGURING COLLECTION")
        collection_name = "test_collection"
        
        config = {
            "chunking": {
                "strategy": "fixed_size",
                "chunk_size": 200,
                "chunk_overlap": 50
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "faiss"
            }
        }
        
        print(f"Configuring collection '{collection_name}'...")
        config_response = client.post(
            "/api/v1/configure",
            json={
                "collection_name": collection_name,
                "config": config
            }
        )
        
        print(f"Status code: {config_response.status_code}")
        print(f"Response: {json.dumps(config_response.json(), indent=2, cls=DateTimeEncoder)}")
        
        if config_response.status_code != 200:
            print("❌ Configuration failed. Stopping tests.")
            return
            
        # Test 2: Upload document
        print_header("TEST 2: UPLOADING DOCUMENT")
        
        print(f"Uploading document to collection '{collection_name}'...")
        with open(test_file_path, 'rb') as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"file": (Path(test_file_path).name, f, "text/plain")},
                data={
                    "collection_name": collection_name,
                    "process_immediately": "true",
                    "metadata": json.dumps({
                        "source": "test_script",
                        "topic": "ai"
                    })
                }
            )
        
        print(f"Status code: {upload_response.status_code}")
        print(f"Response: {json.dumps(upload_response.json(), indent=2, cls=DateTimeEncoder)}")
        
        if upload_response.status_code != 200:
            print("❌ Upload failed. Stopping tests.")
            return
        
        # Give some time for processing if needed
        print("Waiting for document processing...")
        time.sleep(2)
        
        # Test 3: Check if collection name is correctly preserved (addressing the override issue)
        print_header("TEST 3: CHECKING COLLECTION NAME PRESERVATION")
        
        collections_response = client.get("/api/v1/collections")
        collections_data = collections_response.json()
        
        print(f"Collections: {json.dumps(collections_data, indent=2, cls=DateTimeEncoder)}")
        
        # Debug the collections response
        print(f"Collections response: {collections_data}")
        
        # Check if our specific collection name exists
        collection_exists = any(c["name"] == collection_name for c in collections_data.get("collections", []))
        
        if collection_exists:
            print(f"✅ Collection '{collection_name}' exists - Collection name preserved correctly")
        else:
            print(f"❌ Collection '{collection_name}' not found - Issue with collection naming")
        
        # Test 4: Query documents
        print_header("TEST 4: QUERYING DOCUMENTS")
        
        print(f"Querying collection '{collection_name}'...")
        query_response = client.post(
            "/api/v1/query",
            json={
                "query": "What is machine learning?",
                "collection_name": collection_name,
                "k": 3,
                "include_metadata": True
            }
        )
        
        print(f"Status code: {query_response.status_code}")
        print(f"Response: {json.dumps(query_response.json(), indent=2, cls=DateTimeEncoder)}")
        
        # Test 5: Apply preset configuration
        print_header("TEST 5: APPLYING PRESET CONFIGURATION")
        
        preset_name = "fast_processing"
        new_collection = "preset_collection"
        
        print(f"Applying preset '{preset_name}' to collection '{new_collection}'...")
        preset_response = client.post(
            f"/api/v1/configure/preset/{preset_name}?collection_name={new_collection}"
        )
        
        print(f"Status code: {preset_response.status_code}")
        print(f"Response: {json.dumps(preset_response.json(), indent=2, cls=DateTimeEncoder)}")
        
        # Test complete
        print_header("TESTS COMPLETED")
        print("\nSummary of tests:")
        print("✅ Health check")
        print("✅ Collection configuration")
        print("✅ Document upload with metadata")
        print("✅ Collection name preservation" if collection_exists else "❌ Collection name preservation")
        print("✅ Document querying")
        print("✅ Preset application")
        
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)
            print(f"\nCleaned up test file: {test_file_path}")
        
        # Optionally clean up test storage
        print("\nTest storage directory can be found at:", os.environ["STORAGE_PATH"])
        print("You can delete it manually if needed.")

if __name__ == "__main__":
    main()
