"""
Test for batch document processing from a specified folder.

This test allows loading all documents from a folder in sample-docs,
processing them through the RAG pipeline, and running queries.

python tests/test_batch_document_processing.py
python tests/test_batch_document_processing.py --folder your_folder_name --query "Your first query" --query "Your second query"

python tests/test_batch_document_processing.py --folder ml_ai_basics

"""

# Fix Python path for direct script execution
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import requests

import pytest
import tempfile
import json
import time
import argparse
from fastapi.testclient import TestClient

from app.main import app
from app.config import RAGConfig, EmbeddingModel


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def temp_storage(monkeypatch):
    """Create a temporary storage directory and update the environment variable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv("STORAGE_PATH", temp_dir)
        yield temp_dir


def check_model_server_available(url="http://localhost:8001"):
    """Check if the model server is running and available."""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture
def batch_config():
    """Create a configuration optimized for batch document processing."""
    # Check if model server is running, if not use SentenceTransformers instead
    model_server_available = check_model_server_available()
    
    if not model_server_available:
        print("WARNING: Model server not available at http://localhost:8001. Using SentenceTransformers instead.")
        embedding_config = {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    else:
        print("Using local model server for embeddings")
        embedding_config = {
            "model": "local-model-server",
            "server_url": "http://zahrt.sas.upenn.edu:9001"
        }
        
    return {
        "chunking": {
            "strategy": "recursive_text",  # Better for longer documents
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        "embedding": embedding_config,
        "vector_store": {
            "type": "faiss"
        },
        "generation": {
            "model": "llama3-8b-8192",
            "temperature": 0.2,  # Lower temperature for more factual responses
            "max_tokens": 500,  # Longer responses
            "top_p": 0.9
        }
    }


def get_file_extension(file_path):
    """Get the extension of a file."""
    return Path(file_path).suffix.lower()


def get_mime_type(file_extension):
    """Get the MIME type for a file extension."""
    mime_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.md': 'text/markdown'
    }
    return mime_types.get(file_extension, 'application/octet-stream')


def process_documents_in_folder(client, folder_name, configuration_name):
    """
    Process all documents in a folder and index them in the RAG system.
    
    Args:
        client: FastAPI test client
        folder_name: Name of the folder under sample-docs
        configuration_name: Name for the configuration to create
    
    Returns:
        dict: Summary of processing results
    """
    # Path to the documents folder
    docs_folder = Path(__file__).parent / "sample-docs" / folder_name
    
    if not docs_folder.exists() or not docs_folder.is_dir():
        raise ValueError(f"Folder '{folder_name}' not found in sample-docs")
    
    # Get all files in the folder
    files = [f for f in docs_folder.glob("**/*") if f.is_file()]
    
    if not files:
        raise ValueError(f"No files found in folder '{folder_name}'")
    
    print(f"\nProcessing {len(files)} documents from '{folder_name}' folder...")
    
    results = {
        "total_files": len(files),
        "processed_files": 0,
        "failed_files": 0,
        "file_details": []
    }
    
    # Process each file
    for file_path in files:
        print(f"Processing: {file_path.name}")
        
        try:
            # Determine file type
            file_extension = get_file_extension(file_path)
            mime_type = get_mime_type(file_extension)
            
            # Upload the document
            with open(file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/upload",
                    files={"file": (file_path.name, f, mime_type)},
                    data={
                        "configuration_name": configuration_name,
                        "metadata": json.dumps({
                            "source": folder_name, 
                            "filename": file_path.name,
                            "filepath": str(file_path),
                            "title": file_path.stem.replace('_', ' ').replace('-', ' ').title(),
                            "file_type": file_extension[1:] if file_extension else "unknown"
                        }),
                        "process_immediately": "true"
                    }
                )
            
            if response.status_code == 200:
                results["processed_files"] += 1
                data = response.json()
                results["file_details"].append({
                    "filename": file_path.name,
                    "status": "success",
                    "document_id": data.get("document_id", None)
                })
                print(f"✓ Successfully processed {file_path.name}")
            else:
                results["failed_files"] += 1
                results["file_details"].append({
                    "filename": file_path.name,
                    "status": "failed",
                    "error": response.text
                })
                print(f"✗ Failed to process {file_path.name}: {response.text}")
                
        except Exception as e:
            results["failed_files"] += 1
            results["file_details"].append({
                "filename": file_path.name,
                "status": "error",
                "error": str(e)
            })
            print(f"✗ Error processing {file_path.name}: {str(e)}")
    
    print(f"\nProcessing complete: {results['processed_files']} succeeded, {results['failed_files']} failed")
    return results


def run_queries(client, configuration_name, queries):
    """
    Run a list of queries against the configuration and return results.
    
    Args:
        client: FastAPI test client
        configuration_name: Name of the configuration to query
        queries: List of query strings
        
    Returns:
        list: Query results
    """
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nRunning query {i}: '{query}'")
        
        response = client.post(
            "/api/v1/query",
            json={
                "query": query,
                "configuration_name": configuration_name,
                "include_metadata": True,
                "similarity_threshold": 0.5,  # Lower threshold to include more potential matches
                "max_sources": 5,  # Include more sources in response
                "generate": True  # Ensure we generate a response
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Print answer and sources
            print("\nAnswer:")
            print(data["answer"])
            
            print("\nSources:")
            if data["sources"]:
                for i, source in enumerate(data["sources"], 1):
                    metadata = source.get('metadata', {})
                    filename = metadata.get('filename', 'Unknown')
                    title = metadata.get('title', '')
                    display_name = title if title else filename
                    
                    content_preview = source.get('page_content', '')[:100]
                    
                    print(f"  {i}. {display_name} - {content_preview}...")
            else:
                print("  No sources found for this query")
            
            results.append({
                "query": query,
                "answer": data["answer"],
                "sources": data["sources"],
                "status": "success"
            })
        else:
            print(f"Query failed: {response.text}")
            results.append({
                "query": query,
                "status": "failed",
                "error": response.text
            })
    
    return results


def test_batch_document_processing(client, temp_storage, batch_config, folder_name="401k"):
    """Test batch processing of documents from a folder."""
    # 1. Configure a collection
    configuration_name = f"batch_{folder_name}_test"
    
    config_response = client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": configuration_name,
            "config": batch_config
        }
    )
    assert config_response.status_code == 200
    
    # 2. Process all documents in the folder
    processing_results = process_documents_in_folder(client, folder_name, configuration_name)
    assert processing_results["processed_files"] > 0
    
    # 3. Wait for indexing to complete
    print("\nWaiting for indexing to complete...")
    time.sleep(2)
    
    # 4. Check collection info
    configurations_response = client.get("/api/v1/configurations")
    assert configurations_response.status_code == 200
    
    configurations_data = configurations_response.json()
    configuration_info = next((c for c in configurations_data["configurations"] if c["name"] == configuration_name), None)
    
    assert configuration_info is not None
    print(f"\nConfiguration '{configuration_name}' contains {configuration_info.get('document_count', 0)} chunks")
    
    # 5. Run sample queries
    queries = [
        "What are 401k plans?",
        "What are the benefits of 401k plans for small businesses?",
        "What are the contribution limits for 401k plans?",
        "How can employers set up a 401k plan?"
    ]
    
    query_results = run_queries(client, configuration_name, queries)
    
    # 6. Clean up
    delete_response = client.delete(f"/api/v1/configurations/{configuration_name}")
    assert delete_response.status_code == 200
    print(f"\nConfiguration '{configuration_name}' deleted")
    
    return {
        "processing_results": processing_results,
        "query_results": query_results
    }


if __name__ == "__main__":
    # This allows running the script directly with command line arguments
    parser = argparse.ArgumentParser(description='Process documents from a folder and run queries.')
    parser.add_argument('--folder', '-f', type=str, default="401k", 
                        help='Folder name under sample-docs to process')
    parser.add_argument('--query', '-q', type=str, action='append',
                        help='Query to run (can be specified multiple times)')
    
    args = parser.parse_args()
    
    # Create test client
    client = TestClient(app)
    
    # Use environment variable for storage or default to temp directory
    if "STORAGE_PATH" not in os.environ:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["STORAGE_PATH"] = temp_dir
            
            # Run the test with specified folder
            config = {
                "chunking": {
                    "strategy": "recursive_text",
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                },
                "vector_store": {
                    "type": "faiss",
                    "index_path": "./storage/faiss_index",
                    "dimension": 384
                },
                "embedding": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 32,
                    "server_url": "http://zahrt.sas.upenn.edu:9001"
                },
                "generation": {
                    "model": "llama3-vllm",
                    "provider": "triton",
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_p": 0.9,
                    "top_k": 5,
                    "server_url": "http://zahrt.sas.upenn.edu:8000"
                }
            }
            
            configuration_name = f"batch_{args.folder}_test"
            
            # Configure collection
            client.post("/api/v1/configurations", json={"configuration_name": configuration_name, "config": config})
            
            # Process documents
            process_documents_in_folder(client, args.folder, configuration_name)
            
            # Wait for indexing
            time.sleep(2)
            
            # Only run queries if explicitly provided
            if args.query:
                run_queries(client, configuration_name, args.query)
            else:
                print("\nNo queries provided. Use --query to specify queries to run.")
            
            # Clean up
            client.delete(f"/api/v1/configurations/{configuration_name}")
    else:
        # Environment variable already set, just run the test
        config = {
            "chunking": {
                "strategy": "recursive_text",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "faiss"
            },
            "generation": {
                "model": "llama3-8b-8192",
                "temperature": 0.2,
                "max_tokens": 500,
                "top_p": 0.9
            }
        }
        
        configuration_name = f"batch_{args.folder}_test"
        
        # Configure collection
        client.post("/api/v1/configurations", json={"configuration_name": configuration_name, "config": config})
        
        # Process documents
        process_documents_in_folder(client, args.folder, configuration_name)
        
        # Wait for indexing
        time.sleep(2)
        
        # Only run queries if explicitly provided
        if args.query:
            run_queries(client, configuration_name, args.query)
        else:
            print("\nNo queries provided. Use --query to specify queries to run.")
        
        # Clean up
        client.delete(f"/api/v1/configurations/{configuration_name}")
