"""
Integration tests for the RAG service.

These tests verify the entire RAG pipeline works correctly without mocking
the service components, ensuring proper interaction between all layers.
"""

import os
import pytest
import tempfile
import json
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app
from app.config import RAGConfig, ChunkingStrategy, EmbeddingModel, VectorStore


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


@pytest.fixture
def sample_text():
    """Create a sample text content for testing."""
    return """
    Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
    especially computer systems. These processes include learning (the acquisition of information 
    and rules for using the information), reasoning (using rules to reach approximate or definite 
    conclusions) and self-correction.

    Machine learning is a subset of artificial intelligence (AI) that provides systems the ability 
    to automatically learn and improve from experience without being explicitly programmed. 
    Machine learning focuses on the development of computer programs that can access data and use 
    it to learn for themselves.

    Deep learning is part of a broader family of machine learning methods based on artificial 
    neural networks with representation learning. Learning can be supervised, semi-supervised or 
    unsupervised.
    """


@pytest.fixture
def sample_text_file(sample_text):
    """Create a sample text file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(sample_text.encode('utf-8'))
        file_path = f.name
    
    yield file_path
    
    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def config_for_testing():
    """Create a minimal configuration for testing."""
    return {
        "chunking": {
            "strategy": "fixed_size",
            "chunk_size": 200,
            "chunk_overlap": 50
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "server_url": "http://zahrt.sas.upenn.edu:9001"
        },
        "vector_store": {
            "type": "faiss"
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


def test_full_rag_pipeline(client, temp_storage, sample_text_file, config_for_testing):
    """Test the complete RAG pipeline from configuration to querying."""
    # 1. Configure a collection
    configuration_name = "test_integration"
    
    config_response = client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": configuration_name,
            "config": config_for_testing
        }
    )
    assert config_response.status_code == 200
    
    # 2. Upload a document with immediate processing
    with open(sample_text_file, 'rb') as f:
        upload_response = client.post(
            "/api/v1/upload",
            files={"file": (Path(sample_text_file).name, f, "text/plain")},
            data={
                "configuration_name": configuration_name,
                "process_immediately": "true",
                "metadata": json.dumps({
                    "source": "integration_test",
                    "topic": "artificial_intelligence"
                })
            }
        )
    
    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    assert upload_data["status"] == "processing" or upload_data["status"] == "indexed"
    
    # 3. Query the document
    query_response = client.post(
        "/api/v1/query",
        json={
            "query": "What is machine learning?",
            "configuration_name": configuration_name,
            "k": 3,
            "similarity_threshold": 0.6,
            "include_metadata": True
        }
    )
    
    assert query_response.status_code == 200
    query_data = query_response.json()
    
    assert "answer" in query_data
    assert len(query_data["answer"]) > 0
    assert "sources" in query_data
    
    # 4. Verify the sources include metadata
    assert len(query_data["sources"]) > 0
    for source in query_data["sources"]:
        assert "metadata" in source
        assert "similarity_score" in source
    
    # 5. Check that we can list configurations and our test configuration appears
    collections_response = client.get("/api/v1/configurations")
    assert collections_response.status_code == 200
    
    configurations_data = collections_response.json()
    configuration_names = [c["name"] for c in configurations_data["configurations"]]
    assert configuration_name in configuration_names
    
    # 6. Clean up by deleting the configuration
    delete_response = client.delete(f"/api/v1/configurations/{configuration_name}")
    assert delete_response.status_code == 200


def test_configuration_name_respected(client, temp_storage, sample_text_file):
    """Test that the configuration name is properly respected in the configuration.
    
    This test addresses a reported issue where configuration names were being
    forcibly overridden to "user_{user_id}" regardless of configuration settings.
    """
    configuration_name = "specific_configuration_name"
    
    # 1. Configure the configuration with a specific name
    config_response = client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": configuration_name,
            "config": {
                "chunking": {"strategy": "fixed_size", "chunk_size": 300},
                "embedding": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 64,
                    "server_url": "http://zahrt.sas.upenn.edu:9001"
                }
            }
        }
    )
    assert config_response.status_code == 200
    
    # 2. Upload a document to this collection
    with open(sample_text_file, 'rb') as f:
        upload_response = client.post(
            "/api/v1/upload",
            files={"file": (Path(sample_text_file).name, f, "text/plain")},
            data={
                "configuration_name": configuration_name,
                "process_immediately": "true"
            }
        )
    
    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    assert upload_data["configuration_name"] == configuration_name
    
    # 3. Verify the configuration exists with the exact name
    collections_response = client.get("/api/v1/configurations")
    configurations_data = collections_response.json()
    
    assert any(c["name"] == configuration_name for c in configurations_data["configurations"]), \
        f"Configuration '{configuration_name}' not found in the list of configurations"


def test_vectorize_now_validation(client, temp_storage, sample_text_file):
    """Test that document upload with vectorize_now=true works correctly.
    
    This test addresses a reported issue where validation errors occurred when
    vectorize_now=true because document_type, mime_type, and size_bytes fields
    were missing when creating a Document object.
    """
    configuration_name = "vectorize_validation_test"
    
    # 1. Configure the configuration
    client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": configuration_name,
            "config": {
                "chunking": {"strategy": "fixed_size", "chunk_size": 200},
                "embedding": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "server_url": "http://zahrt.sas.upenn.edu:9001"
                }
            }
        }
    )
    
    # 2. Upload a document with vectorize_now=true (process_immediately)
    with open(sample_text_file, 'rb') as f:
        upload_response = client.post(
            "/api/v1/upload",
            files={"file": (Path(sample_text_file).name, f, "text/plain")},
            data={
                "configuration_name": configuration_name,
                "process_immediately": "true"
            }
        )
    
    # 3. The response should be successful (no validation errors)
    assert upload_response.status_code == 200, \
        f"Upload failed with status {upload_response.status_code}: {upload_response.text}"
    
    upload_data = upload_response.json()
    assert "document_id" in upload_data, \
        f"Response missing document_id: {upload_data}"

