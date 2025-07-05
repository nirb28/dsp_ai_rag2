import pytest
import json
import os
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "services" in data
    assert "timestamp" in data

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["message"] == "RAG as a Service API"
    assert data["version"] == "1.0.0"
    assert data["docs"] == "/docs"

def test_configure_collection(client, sample_config):
    """Test configuring a collection."""
    response = client.post(
        "/api/v1/configure",
        json={
            "collection_name": "test_collection",
            "config": sample_config
        }
    )
    assert response.status_code == 200
    
    data = response.json()
    assert data["collection_name"] == "test_collection"
    assert "config" in data
    assert "message" in data

def test_configure_collection_invalid_config(client):
    """Test configuring a collection with invalid config."""
    invalid_config = {
        "chunking": {
            "strategy": "invalid_strategy",  # Invalid strategy
            "chunk_size": 1000
        }
    }
    
    response = client.post(
        "/api/v1/configure",
        json={
            "collection_name": "test_collection",
            "config": invalid_config
        }
    )
    assert response.status_code == 400

def test_get_configuration(client, sample_config):
    """Test getting configuration for a collection."""
    # First set a configuration
    client.post(
        "/api/v1/configure",
        json={
            "collection_name": "test_collection",
            "config": sample_config
        }
    )
    
    # Then get it
    response = client.get("/api/v1/configure/test_collection")
    assert response.status_code == 200
    
    data = response.json()
    assert data["collection_name"] == "test_collection"
    assert "config" in data

@patch('app.services.rag_service.RAGService.upload_document')
def test_upload_document_success(mock_upload, client, sample_text_file):
    """Test successful document upload."""
    from app.models import Document, DocumentStatus
    
    # Mock the upload_document method
    mock_document = Document(
        id="test_doc_id",
        filename="test.txt",
        content="test content",
        collection_name="default",
        file_size=100,
        file_type="txt",
        status=DocumentStatus.INDEXED
    )
    mock_upload.return_value = mock_document
    
    with open(sample_text_file, 'rb') as f:
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", f, "text/plain")},
            data={
                "collection_name": "default",
                "process_immediately": "true"
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "test_doc_id"
    assert data["filename"] == "test.txt"
    assert data["status"] == "indexed"

def test_upload_document_invalid_file_type(client):
    """Test uploading an invalid file type."""
    import tempfile
    
    # Create a file with unsupported extension
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        f.write(b"test content")
        temp_file = f.name
    
    try:
        with open(temp_file, 'rb') as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("test.xyz", f, "application/octet-stream")},
                data={"collection_name": "default"}
            )
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    finally:
        os.unlink(temp_file)

def test_upload_document_with_metadata(client, sample_text_file):
    """Test uploading document with metadata."""
    from app.models import Document, DocumentStatus
    
    with patch('app.services.rag_service.RAGService.upload_document') as mock_upload:
        mock_document = Document(
            id="test_doc_id",
            filename="test.txt",
            content="test content",
            collection_name="default",
            file_size=100,
            file_type="txt",
            status=DocumentStatus.INDEXED
        )
        mock_upload.return_value = mock_document
        
        metadata = {"author": "test_author", "category": "test"}
        
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "collection_name": "default",
                    "metadata": json.dumps(metadata),
                    "process_immediately": "true"
                }
            )
        
        assert response.status_code == 200

def test_upload_document_invalid_metadata(client, sample_text_file):
    """Test uploading document with invalid metadata."""
    with open(sample_text_file, 'rb') as f:
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", f, "text/plain")},
            data={
                "collection_name": "default",
                "metadata": "invalid json",  # Invalid JSON
            }
        )
    
    assert response.status_code == 400
    assert "Invalid metadata JSON format" in response.json()["detail"]

@patch('app.services.rag_service.RAGService.query')
def test_query_documents_success(mock_query, client):
    """Test successful document query."""
    from app.models import QueryResponse
    
    # Mock the query method
    mock_response = QueryResponse(
        query="test query",
        answer="test answer",
        sources=[
            {
                "content": "test content",
                "metadata": {"filename": "test.txt"},
                "similarity_score": 0.9
            }
        ],
        processing_time=0.5,
        collection_name="default"
    )
    mock_query.return_value = mock_response
    
    response = client.post(
        "/api/v1/query",
        json={
            "query": "test query",
            "collection_name": "default",
            "k": 5,
            "similarity_threshold": 0.7,
            "include_metadata": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert data["answer"] == "test answer"
    assert len(data["sources"]) == 1
    assert data["collection_name"] == "default"

def test_query_documents_invalid_request(client):
    """Test query with invalid request."""
    response = client.post(
        "/api/v1/query",
        json={
            "query": "",  # Empty query
            "collection_name": "default"
        }
    )
    
    assert response.status_code == 422  # Validation error

def test_query_documents_without_metadata(client):
    """Test query without including metadata."""
    from app.models import QueryResponse
    
    with patch('app.services.rag_service.RAGService.query') as mock_query:
        mock_response = QueryResponse(
            query="test query",
            answer="test answer",
            sources=[
                {
                    "content": "test content",
                    "metadata": {"filename": "test.txt"},
                    "similarity_score": 0.9
                }
            ],
            processing_time=0.5,
            collection_name="default"
        )
        mock_query.return_value = mock_response
        
        response = client.post(
            "/api/v1/query",
            json={
                "query": "test query",
                "collection_name": "default",
                "include_metadata": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sources"][0]["metadata"] == {}

@patch('app.services.rag_service.RAGService.get_collections')
def test_list_collections(mock_get_collections, client):
    """Test listing collections."""
    # Mock the get_collections method
    mock_get_collections.return_value = [
        {
            "name": "collection1",
            "document_count": 5,
            "config": {"chunking": {"strategy": "recursive_text"}}
        },
        {
            "name": "collection2",
            "document_count": 3,
            "config": {"chunking": {"strategy": "fixed_size"}}
        }
    ]
    
    response = client.get("/api/v1/collections")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["collections"]) == 2
    assert data["collections"][0]["name"] == "collection1"
    assert data["collections"][0]["document_count"] == 5

@patch('app.services.rag_service.RAGService.delete_collection')
def test_delete_collection_success(mock_delete, client):
    """Test successful collection deletion."""
    mock_delete.return_value = True
    
    response = client.delete("/api/v1/collections/test_collection")
    assert response.status_code == 200
    
    data = response.json()
    assert "deleted successfully" in data["message"]

@patch('app.services.rag_service.RAGService.delete_collection')
def test_delete_collection_not_found(mock_delete, client):
    """Test deleting non-existent collection."""
    mock_delete.return_value = False
    
    response = client.delete("/api/v1/collections/nonexistent")
    assert response.status_code == 404
