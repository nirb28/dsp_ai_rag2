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

def test_configure_configuration(client, sample_config):
    """Test configuring a configuration."""
    response = client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": "test configuration",
            "config": sample_config
        }
    )
    assert response.status_code == 200
    
    data = response.json()
    assert data["configuration_name"] == "test configuration"
    assert "config" in data
    assert "message" in data

def test_configure_configuration_invalid_config(client):
    """Test configuring a configuration with invalid config."""
    invalid_config = {
        "chunking": {
            "strategy": "invalid_strategy",  # Invalid strategy
            "chunk_size": 1000
        }
    }
    
    response = client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": "test configuration",
            "config": invalid_config
        }
    )
    assert response.status_code == 400

def test_get_configuration(client, sample_config):
    """Test getting configuration for a configuration."""
    # First set a configuration
    client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": "test configuration",
            "config": sample_config
        }
    )
    
    # Then get it
    response = client.get("/api/v1/configurations/test configuration")
    assert response.status_code == 200
    
    data = response.json()
    assert data["configuration_name"] == "test configuration"
    assert "config" in data

@patch('app.services.rag_service.RAGService.upload_document')
def test_upload_document_success(mock_upload, client, sample_text_file):
    """Test successful document upload."""
    from app.model_schemas import Document, DocumentStatus
    
    # Mock the upload_document method
    mock_document = Document(
        id="test_doc_id",
        filename="test.txt",
        content="test content",
        configuration_name="default",
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
                "configuration_name": "default",
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
                data={"configuration_name": "default"}
            )
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    finally:
        os.unlink(temp_file)

def test_upload_document_with_metadata(client, sample_text_file):
    """Test uploading document with metadata."""
    from app.model_schemas import Document, DocumentStatus
    
    with patch('app.services.rag_service.RAGService.upload_document') as mock_upload:
        mock_document = Document(
            id="test_doc_id",
            filename="test.txt",
            content="test content",
            configuration_name="default",
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
                    "configuration_name": "default",
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
                "configuration_name": "default",
                "metadata": "invalid json",  # Invalid JSON
            }
        )
    
    assert response.status_code == 400
    assert "Invalid metadata JSON format" in response.json()["detail"]

@patch('app.services.rag_service.RAGService.query')
def test_query_documents_success(mock_query, client):
    """Test successful document query."""
    from app.model_schemas import QueryResponse
    
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
        configuration_name="default"
    )
    mock_query.return_value = mock_response
    
    response = client.post(
        "/api/v1/query",
        json={
            "query": "test query",
            "configuration_name": "default",
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
    assert data["configuration_name"] == "default"

def test_query_documents_invalid_request(client):
    """Test query with invalid request."""
    response = client.post(
        "/api/v1/query",
        json={
            "query": "",  # Empty query
            "configuration_name": "default"
        }
    )
    
    assert response.status_code == 422  # Validation error

def test_query_documents_without_metadata(client):
    """Test query without including metadata."""
    from app.model_schemas import QueryResponse
    
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
            configuration_name="default"
        )
        mock_query.return_value = mock_response
        
        response = client.post(
            "/api/v1/query",
            json={
                "query": "test query",
                "configuration_name": "default",
                "include_metadata": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sources"][0]["metadata"] == {}

@patch('app.services.rag_service.RAGService.get_configurations')
def test_list_configurations(mock_get_configurations, client):
    """Test listing configurations."""
    # Mock the get_configurations method
    mock_get_configurations.return_value = [
        {
            "name": "configuration1",
            "document_count": 5,
            "config": {"chunking": {"strategy": "recursive_text"}}
        },
        {
            "name": "configuration2",
            "document_count": 3,
            "config": {"chunking": {"strategy": "fixed_size"}}
        }
    ]
    
    response = client.get("/api/v1/configurations")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["configurations"]) == 2
    assert data["configurations"][0]["name"] == "configuration1"
    assert data["configurations"][0]["document_count"] == 5

@patch('app.services.rag_service.RAGService.delete_configuration')
def test_delete_configuration_success(mock_delete, client):
    """Test successful configuration deletion."""
    mock_delete.return_value = True
    
    response = client.delete("/api/v1/configurations/test configuration")
    assert response.status_code == 200
    
    data = response.json()
    assert "deleted successfully" in data["message"]

@patch('app.services.rag_service.RAGService.delete_configuration')
def test_delete_configuration_not_found(mock_delete, client):
    """Test deleting non-existent collection."""
    mock_delete.return_value = False
    
    response = client.delete("/api/v1/configurations/nonexistent")
    assert response.status_code == 404


@patch('app.services.vector_store.FAISSVectorStore.similarity_search')
def test_retrieve_documents_basic(mock_similarity_search, client):
    """Test basic document retrieval without generating a response."""
    from langchain.schema import Document
    
    # Mock the similarity_search method
    mock_similarity_search.return_value = [
        (Document(page_content="test content 1", metadata={"filename": "doc1.txt"}), 0.95),
        (Document(page_content="test content 2", metadata={"filename": "doc2.txt"}), 0.85)
    ]
    
    # Mock the rag_service.vector_store_manager.configuration_exists method to return True
    with patch('app.api.endpoints.rag_service.vector_store_manager.configuration_exists', return_value=True):
        response = client.post(
            "/api/v1/retrieve",
            json={
                "query": "test query",
                "configuration_name": "default",
                "k": 5,
                "similarity_threshold": 0.7,
                "include_metadata": True
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert len(data["documents"]) == 2
    assert data["documents"][0]["content"] == "test content 1"
    assert data["documents"][0]["similarity_score"] == 0.95
    assert "metadata" in data["documents"][0]
    assert data["configuration_name"] == "default"
    assert data["total_found"] == 2


@patch('app.services.vector_store.FAISSVectorStore.similarity_search')
def test_retrieve_documents_with_config_override(mock_similarity_search, client):
    """Test document retrieval with config override."""
    from langchain.schema import Document
    
    # Mock the similarity_search method
    mock_similarity_search.return_value = [
        (Document(page_content="test content 1", metadata={"filename": "doc1.txt"}), 0.95),
        (Document(page_content="test content 2", metadata={"filename": "doc2.txt"}), 0.85)
    ]
    
    # Create a config override that changes embedding settings
    config_override = {
        "embedding": {
            "model": "custom-embedding-model",
            "dimensions": 768
        }
    }
    
    # Mock the rag_service.vector_store_manager.configuration_exists method to return True
    with patch('app.api.endpoints.rag_service.vector_store_manager.configuration_exists', return_value=True):
        with patch('app.services.embedding_service.EmbeddingService') as mock_embedding:
            response = client.post(
                "/api/v1/retrieve",
                json={
                    "query": "test query",
                    "configuration_name": "default",
                    "k": 5,
                    "similarity_threshold": 0.7,
                    "include_metadata": True,
                    "config": config_override
                }
            )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert len(data["documents"]) == 2
    assert data["configuration_name"] == "default"


@patch('app.services.rag_service.RAGService.query')
def test_query_documents_with_config_override(mock_query, client):
    """Test document query with config override."""
    from app.model_schemas import QueryResponse
    
    # Mock the query method
    mock_response = QueryResponse(
        query="test query",
        answer="test answer with custom configuration",
        sources=[
            {
                "content": "test content",
                "metadata": {"filename": "test.txt"},
                "similarity_score": 0.9
            }
        ],
        processing_time=0.5,
        configuration_name="default"
    )
    mock_query.return_value = mock_response
    
    # Create a config override that changes generation settings
    config_override = {
        "generation": {
            "provider": "openai_compatible",
            "model": "custom-model",
            "temperature": 0.2
        }
    }
    
    response = client.post(
        "/api/v1/query",
        json={
            "query": "test query",
            "configuration_name": "default",
            "include_metadata": True,
            "config": config_override
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert data["answer"] == "test answer with custom configuration"
    assert len(data["sources"]) == 1
    
    # Verify the config_override parameter was passed to the query method
    mock_query.assert_called_once()
    kwargs = mock_query.call_args.kwargs
    assert "config_override" in kwargs and kwargs["config_override"] is not None
