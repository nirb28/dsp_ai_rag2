import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
import os

from app.main import app
from app.config import settings

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    original_storage_path = settings.STORAGE_PATH
    settings.STORAGE_PATH = temp_dir
    
    yield temp_dir
    
    # Cleanup
    settings.STORAGE_PATH = original_storage_path
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    content = """
    This is a sample document for testing the RAG system.
    It contains multiple paragraphs with different topics.
    
    The first topic is about artificial intelligence and machine learning.
    AI has revolutionized many industries and continues to grow.
    
    The second topic discusses natural language processing.
    NLP enables computers to understand and generate human language.
    
    The third topic covers information retrieval systems.
    These systems help users find relevant information quickly.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_file_path = f.name
    
    yield temp_file_path
    
    # Cleanup
    os.unlink(temp_file_path)

@pytest.fixture
def sample_config():
    """Sample RAG configuration for testing."""
    return {
        "configuration_name": "test_configuration",
        "chunking": {
            "strategy": "recursive_text",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "vector_store": {
            "type": "faiss",
            "index_path": "./test_storage/faiss_index",
            "dimension": 384
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 32
        },
        "generation": {
            "model": "llama3-8b-8192",
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9
        },
        "retrieval_k": 5,
        "similarity_threshold": 0.7
    }
