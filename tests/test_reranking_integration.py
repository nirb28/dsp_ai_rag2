"""
Integration tests for reranking functionality using both local model server and Triton.

These tests verify that the RAG pipeline correctly uses reranking capabilities
with different backends.
"""

import os
import pytest
import tempfile
import json
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app
from app.config import (
    RAGConfig, ChunkingStrategy, EmbeddingModel, VectorStore,
    RerankerModel, RerankerConfig
)


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp:
        content = """
        Artificial Intelligence (AI) is transforming industries across the globe.
        Machine learning, a subset of AI, enables computers to learn from data and improve over time.
        Deep learning is a specialized form of machine learning that uses neural networks with many layers.
        Natural Language Processing (NLP) allows computers to understand, interpret, and generate human language.
        Computer vision enables machines to see and interpret visual information from the world.
        Reinforcement learning is an area of machine learning where agents learn to make decisions by taking actions.
        The Transformer architecture revolutionized NLP with models like BERT and GPT.
        Large Language Models (LLMs) can generate human-like text and understand context effectively.
        AI ethics is concerned with ensuring AI systems are fair, transparent, and beneficial to humanity.
        Responsible AI development includes considerations of bias, privacy, and societal impact.
        """
        temp.write(content.encode())
        yield temp.name
    os.unlink(temp.name)


def test_reranking_with_local_model_server(client, sample_text_file):
    """Test the complete RAG pipeline with local model server reranking."""
    configuration_name = "reranker_local_model_server_test"
    
    # 1. Configure the system with reranking enabled using local model server
    client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": configuration_name,
            "config": {
                "chunking": {
                    "strategy": "fixed_size",
                    "chunk_size": 200,
                    "chunk_overlap": 50
                },
                "reranking": {
                    "enabled": True,
                    "model": "local-model-server",
                    "model_name": "ms-marco-minilm",
                    "server_url": "http://localhost:9001",
                    "top_n": 5,
                    "score_threshold": 0.2
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
                    "max_tokens": 350,
                    "top_p": 0.9,
                    "top_k": 5,
                    "server_url": "http://zahrt.sas.upenn.edu:8000"
                } 
            }
        }
    )
    
    # 2. Upload a document
    with open(sample_text_file, 'rb') as f:
        upload_response = client.post(
            "/api/v1/upload",
            files={"file": (Path(sample_text_file).name, f, "text/plain")},
            data={
                "configuration_name": configuration_name,
                "process_immediately": "true"
            }
        )
    
    # Check upload succeeded
    assert upload_response.status_code == 200, \
        f"Upload failed with status {upload_response.status_code}: {upload_response.text}"
    
    upload_data = upload_response.json()
    document_id = upload_data["document_id"]
    
    # 3. Wait for processing to complete
    # In a real test, you might want to add polling logic here
    # For simplicity, we'll assume processing is fast in the test env
    
    # 4. Query with a relevant question that should trigger reranking and generation
    query_response = client.post(
        "/api/v1/query",
        json={
            "configuration_name": configuration_name,
            "query": "What is machine learning?",
            "document_ids": [document_id],
            "generate": True
        }
    )
    
    # Check query succeeded
    assert query_response.status_code == 200, \
        f"Query failed with status {query_response.status_code}: {query_response.text}"
    
    query_data = query_response.json()
    
    # 5. Verify that we got results and they have reranking information
    assert "documents" in query_data, "Response missing documents array"
    assert len(query_data["documents"]) > 0, "No documents returned"
    
    # Check if the most relevant document has the expected content
    first_doc = query_data["documents"][0]
    assert "machine learning" in first_doc["content"].lower(), \
        f"First document does not mention machine learning: {first_doc['content']}"
    
    # If reranking was applied, documents should have original_similarity_score
    assert "original_similarity_score" in first_doc, \
        "Document doesn't have original_similarity_score, suggesting reranking wasn't applied"
        
    # 6. Verify that we got a generated answer
    assert "answer" in query_data, "Response missing generated answer"
    assert len(query_data["answer"]) > 0, "Empty generated answer"
    
    # Check if the answer discusses machine learning
    assert any(term in query_data["answer"].lower() for term in ["machine learning", "ml", "artificial intelligence"]), \
        f"Generated answer not relevant to machine learning: {query_data['answer']}"


def test_reranking_combined_with_generation(client, sample_text_file):
    """Test the complete RAG pipeline with reranking and generation combined."""
    configuration_name = "reranker_and_generator_test"
    
    # 1. Configure the system with both reranking and generation
    client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": configuration_name,
            "config": {
                "chunking": {
                    "strategy": "fixed_size",
                    "chunk_size": 200,
                    "chunk_overlap": 50
                },
                "embedding": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "server_url": "http://localhost:9001"
                },
                "vector_store": {
                    "type": "faiss"
                },
                "reranking": {
                    "enabled": True,
                    "model": "local-model-server",
                    "model_name": "ms-marco-minilm", 
                    "server_url": "http://localhost:9001",
                    "top_n": 5,
                    "score_threshold": 0.2
                },
                "generation": {
                    "model": "llama3-vllm",
                    "provider": "triton",
                    "temperature": 0.3,  # Lower temperature for more focused answers
                    "max_tokens": 350,
                    "top_p": 0.9,
                    "top_k": 5,
                    "server_url": "http://zahrt.sas.upenn.edu:8000"
                }
            }
        }
    )
    
    # 2. Upload a document
    with open(sample_text_file, 'rb') as f:
        upload_response = client.post(
            "/api/v1/upload",
            files={"file": (Path(sample_text_file).name, f, "text/plain")},
            data={
                "configuration_name": configuration_name,
                "process_immediately": "true"
            }
        )
    
    # Check upload succeeded
    assert upload_response.status_code == 200, \
        f"Upload failed with status {upload_response.status_code}: {upload_response.text}"
    
    upload_data = upload_response.json()
    document_id = upload_data["document_id"]
    
    # 3. Query with specialized prompt for generation
    query_response = client.post(
        "/api/v1/query",
        json={
            "configuration_name": configuration_name,
            "query": "Compare and contrast deep learning with traditional machine learning techniques.",
            "document_ids": [document_id],
            "generate": True,
            "generation_options": {
                "prompt_template": "You are an AI assistant specialized in explaining technical concepts. Based on the retrieved information and your knowledge, {query}\n\nRetrieved information:\n{context}\n\nResponse:"
            }
        }
    )
    
    # Check query succeeded
    assert query_response.status_code == 200, \
        f"Query failed with status {query_response.status_code}: {query_response.text}"
    
    query_data = query_response.json()
    
    # 4. Verify that reranking was applied
    assert "documents" in query_data, "Response missing documents array"
    assert len(query_data["documents"]) > 0, "No documents returned"
    first_doc = query_data["documents"][0]
    assert "original_similarity_score" in first_doc, \
        "Document doesn't have original_similarity_score, suggesting reranking wasn't applied"
    
    # 5. Verify generation was performed
    assert "answer" in query_data, "Response missing generated answer"
    assert len(query_data["answer"]) > 0, "Empty generated answer"
    
    # Check for a comprehensive answer covering both deep learning and machine learning
    answer = query_data["answer"].lower()
    assert "deep learning" in answer or "neural networks" in answer, \
        "Generated answer doesn't cover deep learning"
    assert "machine learning" in answer, "Generated answer doesn't cover machine learning"
    assert len(answer.split()) > 30, "Generated answer is too short for a comparison"
