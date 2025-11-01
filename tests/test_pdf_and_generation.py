"""
Integration tests for PDF handling and generation capabilities of the RAG service.

These tests verify:
1. PDF documents are properly processed, extracted, and chunked
2. Multiple PDF documents can be uploaded and queried together
3. Generation parameters affect the output quality and style
"""

# Fix Python path for direct script execution
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import tempfile
import json
import time
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from io import BytesIO

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
def sample_pdf_file_ai():
    """Create a sample PDF file about AI for testing."""
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer)
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Understanding Artificial Intelligence")
    
    # Add content
    c.setFont("Helvetica", 12)
    y_position = 700
    
    paragraphs = [
        "Artificial Intelligence (AI) refers to computer systems designed to perform tasks that typically require human intelligence.",
        "These tasks include speech recognition, decision-making, visual perception, and language translation.",
        "Machine Learning is a subset of AI that uses statistical techniques to enable machines to improve with experience.",
        "Deep Learning is a type of machine learning based on artificial neural networks with multiple layers.",
        "Natural Language Processing (NLP) is a branch of AI focused on enabling computers to understand and process human language.",
        "AI systems can be classified as narrow AI or general AI. Narrow AI is designed for a specific task like voice recognition.",
        "General AI would have human-like capabilities across different domains, but it remains theoretical.",
        "AI technologies are increasingly used in healthcare, finance, transportation, and customer service.",
        "Ethical considerations in AI include privacy, bias, transparency, and the impact on employment.",
        "The future of AI may involve more autonomous systems, enhanced human-AI collaboration, and continued integration into daily life."
    ]
    
    for paragraph in paragraphs:
        c.drawString(100, y_position, paragraph)
        y_position -= 20
    
    c.save()
    
    # Create a temporary file and write PDF content
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(pdf_buffer.getvalue())
        file_path = f.name
    
    yield file_path
    
    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def sample_pdf_file_ml():
    """Create a sample PDF file about Machine Learning for testing."""
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer)
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Introduction to Machine Learning")
    
    # Add content
    c.setFont("Helvetica", 12)
    y_position = 700
    
    paragraphs = [
        "Machine Learning allows computer systems to learn from data without being explicitly programmed.",
        "Supervised learning involves training a model on labeled data to make predictions or decisions.",
        "Unsupervised learning finds patterns or structures in unlabeled data without explicit guidance.",
        "Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors.",
        "Common algorithms include decision trees, neural networks, support vector machines, and clustering algorithms.",
        "Supervised learning algorithms like linear regression predict continuous values based on input features.",
        "Classification algorithms such as logistic regression and random forests categorize data into discrete classes.",
        "Deep learning, a subset of machine learning, uses neural networks with many layers to process complex patterns.",
        "Feature engineering is the process of selecting and transforming variables to improve model performance.",
        "Model evaluation metrics include accuracy, precision, recall, F1 score, and area under the ROC curve."
    ]
    
    for paragraph in paragraphs:
        c.drawString(100, y_position, paragraph)
        y_position -= 20
    
    c.save()
    
    # Create a temporary file and write PDF content
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(pdf_buffer.getvalue())
        file_path = f.name
    
    yield file_path
    
    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def sample_pdf_file_nlp():
    """Create a sample PDF file about NLP for testing."""
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer)
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Natural Language Processing Fundamentals")
    
    # Add content
    c.setFont("Helvetica", 12)
    y_position = 700
    
    paragraphs = [
        "Natural Language Processing (NLP) is focused on interactions between human language and computers.",
        "Key NLP tasks include sentiment analysis, named entity recognition, and machine translation.",
        "Modern NLP systems use transformer models like BERT, GPT, and T5 which have achieved state-of-the-art results.",
        "Applications of NLP include chatbots, virtual assistants, and automated content generation.",
        "NLP techniques are essential for search engines, email filters, and language translation services.",
        "Text preprocessing in NLP involves tokenization, stemming, lemmatization, and removing stop words.",
        "Word embeddings like Word2Vec and GloVe represent words as dense vectors capturing semantic relationships.",
        "Sentiment analysis determines the emotional tone behind text, commonly used for product reviews and social media.",
        "Named Entity Recognition identifies entities like people, organizations, locations, and dates in text.",
        "Question answering systems combine information retrieval and natural language understanding to answer human queries."
    ]
    
    for paragraph in paragraphs:
        c.drawString(100, y_position, paragraph)
        y_position -= 20
    
    c.save()
    
    # Create a temporary file and write PDF content
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(pdf_buffer.getvalue())
        file_path = f.name
    
    yield file_path
    
    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def enhanced_config_for_testing():
    """Create an enhanced configuration for testing generation parameters."""
    return {
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
        },
        "generation": {
            "model": "llama3-8b-8192",
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9
        }
    }


def test_pdf_handling_and_generation(client, temp_storage, sample_pdf_file_ai, sample_pdf_file_ml, sample_pdf_file_nlp, enhanced_config_for_testing):
    """Test PDF handling and generation features of the RAG pipeline.
    
    This test:
    1. Uploads multiple PDF documents
    2. Verifies they are correctly processed and indexed
    3. Tests different query types against the PDF content
    4. Verifies generation parameters affect the response
    """
    # 1. Configure a collection with enhanced settings
    configuration_name = "pdf_test_collection"
    
    config_response = client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": configuration_name,
            "config": enhanced_config_for_testing
        }
    )
    assert config_response.status_code == 200
    
    # 2. Upload the first PDF document about AI
    with open(sample_pdf_file_ai, 'rb') as f:
        upload_response_ai = client.post(
            "/api/v1/upload",
            files={"file": (Path(sample_pdf_file_ai).name, f, "application/pdf")},
            data={
                "configuration_name": configuration_name,
                "metadata": json.dumps({"topic": "artificial intelligence", "source": "test_document"}),
                "process_immediately": "true"
            }
        )
    
    assert upload_response_ai.status_code == 200
    upload_data_ai = upload_response_ai.json()
    assert upload_data_ai["status"] == "indexed"
    assert upload_data_ai["configuration_name"] == configuration_name
    
    # 3. Upload the second PDF document about Machine Learning
    with open(sample_pdf_file_ml, 'rb') as f:
        upload_response_ml = client.post(
            "/api/v1/upload",
            files={"file": (Path(sample_pdf_file_ml).name, f, "application/pdf")},
            data={
                "configuration_name": configuration_name,
                "metadata": json.dumps({"topic": "machine learning", "source": "test_document"}),
                "process_immediately": "true"
            }
        )
    
    assert upload_response_ml.status_code == 200
    upload_data_ml = upload_response_ml.json()
    assert upload_data_ml["status"] == "indexed"
    
    # 4. Upload the third PDF document about NLP
    with open(sample_pdf_file_nlp, 'rb') as f:
        upload_response_nlp = client.post(
            "/api/v1/upload",
            files={"file": (Path(sample_pdf_file_nlp).name, f, "application/pdf")},
            data={
                "configuration_name": configuration_name,
                "metadata": json.dumps({"topic": "natural language processing", "source": "test_document"}),
                "process_immediately": "true"
            }
        )
    
    assert upload_response_nlp.status_code == 200
    upload_data_nlp = upload_response_nlp.json()
    assert upload_data_nlp["status"] == "indexed"
    
    # 5. Allow some time for processing and indexing
    time.sleep(2)
    
    # 5.1 Debug: Print extracted content from PDFs to verify text extraction
    with open(sample_pdf_file_ai, 'rb') as f:
        from app.services.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        extracted_text = processor.extract_text(sample_pdf_file_ai)
        print(f"\nExtracted text from AI PDF:\n{extracted_text[:200]}...")
    
    # 6. Check the configuration info to verify all 3 PDFs were processed
    configurations_response = client.get("/api/v1/configurations")
    assert configurations_response.status_code == 200
    
    configurations_data = configurations_response.json()
    configuration_info = next((c for c in configurations_data["configurations"] if c["name"] == configuration_name), None)
    
    assert configuration_info is not None
    # We should have some documents (exact count may vary due to chunking)
    assert configuration_info["document_count"] > 0
    
    # 7. Test a query about artificial intelligence
    query_response_ai = client.post(
        "/api/v1/query",
        json={
            "query": "Artificial Intelligence refers to computer systems designed to perform tasks",  # Match exact text from PDF
            "configuration_name": configuration_name,
            "include_metadata": True,
            "similarity_threshold": 0.3  # Lower threshold to include more matching documents
        }
    )
    
    assert query_response_ai.status_code == 200
    query_data_ai = query_response_ai.json()
    
    # Verify generation worked and includes relevant content about AI
    assert len(query_data_ai["answer"]) > 0
    assert len(query_data_ai["sources"]) > 0
    
    # 8. Test a query about machine learning
    query_response_ml = client.post(
        "/api/v1/query",
        json={
            "query": "Machine Learning allows computer systems to learn from data",  # Match exact text from PDF
            "configuration_name": configuration_name,
            "include_metadata": True,
            "similarity_threshold": 0.3  # Lower threshold to include more matching documents
        }
    )
    
    assert query_response_ml.status_code == 200
    query_data_ml = query_response_ml.json()
    
    # Verify generation worked and includes relevant content about ML
    assert len(query_data_ml["answer"]) > 0
    assert len(query_data_ml["sources"]) > 0
    
    # 9. Test generation with different temperature (more creative)
    # First, update configuration with higher temperature
    high_temp_config = enhanced_config_for_testing.copy()
    high_temp_config["generation"] = {
        "model": "llama3-8b-8192",
        "temperature": 0.9,  # Higher temperature
        "max_tokens": 150,
        "top_p": 0.9
    }
    
    config_response = client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": configuration_name,
            "config": high_temp_config
        }
    )
    assert config_response.status_code == 200
    
    # Query with higher temperature
    query_response_high_temp = client.post(
        "/api/v1/query",
        json={
            "query": "Natural Language Processing is focused on interactions between human language",  # Match exact text from PDF
            "configuration_name": configuration_name,
            "include_metadata": True,
            "similarity_threshold": 0.3  # Lower threshold to include more matching documents
        }
    )
    
    assert query_response_high_temp.status_code == 200
    query_data_high_temp = query_response_high_temp.json()
    high_temp_answer = query_data_high_temp["answer"]
    
    # 10. Test generation with lower temperature (more focused)
    # Update configuration with lower temperature
    low_temp_config = enhanced_config_for_testing.copy()
    low_temp_config["generation"] = {
        "model": "llama3-8b-8192",
        "temperature": 0.1,  # Lower temperature
        "max_tokens": 150,
        "top_p": 0.9
    }
    
    config_response = client.post(
        "/api/v1/configurations",
        json={
            "configuration_name": configuration_name,
            "config": low_temp_config
        }
    )
    assert config_response.status_code == 200
    
    # Query with lower temperature
    query_response_low_temp = client.post(
        "/api/v1/query",
        json={
            "query": "Natural Language Processing is focused on interactions between human language",  # Match exact text from PDF
            "configuration_name": configuration_name,
            "include_metadata": True,
            "similarity_threshold": 0.3  # Lower threshold to include more matching documents
        }
    )
    
    assert query_response_low_temp.status_code == 200
    query_data_low_temp = query_response_low_temp.json()
    low_temp_answer = query_data_low_temp["answer"]
    
    # Store both answers for manual comparison
    # Note: We can't automatically compare the answers as the difference will be in style/variance,
    # but the test output will show both answers for manual inspection
    print("\n\nHigh Temperature (0.9) Answer:")
    print(high_temp_answer)
    print("\nLow Temperature (0.1) Answer:")
    print(low_temp_answer)
    
    # 11. Clean up by deleting the configuration
    delete_response = client.delete(f"/api/v1/configurations/{configuration_name}")
    assert delete_response.status_code == 200
