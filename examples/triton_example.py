#!/usr/bin/env python
"""
Example script demonstrating how to use Triton Inference Server with the RAG system.
This example shows how to configure and use both embedding and generation via Triton.

triton_example.py --embed-only --configuration triton-demo
"""

import sys
import os
import asyncio
import logging

# Configure logging to show INFO level logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the project root to the Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.config import (
    RAGConfig, 
    EmbeddingConfig, 
    GenerationConfig, 
    ChunkingConfig,
    VectorStoreConfig,
    EmbeddingModel,
    GenerationModel,
    ChunkingStrategy
)
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationServiceFactory
from app.services.rag_service import RAGService
from app.services.vector_store import FAISSVectorStore
from app.services.document_processor import DocumentProcessor
import argparse


async def test_with_existing_index(configuration_name="triton-demo"):
    """Test querying with an existing FAISS index"""
    print("\n=== Testing with existing FAISS index ===")
    
    # Initialize RAG service
    rag_service = RAGService()
    
    # Get existing configuration
    config = rag_service.get_configuration(configuration_name)
    if not config:
        print(f"Error: Configuration '{configuration_name}' not found")
        return
    
    print(f"Using existing configuration for collection: {configuration_name}")
    print(f"Index path: {config.vector_store.index_path}")
    print(f"Embedding model: {config.embedding.model.value}")
    print(f"Generation model: {config.generation.model.value}")
    
    # Test querying
    test_query = "What is Triton Inference Server used for?"
    
    print(f"\nQuerying: '{test_query}'")
    try:
        response = await rag_service.query(test_query, configuration_name=configuration_name)
        
        print("\n=== Response ===")
        print(f"Query: {response.query}")
        print(f"Answer: {response.answer}")
        print(f"Processing time: {response.processing_time:.2f} seconds")
        print("\nSources:")
        for i, source in enumerate(response.sources, 1):
            print(f"Source {i} - Score: {source['similarity_score']:.4f}")
            print(f"Content: {source['content'][:100]}...")
            print()
    except Exception as e:
        print(f"Error during query: {str(e)}")


async def perform_embedding_only(configuration_name="triton-demo", text="This is a test text for embedding only."):
    """Perform embedding only using the configured embedding service"""
    print("\n=== Performing embedding only ===")
    
    # Initialize RAG service to get configuration
    rag_service = RAGService()
    
    # Get existing configuration
    config = rag_service.get_configuration(configuration_name)
    if not config:
        print(f"Error: Configuration '{configuration_name}' not found")
        return
    
    # Initialize embedding service with the collection's configuration
    embedding_service = EmbeddingService(config.embedding)
    
    print(f"Using embedding model: {config.embedding.model.value}")
    
    try:
        # Generate embeddings
        embeddings = embedding_service.embed_texts([text])
        embedding_dim = len(embeddings[0])
        
        print(f"Successfully generated embeddings with dimension: {embedding_dim}")
        print(f"First 5 values: {embeddings[0][:5]}")
        return embeddings[0]
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return None


async def main():
    print("=== Triton Inference Server RAG Example ===")
    
    # Configure the RAG system to use Triton for embeddings and generation
    config = RAGConfig(
        configuration_name="triton-demo",
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_TEXT,
            chunk_size=1000,
            chunk_overlap=200
        ),
        vector_store=VectorStoreConfig(
            index_path="./storage/triton_faiss_index",
            # This dimension should match your Triton embedding model's output dimension
            dimension=1024  # Adjust based on your model's dimension
        ),
        embedding=EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM,
            batch_size=32
        ),
        generation=GenerationConfig(
            model=GenerationModel.TRITON_LLAMA_3_70B,
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9
        ),
        retrieval_k=5,
        similarity_threshold=0.7
    )
    
    # Initialize the RAG service
    print("Initializing RAG service...")
    rag_service = RAGService()
    
    # Set the configuration for the collection
    print("Setting Triton configuration for collection...")
    rag_service.set_configuration("triton-demo", config)
    
    # Create a test document file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        f.write("Triton Inference Server is an open-source inference serving software that streamlines AI inference by standardizing model deployment and execution across different frameworks. It helps manage scaling, batching, and provides a consistent API for serving AI models.")
        test_file_path = f.name

    try:
        # Process and index the document
        print("\nUploading and processing test document...")
        await rag_service.upload_document(
            file_path=test_file_path,
            filename="triton_info.txt",
            configuration_name="triton-demo",
            metadata={"source": "test"},
            process_immediately=True
        )
        
        # Test querying
        test_query = "What is Triton Inference Server used for?"
        
        print(f"\nQuerying: '{test_query}'")
        response = await rag_service.query(test_query, configuration_name="triton-demo")
        
        print("\n=== Response ===")
        print(f"Query: {response.query}")
        print(f"Answer: {response.answer}")
        print(f"Processing time: {response.processing_time:.2f} seconds")
        print("\nSources:")
        for i, source in enumerate(response.sources, 1):
            print(f"Source {i} - Score: {source['similarity_score']:.4f}")
            print(f"Content: {source['content'][:100]}...")
            print()
    finally:
        # Clean up the temporary file
        import os
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Inference Server RAG Example")
    parser.add_argument("--test-existing", action="store_true", help="Test with existing FAISS index")
    parser.add_argument("--embed-only", action="store_true", help="Perform embedding only")
    parser.add_argument("--configuration", type=str, default="triton-demo", help="Configuration name to use")
    parser.add_argument("--text", type=str, default="This is a test text for embedding only.", 
                        help="Text to embed when using --embed-only")
    
    args = parser.parse_args()
    
    if args.test_existing:
        asyncio.run(test_with_existing_index(args.collection))
    elif args.embed_only:
        asyncio.run(perform_embedding_only(args.collection, args.text))
    else:
        asyncio.run(main())
