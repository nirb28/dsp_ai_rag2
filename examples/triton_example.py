#!/usr/bin/env python
"""
Example script demonstrating how to use Triton Inference Server with the RAG system.
This example shows how to configure and use both embedding and generation via Triton.
"""

import os
import sys
import asyncio

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

async def main():
    print("=== Triton Inference Server RAG Example ===")
    
    # Configure the RAG system to use Triton for embeddings and generation
    config = RAGConfig(
        collection_name="triton-demo",
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
            collection_name="triton-demo",
            metadata={"source": "test"},
            process_immediately=True
        )
        
        # Test querying
        test_query = "What is Triton Inference Server used for?"
        
        print(f"\nQuerying: '{test_query}'")
        response = await rag_service.query(test_query, collection_name="triton-demo")
        
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
    asyncio.run(main())
