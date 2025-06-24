#!/usr/bin/env python
"""
Example script demonstrating how to use Triton Inference Server with the RAG system.
This example shows how to configure and use both embedding and generation via Triton.
"""

import asyncio
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
from app.services.vector_store import FaissVectorStore
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
            model=EmbeddingModel.TRITON_EMBEDDING,
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
    
    # Initialize services
    print("Initializing embedding service with Triton...")
    embedding_service = EmbeddingService(config.embedding)
    
    print("Initializing vector store...")
    vector_store = FaissVectorStore(config.vector_store)
    
    print("Initializing document processor...")
    doc_processor = DocumentProcessor(config.chunking)
    
    print("Initializing generation service with Triton...")
    generation_service = GenerationServiceFactory.create_service(config.generation)
    
    # Initialize RAG service
    rag_service = RAGService(
        doc_processor=doc_processor,
        embedding_service=embedding_service,
        vector_store=vector_store,
        generation_service=generation_service,
        retrieval_k=config.retrieval_k,
        similarity_threshold=config.similarity_threshold
    )
    
    # Test document processing and embedding
    test_doc = {
        "content": "Triton Inference Server is an open-source inference serving software that streamlines AI inference by standardizing model deployment and execution across different frameworks. It helps manage scaling, batching, and provides a consistent API for serving AI models.",
        "metadata": {
            "filename": "triton_info.txt",
            "source": "test"
        }
    }
    
    # Process and index the document
    print("\nProcessing and indexing test document...")
    chunks = doc_processor.process_text(test_doc["content"])
    
    document_ids = []
    for i, chunk in enumerate(chunks):
        doc_id = f"triton-doc-{i}"
        document_ids.append(doc_id)
        await rag_service.index_text(
            doc_id,
            chunk,
            {"filename": test_doc["metadata"]["filename"], "chunk_id": i}
        )
    
    print(f"Indexed {len(chunks)} chunks")
    
    # Test querying
    test_query = "What is Triton Inference Server used for?"
    
    print(f"\nQuerying: '{test_query}'")
    response = await rag_service.query(test_query)
    
    print("\n=== Response ===")
    print(response)
    
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
