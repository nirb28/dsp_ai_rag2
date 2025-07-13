import os
import sys
import logging
import asyncio
from typing import List, Dict, Any
from pathlib import Path

# Add the project root to Python path to ensure imports work correctly
sys.path.append(str(Path(__file__).parent.parent))

from app.config import (
    RAGConfig, 
    ChunkingConfig, 
    VectorStoreConfig, 
    EmbeddingConfig,
    RerankerConfig, 
    RerankerModel, 
    VectorStore,
    EmbeddingModel
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreManager
from app.services.reranker_service import RerankerService
from app.services.document_processor import DocumentProcessor
from langchain.docstore.document import Document as LangchainDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test document content
SAMPLE_DOCUMENTS = [
    {
        "title": "Python Programming",
        "content": """
        Python is a high-level, interpreted programming language known for its readability and simplicity.
        It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
        Python was created by Guido van Rossum and first released in 1991.
        """
    },
    {
        "title": "Machine Learning",
        "content": """
        Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
        Common machine learning algorithms include linear regression, decision trees, and neural networks.
        Supervised learning requires labeled training data, while unsupervised learning works with unlabeled data.
        """
    },
    {
        "title": "Natural Language Processing",
        "content": """
        Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language.
        NLP tasks include text classification, named entity recognition, sentiment analysis, and machine translation.
        Modern NLP systems often use transformer models like BERT, GPT, and T5 which have revolutionized the field.
        """
    },
    {
        "title": "Vector Databases",
        "content": """
        Vector databases are specialized systems designed to store and query high-dimensional vectors efficiently.
        They are commonly used in AI applications for similarity search operations.
        Popular vector databases include FAISS, Milvus, Pinecone, and Weaviate.
        These systems often use approximate nearest neighbor algorithms like HNSW or IVF.
        """
    },
    {
        "title": "Reranking in RAG",
        "content": """
        Reranking is a critical step in Retrieval Augmented Generation (RAG) pipelines.
        It improves retrieval quality by applying more sophisticated relevance models to reorder initially retrieved documents.
        Cross-encoders, which process query and document pairs together, often outperform bi-encoders for reranking tasks.
        Popular reranking models include Cohere Rerank, BGE Reranker, and various cross-encoders from the SentenceTransformers library.
        """
    }
]

# Test queries
TEST_QUERIES = [
    "Who created Python?",
    "How do supervised and unsupervised learning differ?",
    "What are transformer models used for?",
    "Which algorithms are used in vector databases?",
    "Why is reranking important in RAG pipelines?"
]

async def setup_vector_stores(config1: RAGConfig, config2: RAGConfig):
    """Set up and populate vector stores for both configurations."""
    # Create vector store manager
    vs_manager = VectorStoreManager()
    
    # Get vector stores for both configurations
    vs1 = vs_manager.get_vector_store(
        configuration_name=config1.configuration_name,
        config=config1.vector_store,
        embedding_config=config1.embedding.dict()
    )
    
    vs2 = vs_manager.get_vector_store(
        configuration_name=config2.configuration_name,
        config=config2.vector_store,
        embedding_config=config2.embedding.dict()
    )
    
    # Convert to LangchainDocuments
    langchain_docs = []
    for idx, doc in enumerate(SAMPLE_DOCUMENTS):
        langchain_docs.append(
            LangchainDocument(
                page_content=doc["content"],
                metadata={"title": doc["title"], "id": f"doc_{idx}"}
            )
        )
    
    # Add documents to both vector stores
    vs1.add_documents(langchain_docs)
    vs2.add_documents(langchain_docs)
    
    logger.info(f"Added {len(langchain_docs)} documents to both vector stores")
    
    return vs1, vs2

async def run_query_comparison(
    query: str, 
    vs1, 
    vs2, 
    config1: RAGConfig, 
    config2: RAGConfig,
    reranker1: RerankerService,
    reranker2: RerankerService
):
    """Run the same query through both configurations and compare results."""
    
    # First configuration search
    results1 = vs1.similarity_search(
        query=query, 
        k=config1.reranking.top_n if config1.reranking.enabled else config1.retrieval_k,
        similarity_threshold=config1.similarity_threshold
    )
    
    # Second configuration search
    results2 = vs2.similarity_search(
        query=query, 
        k=config2.reranking.top_n if config2.reranking.enabled else config2.retrieval_k,
        similarity_threshold=config2.similarity_threshold
    )
    
    # Apply reranking for first configuration
    if config1.reranking.enabled:
        # Convert to the format expected by reranker
        docs1 = [
            {"content": doc.page_content, "metadata": doc.metadata, "similarity_score": score} 
            for doc, score in results1
        ]
        reranked1 = await reranker1.rerank(query, docs1)
        # Keep only the top k
        reranked1 = reranked1[:config1.retrieval_k]
    else:
        # Convert to the same format for consistency
        reranked1 = [
            {"content": doc.page_content, "metadata": doc.metadata, "similarity_score": score} 
            for doc, score in results1[:config1.retrieval_k]
        ]
    
    # Apply reranking for second configuration
    if config2.reranking.enabled:
        # Convert to the format expected by reranker
        docs2 = [
            {"content": doc.page_content, "metadata": doc.metadata, "similarity_score": score} 
            for doc, score in results2
        ]
        reranked2 = await reranker2.rerank(query, docs2)
        # Keep only the top k
        reranked2 = reranked2[:config2.retrieval_k]
    else:
        # Convert to the same format for consistency
        reranked2 = [
            {"content": doc.page_content, "metadata": doc.metadata, "similarity_score": score} 
            for doc, score in results2[:config2.retrieval_k]
        ]
    
    # Print results
    print(f"\n\n{'=' * 80}")
    print(f"QUERY: {query}")
    print(f"{'=' * 80}")
    
    print(f"\n{'-' * 40}")
    print(f"CONFIG 1: {config1.configuration_name} ({config1.reranking.model.value if config1.reranking.enabled else 'No reranking'})")
    print(f"{'-' * 40}")
    for i, doc in enumerate(reranked1):
        print(f"[{i+1}] {doc['metadata']['title']} (Score: {doc['similarity_score']:.4f})")
        print(f"    {doc['content'][:100]}...")
    
    print(f"\n{'-' * 40}")
    print(f"CONFIG 2: {config2.configuration_name} ({config2.reranking.model.value if config2.reranking.enabled else 'No reranking'})")
    print(f"{'-' * 40}")
    for i, doc in enumerate(reranked2):
        print(f"[{i+1}] {doc['metadata']['title']} (Score: {doc['similarity_score']:.4f})")
        print(f"    {doc['content'][:100]}...")
    
    return reranked1, reranked2

def create_config_1():
    """Create the first RAG configuration with BGE Reranker."""
    return RAGConfig(
        configuration_name="bge_reranker_config",
        chunking=ChunkingConfig(
            strategy="recursive_text",
            chunk_size=1000,
            chunk_overlap=200
        ),
        vector_store=VectorStoreConfig(
            type=VectorStore.FAISS,
            index_path="./storage/faiss_index/bge_reranker_test",
            dimension=384
        ),
        embedding=EmbeddingConfig(
            model="all-MiniLM-L6-v2",
            server_url= "http://localhost:9001",
            batch_size=32
        ),
        retrieval_k=3,
        similarity_threshold=0.6,
        reranking=RerankerConfig(
            enabled=True,
            model=RerankerModel.BGE_RERANKER,
            top_n=10,
            score_threshold=0.1
        )
    )

def create_config_2():
    """Create the second RAG configuration with Cross-Encoder Reranker."""
    return RAGConfig(
        configuration_name="cross_encoder_config",
        chunking=ChunkingConfig(
            strategy="recursive_text",
            chunk_size=1000,
            chunk_overlap=200
        ),
        vector_store=VectorStoreConfig(
            type=VectorStore.FAISS,
            index_path="./storage/faiss_index/cross_encoder_test",
            dimension=384
        ),
        embedding=EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM,
            batch_size=32
        ),
        retrieval_k=3,
        similarity_threshold=0.6,
        reranking=RerankerConfig(
            enabled=True,
            model=RerankerModel.SENTENCE_TRANSFORMERS_CROSS_ENCODER,
            top_n=10,
            score_threshold=0.2  # Different threshold for comparison
        )
    )

async def main():
    """Main execution function."""
    # Create configurations
    config1 = create_config_1()
    config2 = create_config_2()
    
    # Initialize rerankers
    reranker1 = RerankerService(config1.reranking)
    reranker2 = RerankerService(config2.reranking)
    
    # Set up vector stores and add sample documents
    vs1, vs2 = await setup_vector_stores(config1, config2)
    
    # Run queries through both configurations
    for query in TEST_QUERIES:
        await run_query_comparison(query, vs1, vs2, config1, config2, reranker1, reranker2)

if __name__ == "__main__":
    asyncio.run(main())
