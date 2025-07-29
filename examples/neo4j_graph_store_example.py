#!/usr/bin/env python
"""
Neo4j Graph Store Example

This script demonstrates how to use Neo4j as a graph-based vector store in the RAG system.
It shows how to:
1. Set up a Neo4j configuration
2. Add documents to the Neo4j graph store
3. Perform similarity searches using graph-based algorithms
4. Visualize the graph structure (if visualization libraries are available)

Requirements:
- Neo4j database running (default: neo4j://localhost:7687)
- Neo4j Python driver installed (pip install neo4j)
- Optional: Install Neo4j Graph Data Science plugin for advanced graph algorithms

Note: This is a simplified example to demonstrate the basic functionality.
"""

import os
import sys
import logging
from pathlib import Path
import json
import time
from typing import List, Dict, Any

# Add parent directory to path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import VectorStore, VectorStoreConfig
from app.services.vector_store import VectorStoreManager
from langchain.docstore.document import Document as LangchainDocument

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample documents for testing
SAMPLE_DOCS = [
    {
        "title": "Introduction to Neo4j",
        "content": """
        Neo4j is a graph database management system developed by Neo4j, Inc.
        It is a native graph database that uses nodes and relationships to represent and store data.
        Neo4j is designed for storing, mapping, and querying relationships.
        It has its own query language called Cypher, which is specifically designed for working with graph data.
        """
    },
    {
        "title": "Graph Database Concepts",
        "content": """
        A graph database is a database that uses graph structures for semantic queries with
        nodes, edges, and properties to represent and store data. The key concept of the system
        is the graph, which directly relates data items in the store. Graph databases are often
        used when modeling domains where relationships between entities are important.
        """
    },
    {
        "title": "Vector Embeddings in Graph Databases",
        "content": """
        Vector embeddings can be integrated with graph databases to enhance semantic search capabilities.
        By storing vector embeddings as properties on nodes, graph databases can combine traditional graph
        traversal with vector similarity search. This hybrid approach enables powerful knowledge retrieval
        that leverages both the structural information in the graph and the semantic information in embeddings.
        """
    },
    {
        "title": "Knowledge Graphs and RAG Systems",
        "content": """
        Knowledge graphs are a powerful foundation for Retrieval Augmented Generation (RAG) systems.
        They store information as entities and relationships, which allows for more nuanced retrieval
        based on the connections between different pieces of information. When combined with LLMs,
        knowledge graphs can provide context-aware, relationally rich information that improves
        the quality of generated responses.
        """
    },
]

def create_neo4j_configuration() -> Dict[str, Any]:
    """Create a configuration for Neo4j vector store."""
    # Get Neo4j connection details from environment variables or use defaults
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")  # Replace with your password
    
    # Create vector store config
    vector_store_config = {
        "type": "neo4j",
        "neo4j_uri": neo4j_uri,
        "neo4j_user": neo4j_user,
        "neo4j_password": neo4j_password,
        "neo4j_database": "neo4j"  # Default database name
    }
    
    # Create embedding config (optional but recommended for better search)
    embedding_config = {
        "enabled": True,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32
    }
    
    return {
        "vector_store": vector_store_config,
        "embedding": embedding_config
    }

def prepare_documents() -> List[LangchainDocument]:
    """Prepare sample documents for indexing."""
    documents = []
    for idx, doc in enumerate(SAMPLE_DOCS):
        # Create LangChain document with content and metadata
        document = LangchainDocument(
            page_content=doc["content"],
            metadata={
                "title": doc["title"],
                "id": f"doc_{idx}",
                "source": "sample_data"
            }
        )
        documents.append(document)
    return documents

def main():
    """Main function to demonstrate Neo4j graph store usage."""
    try:
        # Create configuration
        logger.info("Creating Neo4j configuration...")
        config = create_neo4j_configuration()
        
        # Initialize vector store manager
        vector_store_manager = VectorStoreManager()
        
        # Get Neo4j graph store
        config_name = "neo4j_demo"
        logger.info(f"Getting Neo4j graph store with configuration '{config_name}'...")
        vector_store_config = VectorStoreConfig(**config["vector_store"])
        neo4j_store = vector_store_manager.get_vector_store(
            configuration_name=config_name,
            config=vector_store_config,
            embedding_config=config["embedding"]
        )
        
        # Prepare documents
        documents = prepare_documents()
        
        # Add documents to the graph store
        logger.info(f"Adding {len(documents)} documents to Neo4j graph store...")
        doc_ids = neo4j_store.add_documents(documents)
        logger.info(f"Added documents with IDs: {doc_ids}")
        
        # Get graph statistics
        logger.info("Getting graph statistics...")
        if hasattr(neo4j_store, "get_graph_stats"):
            stats = neo4j_store.get_graph_stats()
            logger.info(f"Graph statistics: {json.dumps(stats, indent=2)}")
        
        # Perform similarity search
        queries = [
            "What are the key concepts of graph databases?",
            "How do vector embeddings work with graph databases?",
            "Tell me about knowledge graphs in RAG systems."
        ]
        
        for query in queries:
            logger.info(f"\nExecuting search query: '{query}'")
            start_time = time.time()
            results = neo4j_store.similarity_search(
                query=query,
                k=2,  # Return top 2 results
                similarity_threshold=0.3
            )
            end_time = time.time()
            
            logger.info(f"Search completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Found {len(results)} results\n")
            
            for i, (doc, score) in enumerate(results):
                logger.info(f"Result {i+1} [Score: {score:.4f}]")
                logger.info(f"Title: {doc.metadata.get('title', 'Unknown')}")
                logger.info(f"Content: {doc.page_content[:150]}...\n")
        
        logger.info("Neo4j graph store example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
