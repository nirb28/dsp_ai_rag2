# Neo4j Integration for DSP AI RAG

This document explains how to use Neo4j as a graph database for document storage and retrieval in the DSP AI RAG2 system.

## Overview

The Neo4j integration allows you to:

1. Store documents in a Neo4j graph database
2. Extract entities and keywords from document content
3. Create relationships between documents, entities, and keywords
4. Perform similarity search using graph traversal and vector embeddings
5. Leverage the power of graph databases for knowledge graph applications

## Requirements

- Neo4j database server (version 4.4+ recommended)
- Neo4j Python driver (`neo4j`)
- Optional: Neo4j Graph Data Science plugin for advanced graph algorithms
- Optional: Neo4j vector search capabilities for embedding-based similarity

## Configuration

To use Neo4j as your vector store, update your configuration as follows:

```python
from app.config import VectorStore, VectorStoreConfig

# Neo4j configuration
vector_store_config = VectorStoreConfig(
    type=VectorStore.NEO4J,
    neo4j_uri="neo4j://localhost:7687",  # Neo4j URI
    neo4j_user="neo4j",                  # Neo4j username
    neo4j_password="your_password",      # Neo4j password
    neo4j_database="neo4j"               # Neo4j database name
)

# Embedding configuration (optional but recommended)
embedding_config = {
    "enabled": True,
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32
}
```

## Security Considerations

For production use, it's recommended to:

1. Use environment variables for sensitive information like passwords
2. Set up proper authentication and role-based access control in Neo4j
3. Enable TLS encryption for Neo4j connections
4. Use dedicated Neo4j users with appropriate permissions

Example with environment variables:

```python
import os
from app.config import VectorStore, VectorStoreConfig

vector_store_config = VectorStoreConfig(
    type=VectorStore.NEO4J,
    neo4j_uri=os.environ.get("NEO4J_URI", "neo4j://localhost:7687"),
    neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
    neo4j_password=os.environ.get("NEO4J_PASSWORD"),
    neo4j_database=os.environ.get("NEO4J_DATABASE", "neo4j")
)
```

## Usage

### Initializing the Neo4j Graph Store

```python
from app.services.vector_store import VectorStoreManager
from app.config import VectorStoreConfig

# Create vector store manager
vector_store_manager = VectorStoreManager()

# Get Neo4j graph store instance
neo4j_store = vector_store_manager.get_vector_store(
    configuration_name="my_neo4j_config",
    config=vector_store_config,
    embedding_config=embedding_config
)
```

### Adding Documents

```python
from langchain.docstore.document import Document as LangchainDocument

# Create documents
documents = [
    LangchainDocument(
        page_content="Document content goes here...",
        metadata={
            "title": "Document Title",
            "source": "document_source",
            "author": "Author Name",
            # Any other metadata
        }
    )
]

# Add documents to Neo4j
document_ids = neo4j_store.add_documents(documents)
```

### Performing Similarity Search

```python
# Basic search
results = neo4j_store.similarity_search(
    query="Your search query here",
    k=5,  # Number of results to return
    similarity_threshold=0.3  # Minimum similarity score (0-1)
)

# Search with metadata filters
results = neo4j_store.similarity_search(
    query="Your search query here",
    k=5,
    filter_metadata={"source": "specific_source"}  # Filter by metadata
)

# Process results
for doc, score in results:
    print(f"Score: {score}")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### Deleting Documents

```python
# Delete documents by ID
neo4j_store.delete_documents(document_ids=["doc_id_1", "doc_id_2"])
```

## Graph Structure

The Neo4jGraphStore creates the following graph structure:

- Nodes:
  - `:Document` - Document nodes with content and metadata
  - `:Entity` - Extracted entities from document content
  - `:Keyword` - Extracted keywords from document content

- Relationships:
  - `(:Document)-[:CONTAINS]->(:Entity)` - Document contains entity
  - `(:Document)-[:HAS_KEYWORD]->(:Keyword)` - Document has keyword
  - `(:Document)-[:RELATED_TO]->(:Document)` - Documents are related (shared entities/keywords)

## Advanced Features

### Getting Graph Statistics

```python
# Get statistics about the graph
stats = neo4j_store.get_graph_stats()
print(stats)
```

### Custom Cypher Queries

If you need to perform custom queries against the Neo4j database:

```python
# Get the Neo4j driver session from the store
session = neo4j_store._get_session()

# Execute custom Cypher query
with session:
    result = session.run("""
    MATCH (d:Document)-[:CONTAINS]->(e:Entity)
    WHERE e.name = $entity_name
    RETURN d.title AS title, d.content AS content
    LIMIT 10
    """, entity_name="specific_entity")
    
    for record in result:
        print(record["title"], record["content"])
```

## Example

A complete example script is available at:

```
examples/neo4j_graph_store_example.py
```

## Troubleshooting

### Connection Issues

- Make sure Neo4j server is running and accessible
- Check URI, username, and password
- Verify network connectivity and firewall settings

### Vector Search Not Working

- Ensure Neo4j has vector search capabilities enabled
- Check if embedding service is correctly configured
- Verify embeddings are being created and stored properly

### Performance Optimization

- Create appropriate Neo4j indexes for better performance
- Consider using Neo4j Graph Data Science algorithms for advanced similarity
- Tune the entity and keyword extraction parameters for better results

## Additional Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)
