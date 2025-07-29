# NetworkX Graph Store POC

This document describes the NetworkX-based graph store implementation as an alternative to traditional vector stores in the RAG system.

## Overview

The NetworkX Graph Store represents documents as nodes in a graph and creates relationships between documents based on shared entities, keywords, and semantic connections. This approach enables more interpretable and relationship-aware document retrieval.

## Key Features

### Relationship-Based Storage
- Documents are stored as nodes in a NetworkX graph
- Relationships are created based on shared entities and keywords
- Graph structure enables discovery of related documents through traversal

### Intelligent Similarity Search
- Uses multiple strategies for document retrieval:
  1. **Direct Matching**: Entity and keyword overlap
  2. **Graph Centrality**: Documents connected to many others score higher
  3. **Graph Traversal**: Expands search through connected documents
  4. **Content Analysis**: Considers document length and quality

### Graph Analytics
- Provides graph statistics (nodes, edges, density, connectivity)
- Supports graph algorithms for advanced analysis
- No dependency on vector embeddings (optional)

## Installation

1. Install NetworkX dependency:
```bash
pip install networkx
```

2. The NetworkX graph store is automatically available as a vector store type.

## Configuration

### Basic Configuration

```json
{
  "vector_store": {
    "type": "networkx",
    "index_path": "./storage/networkx_graph",
    "dimension": 384
  },
  "embedding": {
    "enabled": false
  }
}
```

### With Optional Embeddings

```json
{
  "vector_store": {
    "type": "networkx",
    "index_path": "./storage/networkx_graph",
    "dimension": 384
  },
  "embedding": {
    "enabled": true,
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

## Usage Examples

### 1. Basic Setup

```python
from app.services.rag_service import RAGService
from app.config import RAGConfig, VectorStoreConfig

# Create configuration
config = RAGConfig(
    vector_store=VectorStoreConfig(
        type="networkx",
        index_path="./storage/my_graph"
    )
)

# Initialize service
rag_service = RAGService()
rag_service.set_configuration("my_config", config)
```

### 2. Upload Documents

```python
# Upload documents - relationships are automatically created
result = rag_service.upload_text_content(
    content="Python is a programming language used for machine learning...",
    filename="python_ml.txt",
    configuration_name="my_config",
    metadata={"topic": "programming", "difficulty": "beginner"}
)
```

### 3. Query Documents

```python
# Query using graph-based similarity
response = rag_service.query(
    query="What programming languages are used for AI?",
    configuration_name="my_config",
    k=5
)
```

### 4. Get Graph Statistics

```python
# Access the graph store directly for analytics
vector_store = rag_service._get_vector_store("my_config")
stats = vector_store.get_graph_stats()

print(f"Nodes: {stats['num_nodes']}")
print(f"Edges: {stats['num_edges']}")
print(f"Density: {stats['density']}")
print(f"Connected: {stats['is_connected']}")
```

## How It Works

### Document Processing

1. **Entity Extraction**: Identifies proper nouns and named entities
2. **Keyword Extraction**: Extracts significant terms (frequency-based)
3. **Node Creation**: Creates a graph node for each document
4. **Relationship Building**: Connects documents with shared entities/keywords

### Similarity Search Algorithm

1. **Query Analysis**: Extracts entities and keywords from the query
2. **Direct Matching**: Finds documents with matching entities/keywords
3. **Graph Scoring**: Applies multiple scoring factors:
   - Entity overlap score (40% weight)
   - Keyword overlap score (30% weight)
   - Graph centrality score (20% weight)
   - Content quality score (10% weight)
4. **Graph Expansion**: Uses connected documents to expand results
5. **Ranking**: Sorts by combined score and applies threshold filtering

### Relationship Types

- **Shared Entity**: Documents mentioning the same entities (people, places, organizations)
- **Shared Keywords**: Documents with common significant terms
- **Graph Proximity**: Documents connected through intermediate nodes

## Testing

### Run Basic Tests

```bash
python test_networkx_integration.py
```

### Run Full POC

```bash
python examples/networkx_graph_poc.py
```

## API Integration

The NetworkX graph store integrates seamlessly with existing RAG API endpoints:

### Create Configuration

```bash
POST /configurations
{
  "configuration_name": "graph_config",
  "config": {
    "vector_store": {
      "type": "networkx",
      "index_path": "./storage/graph_store"
    }
  }
}
```

### Upload Documents

```bash
POST /upload
{
  "configuration_name": "graph_config",
  "metadata": {"topic": "ai"}
}
```

### Query Documents

```bash
POST /query
{
  "query": "What is machine learning?",
  "configuration_name": "graph_config",
  "k": 5
}
```

## Advantages

### ✅ Pros
- **Interpretable**: Relationships are explicit and understandable
- **No Embeddings Required**: Works without vector embeddings
- **Relationship Discovery**: Finds connections between documents
- **Graph Analytics**: Supports advanced graph algorithms
- **Flexible Scoring**: Multiple similarity strategies
- **Local Storage**: No external dependencies

### ⚠️ Considerations
- **Scalability**: Graph operations may be slower for very large datasets
- **Entity Extraction**: Simple implementation (can be enhanced with NER)
- **Memory Usage**: Stores full graph structure in memory
- **Cold Start**: Requires multiple documents to build meaningful relationships

## Performance Characteristics

- **Best For**: 
  - Knowledge bases with clear entity relationships
  - Datasets where document connections are important
  - Scenarios requiring interpretable similarity
  - Small to medium-sized document collections (< 10K documents)

- **Consider Alternatives For**:
  - Very large document collections (> 100K documents)
  - Scenarios requiring sub-second query response
  - Documents with minimal entity overlap

## Future Enhancements

1. **Advanced NER**: Integrate spaCy or similar for better entity extraction
2. **Semantic Embeddings**: Combine graph structure with vector similarity
3. **Graph Algorithms**: Implement PageRank, community detection
4. **Persistent Storage**: Add database backend for large graphs
5. **Visualization**: Graph visualization tools for relationship exploration

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure NetworkX is installed (`pip install networkx`)
2. **Empty Results**: Check similarity threshold (try lower values like 0.1)
3. **Slow Performance**: Consider reducing graph size or using embeddings
4. **Memory Issues**: Implement graph pruning for large datasets

### Debug Information

Enable debug logging to see graph operations:

```python
import logging
logging.getLogger('app.services.networkx_graph_store').setLevel(logging.DEBUG)
```

## Conclusion

The NetworkX Graph Store provides a novel approach to document storage and retrieval in RAG systems. It's particularly valuable for knowledge bases where document relationships are important and interpretability is desired. While it may not match the raw performance of optimized vector stores for very large datasets, it offers unique capabilities for relationship-aware document retrieval.
