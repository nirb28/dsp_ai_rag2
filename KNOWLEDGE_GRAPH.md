# Neo4j Knowledge Graph Integration

This document describes the Neo4j Knowledge Graph integration in the DSP AI RAG2 project, which uses LangGraph to convert text into structured knowledge graphs.

## Overview

The knowledge graph implementation provides a true graph database approach to document storage and retrieval, moving beyond simple vector similarity to semantic understanding through entity and relationship extraction.

### Key Features

- **Text-to-Graph Conversion**: Uses LangGraph's `LLMGraphTransformer` to extract entities and relationships from text
- **True Knowledge Graph**: Stores entities and relationships as nodes and edges in Neo4j
- **Semantic Search**: Retrieves documents based on entity and relationship matching
- **Graph Querying**: Supports custom Cypher queries for complex graph operations
- **LLM Integration**: Uses existing LLM configurations for graph extraction
- **Backward Compatibility**: Implements the same interface as other vector stores

## Architecture

```
Text Document → LangGraph Transformer → Entities & Relationships → Neo4j Graph → Semantic Search
```

### Components

1. **Neo4jKnowledgeGraphStore**: Core service for graph operations
2. **Neo4jKnowledgeGraphAdapter**: Adapter to maintain BaseVectorStore compatibility
3. **LLMGraphTransformer**: LangChain component for text-to-graph conversion
4. **Neo4j Database**: Graph storage backend

## Configuration

### Vector Store Configuration

```json
{
  "vector_store": {
    "type": "neo4j_knowledge_graph",
    "neo4j_uri": "neo4j://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "password",
    "neo4j_database": "neo4j",
    "kg_llm_config_name": "graph-extraction-llm"
  }
}
```

### Required LLM Configuration

The knowledge graph requires an LLM configuration for entity and relationship extraction:

```json
{
  "name": "graph-extraction-llm",
  "provider": "groq",
  "model": "llama3-8b-8192",
  "endpoint": "https://api.groq.com/openai/v1/chat/completions",
  "api_key": "your-api-key",
  "system_prompt": "Extract entities and relationships from text...",
  "temperature": 0.1,
  "max_tokens": 1024
}
```

## Usage

### Basic Setup

```python
from app.services.rag_service import RAGService

# Initialize RAG service
rag_service = RAGService()

# Create LLM configuration for graph extraction
await rag_service.add_llm_config(
    name="graph-extraction-llm",
    provider="groq",
    model="llama3-8b-8192",
    endpoint="https://api.groq.com/openai/v1/chat/completions",
    api_key="your-groq-api-key",
    system_prompt="Extract entities and relationships from text.",
    temperature=0.1,
    max_tokens=1024
)

# Create knowledge graph configuration
config_data = {
    "vector_store": {
        "type": "neo4j_knowledge_graph",
        "neo4j_uri": "neo4j://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "neo4j_database": "neo4j",
        "kg_llm_config_name": "graph-extraction-llm"
    },
    "embedding": {"enabled": False},
    "generation": {
        "model": "llama3-8b-8192",
        "provider": "groq",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": "your-groq-api-key"
    }
}

await rag_service.add_configuration("knowledge_graph", config_data)
```

### Adding Documents

```python
from langchain.docstore.document import Document

documents = [
    Document(
        page_content="Albert Einstein was a physicist who worked at Princeton University.",
        metadata={"title": "Einstein Biography", "category": "science"}
    )
]

doc_ids = await rag_service.add_documents("knowledge_graph", documents)
```

### Querying the Graph

```python
# Standard RAG query (uses semantic search)
result = await rag_service.query("knowledge_graph", "Who was Albert Einstein?")

# Access the vector store for advanced operations
vector_store = rag_service.vector_store_manager.get_vector_store(
    "knowledge_graph", 
    config.vector_store, 
    config.embedding.dict()
)

# Find entities
entities = vector_store.find_entities(entity_name="Einstein", limit=10)

# Find relationships
relationships = vector_store.find_relationships(source_entity="Einstein", limit=10)

# Execute custom Cypher queries
results = vector_store.query_graph("""
    MATCH (p:Entity {type: 'Person'})-[r]-(org:Entity {type: 'Organization'})
    RETURN p.name, type(r), org.name
    LIMIT 10
""")

# Find connected entities
connected = vector_store.find_connected_entities("Einstein", max_depth=2, limit=20)
```

## Entity and Relationship Types

The graph transformer extracts the following types by default:

### Entity Types
- **Person**: People mentioned in the text
- **Organization**: Companies, institutions, groups
- **Location**: Places, cities, countries
- **Event**: Significant events or occurrences
- **Concept**: Abstract concepts or ideas
- **Technology**: Technologies, tools, systems
- **Product**: Products, services, offerings

### Relationship Types
- **WORKS_FOR**: Employment relationships
- **LOCATED_IN**: Location relationships
- **PART_OF**: Hierarchical relationships
- **RELATED_TO**: General associations
- **CREATED**: Creation relationships
- **USES**: Usage relationships
- **PARTICIPATES_IN**: Participation relationships
- **OWNS**: Ownership relationships
- **MANAGES**: Management relationships

## Graph Schema

The Neo4j database uses the following schema:

### Node Types

```cypher
// Document nodes
(:Document {
  id: string,
  content: string,
  filename: string,
  created_at: datetime,
  metadata: string (JSON)
})

// Entity nodes
(:Entity {
  id: string,
  name: string,
  type: string,
  properties: string (JSON),
  created_at: datetime
})
```

### Relationship Types

```cypher
// Document mentions entity
(:Document)-[:MENTIONS]->(:Entity)

// Entity relationships (dynamic based on extraction)
(:Entity)-[:WORKS_FOR]->(:Entity)
(:Entity)-[:LOCATED_IN]->(:Entity)
(:Entity)-[:RELATED_TO]->(:Entity)
// ... other relationship types
```

## Performance Considerations

### Indexing

The system automatically creates the following indexes:

```cypher
CREATE CONSTRAINT entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT document_id FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE INDEX entity_name FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type FOR (e:Entity) ON (e.type);
CREATE INDEX document_content FOR (d:Document) ON (d.content);
```

### Optimization Tips

1. **LLM Selection**: Use faster models for graph extraction (e.g., Llama 3 8B vs 70B)
2. **Batch Processing**: Process documents in batches for better performance
3. **Graph Pruning**: Regularly clean up orphaned entities
4. **Query Optimization**: Use specific entity types and relationship filters in Cypher queries

## Examples

### Running the Example

```bash
# Set environment variables
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export GROQ_API_KEY="your-groq-api-key"

# Run the knowledge graph example
python examples/neo4j_knowledge_graph_example.py

# Run the test script
python test_knowledge_graph.py
```

### Sample Queries

```python
# Find all people and their organizations
results = vector_store.query_graph("""
    MATCH (p:Entity {type: 'Person'})-[:WORKS_FOR]->(org:Entity {type: 'Organization'})
    RETURN p.name as person, org.name as organization
""")

# Find documents mentioning specific entities
results = vector_store.query_graph("""
    MATCH (d:Document)-[:MENTIONS]->(e:Entity {name: 'Albert Einstein'})
    RETURN d.filename, d.content
""")

# Find shortest path between entities
results = vector_store.query_graph("""
    MATCH path = shortestPath((a:Entity {name: 'Einstein'})-[*]-(b:Entity {name: 'Princeton'}))
    RETURN path
""")
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Ensure Neo4j is running on the specified URI
   - Check credentials and database name
   - Verify network connectivity

2. **LLM Configuration Missing**
   - Ensure `kg_llm_config_name` is specified in vector store config
   - Verify the LLM configuration exists and is valid
   - Check API key permissions

3. **Graph Extraction Errors**
   - Review LLM system prompt for entity extraction
   - Check document content for extractable entities
   - Verify LLM model supports the required functionality

4. **Performance Issues**
   - Use faster LLM models for extraction
   - Implement document batching
   - Optimize Cypher queries with proper indexes

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Migration from Vector Store

To migrate from the old Neo4j vector store to the knowledge graph:

1. **Update Configuration**: Change `type` from `"neo4j"` to `"neo4j_knowledge_graph"`
2. **Add LLM Config**: Specify `kg_llm_config_name` in the configuration
3. **Create LLM Configuration**: Add an LLM configuration for graph extraction
4. **Re-index Documents**: Documents need to be re-processed for graph extraction
5. **Update Queries**: Leverage new graph-specific query methods

## Dependencies

The knowledge graph implementation requires:

```
langchain-experimental  # For LLMGraphTransformer
neo4j                  # Neo4j Python driver
langchain              # Core LangChain functionality
```

These are automatically included in `requirements.txt`.

## Future Enhancements

Planned improvements include:

- **Custom Entity Types**: Support for domain-specific entity types
- **Relationship Scoring**: Confidence scores for extracted relationships
- **Graph Visualization**: Built-in graph visualization tools
- **Advanced Algorithms**: Integration with Neo4j Graph Data Science
- **Multi-hop Reasoning**: Complex reasoning across graph relationships
- **Graph Embeddings**: Hybrid approach combining graph structure with embeddings
