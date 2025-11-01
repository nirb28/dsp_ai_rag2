#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Store Example

This example demonstrates how to use the Neo4j Knowledge Graph Store with LangGraph
for text-to-graph conversion and graph-based document retrieval.

Features demonstrated:
1. Creating a knowledge graph configuration with LLM
2. Adding documents and extracting knowledge graph structure
3. Querying the graph using Cypher queries
4. Finding entities and relationships
5. Semantic search using graph structure
6. Graph statistics and visualization

Requirements:
- Neo4j database running (default: neo4j://localhost:7687)
- LLM configuration for graph extraction
- langchain-experimental package for graph transformers
"""

import os
import sys
import asyncio
import json
from typing import List, Dict, Any
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain.docstore.document import Document as LangchainDocument

from app.services.rag_service import RAGService
from app.config import RAGConfig, VectorStoreConfig, VectorStore, EmbeddingConfig, GenerationConfig, LLMConfig, LLMProvider

# Sample documents for knowledge graph extraction
SAMPLE_DOCS = [
    {
        "content": """
        Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, 
        one of the two pillars of modern physics. He was born in Ulm, Germany in 1879 and later moved 
        to Princeton, New Jersey. Einstein worked at Princeton University and the Institute for Advanced Study. 
        He won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.
        """,
        "metadata": {
            "title": "Albert Einstein Biography",
            "category": "science",
            "source": "encyclopedia"
        }
    },
    {
        "content": """
        Princeton University is a private Ivy League research university located in Princeton, New Jersey. 
        Founded in 1746, it is one of the oldest universities in the United States. The university has 
        been home to many notable faculty members including Albert Einstein, John Nash, and Woodrow Wilson. 
        Princeton is known for its strong programs in physics, mathematics, economics, and public policy.
        """,
        "metadata": {
            "title": "Princeton University Overview",
            "category": "education",
            "source": "university_guide"
        }
    },
    {
        "content": """
        The theory of relativity consists of two interrelated theories by Albert Einstein: special relativity 
        and general relativity. Special relativity applies to all physical phenomena in the absence of gravity, 
        while general relativity explains the law of gravitation and its relation to other forces of nature. 
        This theory revolutionized our understanding of space, time, and gravity.
        """,
        "metadata": {
            "title": "Theory of Relativity Explained",
            "category": "science",
            "source": "physics_textbook"
        }
    },
    {
        "content": """
        The Nobel Prize in Physics is awarded annually by the Royal Swedish Academy of Sciences to scientists 
        who have made outstanding contributions to physics. Notable recipients include Albert Einstein (1921), 
        Marie Curie (1903), and Richard Feynman (1965). The prize recognizes discoveries that have conferred 
        the greatest benefit to humankind.
        """,
        "metadata": {
            "title": "Nobel Prize in Physics",
            "category": "awards",
            "source": "nobel_foundation"
        }
    }
]


def get_neo4j_credentials():
    """Get Neo4j connection credentials from environment or use defaults."""
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    print(f"Using Neo4j connection: {neo4j_uri}")
    print(f"Neo4j user: {neo4j_user}")
    
    return neo4j_uri, neo4j_user, neo4j_password


def create_knowledge_graph_configuration() -> Dict[str, Any]:
    """Create a knowledge graph configuration."""
    neo4j_uri, neo4j_user, neo4j_password = get_neo4j_credentials()
    
    # Vector store config for knowledge graph
    vector_store_config = {
        "type": "neo4j_knowledge_graph",
        "neo4j_uri": neo4j_uri,
        "neo4j_user": neo4j_user,
        "neo4j_password": neo4j_password,
        "neo4j_database": "neo4j",
        "kg_llm_config_name": "nvidia-llama3-70b"  # Reference to LLM config
    }
    
    # Embedding config (not used for knowledge graph but required for interface)
    embedding_config = {
        "enabled": False,  # Knowledge graph doesn't need embeddings
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    # Generation config for responses
    generation_config = {
        "model": "llama3-8b-8192",
        "provider": "groq",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": os.getenv("GROQ_API_KEY", "your-groq-api-key"),
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    return {
        "vector_store": vector_store_config,
        "embedding": embedding_config,
        "generation": generation_config
    }


def prepare_documents() -> List[LangchainDocument]:
    """Prepare sample documents for indexing."""
    documents = []
    for idx, doc in enumerate(SAMPLE_DOCS):
        # Create LangChain document with content and metadata
        document = LangchainDocument(
            page_content=doc["content"],
            metadata={
                **doc["metadata"],
                "id": f"doc_{idx}",
                "filename": f"{doc['metadata']['title'].lower().replace(' ', '_')}.txt"
            }
        )
        documents.append(document)
    
    return documents


async def demonstrate_knowledge_graph():
    """Demonstrate knowledge graph functionality."""
    print("🔗 Neo4j Knowledge Graph Store Example")
    print("=" * 50)
    
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        # Set LLM configuration
        llm_config = {
            "name": "nvidia-llama3-70b",
            "provider": "huggingface",
            "model": "nvidia/llama-70b",
            "system_prompt": "Knowledge graph extraction",
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9
        }
        llm_config_obj = LLMConfig(
            provider=LLMProvider(llm_config["provider"]),
            model=llm_config["model"],
            endpoint=llm_config.get("endpoint", None),
            api_key=llm_config.get("api_key", None),
            system_prompt=llm_config["system_prompt"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            top_p=llm_config["top_p"]
        )
        rag_service.set_llm_configuration(llm_config["name"], llm_config_obj)
        
        # Create knowledge graph configuration
        print("\n2. Creating knowledge graph configuration...")
        config_data = create_knowledge_graph_configuration()
        config_name = "knowledge_graph_demo"
        
        # Set configuration
        config = RAGConfig(**config_data)
        rag_service.set_configuration(config_name, config)
        print(f"✅ Created configuration: {config_name}")
        
        # Prepare and add documents
        print("\n3. Adding documents and extracting knowledge graph...")
        documents = prepare_documents()
        
        # Add documents to the knowledge graph
        doc_ids = []
        for i, doc in enumerate(documents):
            doc_id = rag_service.upload_text_content(
                content=doc.page_content,
                filename=doc.metadata.get('filename', f'doc_{i}.txt'),
                configuration_name=config_name,
                metadata=doc.metadata,
                process_immediately=True
            )
            doc_ids.append(doc_id['document_id'])
        
        print(f"✅ Added {len(doc_ids)} documents to knowledge graph")
        print(f"Document IDs: {doc_ids}")
        
        # Get the vector store to access knowledge graph specific methods
        vector_store = rag_service.vector_store_manager.get_vector_store(
            config_name, 
            rag_service.configurations[config_name].vector_store,
            rag_service.configurations[config_name].embedding.dict()
        )
        
        # Display graph statistics
        print("\n4. Knowledge Graph Statistics:")
        stats = vector_store.get_graph_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Find entities
        print("\n5. Finding entities in the graph...")
        entities = vector_store.find_entities(limit=10)
        print("Found entities:")
        for entity in entities:
            print(f"   - {entity['name']} ({entity['type']})")
        
        # Find relationships
        print("\n6. Finding relationships in the graph...")
        relationships = vector_store.find_relationships(limit=10)
        print("Found relationships:")
        for rel in relationships:
            print(f"   - {rel['source_name']} --[{rel['relationship_type']}]--> {rel['target_name']}")
        
        # Perform semantic search
        print("\n7. Performing semantic search...")
        search_queries = [
            "Who is Albert Einstein?",
            "What is the theory of relativity?",
            "Tell me about Princeton University",
            "Nobel Prize winners in physics"
        ]
        
        for query in search_queries:
            print(f"\nQuery: {query}")
            results = rag_service.query(config_name, query, k=3)
            print(f"Answer: {results['answer']}")
            print("Sources:")
            for source in results['sources']:
                print(f"   - {source.get('source', 'Unknown')} (score: {source.get('score', 0):.3f})")
        
        # Demonstrate graph queries
        print("\n8. Executing custom Cypher queries...")
        
        # Find all people and their connections
        cypher_query = """
        MATCH (p:Entity {type: 'Person'})-[r]-(connected:Entity)
        RETURN p.name as person, type(r) as relationship, connected.name as connected_entity, connected.type as entity_type
        LIMIT 10
        """
        
        try:
            graph_results = vector_store.query_graph(cypher_query)
            print("People and their connections:")
            for result in graph_results:
                print(f"   - {result['person']} --[{result['relationship']}]--> {result['connected_entity']} ({result['entity_type']})")
        except Exception as e:
            print(f"   Error executing Cypher query: {e}")
        
        # Find connected entities
        print("\n9. Finding entities connected to 'Albert Einstein'...")
        try:
            connected = vector_store.find_connected_entities("Albert Einstein", max_depth=2, limit=10)
            print("Connected entities:")
            for entity in connected:
                print(f"   - {entity['name']} ({entity['type']}) - distance: {entity['distance']}")
        except Exception as e:
            print(f"   Error finding connected entities: {e}")
        
        print("\n✅ Knowledge graph demonstration completed!")
        print("\nNext steps:")
        print("- Explore the Neo4j browser at http://localhost:7474")
        print("- Try custom Cypher queries")
        print("- Add more documents to expand the knowledge graph")
        print("- Experiment with different LLM configurations for better extraction")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the knowledge graph example."""
    print("Starting Neo4j Knowledge Graph Example...")
    print("Make sure Neo4j is running and accessible!")
    print("Default connection: neo4j://localhost:7687")
    print("Default credentials: neo4j/password")
    print()
    
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY not set. Please set it in your environment or .env file")
        print("   You can get a free API key from: https://console.groq.com/")
        print()
    
    # Run the demonstration
    asyncio.run(demonstrate_knowledge_graph())


if __name__ == "__main__":
    main()
