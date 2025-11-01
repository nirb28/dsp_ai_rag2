import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from neo4j import GraphDatabase
from langchain.docstore.document import Document as LangchainDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain.schema import Document
from langchain_core.language_models.base import BaseLanguageModel

from app.services.query_expansion_service import QueryExpansionService

logger = logging.getLogger(__name__)


class Neo4jKnowledgeGraphStore:
    """
    Neo4j-based knowledge graph store that uses LangGraph to convert text into nodes and edges.
    
    This implementation:
    - Uses LangGraph's LLMGraphTransformer to extract entities and relationships from text
    - Stores the extracted graph structure in Neo4j as a true knowledge graph
    - Supports graph-based querying using Cypher queries
    - Integrates with the existing LLM configuration system
    """
    
    def __init__(self, config: Dict[str, Any], llm_config_name: Optional[str] = None):
        """Initialize the Neo4j knowledge graph store."""
        self.config = config
        self.llm_config_name = llm_config_name
        
        # Neo4j connection parameters
        self.uri = config.get('neo4j_uri', 'neo4j://localhost:7687')
        self.user = config.get('neo4j_user', 'neo4j')
        self.password = config.get('neo4j_password', '')
        self.database = config.get('neo4j_database', 'neo4j')
        
        # Initialize Neo4j connection
        self._initialize_database()
        
        # Initialize LangChain Neo4j graph
        self.graph = Neo4jGraph(
            url=self.uri,
            username=self.user,
            password=self.password,
            database=self.database
        )
        
        # Initialize query expansion service for LLM access
        self.query_expansion_service = QueryExpansionService()
        
        # Graph transformer will be initialized when needed with LLM
        self.graph_transformer = None
        
    def _initialize_database(self):
        """Initialize the Neo4j database connection and schema."""
        try:
            # Connect to Neo4j
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 'Connection successful' AS message")
                for record in result:
                    logger.info(f"Neo4j KG Store: {record['message']}")
            
            # Initialize schema for knowledge graph
            with self.driver.session(database=self.database) as session:
                # Create constraints for entities
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS 
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
                
                # Create constraints for documents
                session.run("""
                    CREATE CONSTRAINT document_id IF NOT EXISTS 
                    FOR (d:Document) REQUIRE d.id IS UNIQUE
                """)
                
                # Create indexes for better performance
                session.run("""
                    CREATE INDEX entity_name IF NOT EXISTS 
                    FOR (e:Entity) ON (e.name)
                """)
                
                session.run("""
                    CREATE INDEX entity_type IF NOT EXISTS 
                    FOR (e:Entity) ON (e.type)
                """)
                
                session.run("""
                    CREATE INDEX document_content IF NOT EXISTS 
                    FOR (d:Document) ON (d.content)
                """)
                
            logger.info("Neo4j knowledge graph schema initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Neo4j knowledge graph database: {str(e)}")
            raise
    
    async def _get_llm_for_graph_extraction(self) -> BaseLanguageModel:
        """Get LLM instance for graph extraction using the configured LLM."""
        if not self.llm_config_name:
            raise ValueError("LLM configuration name is required for graph extraction")
        
        # Use the query expansion service to get LLM instance
        llm = await self.query_expansion_service._get_llm_instance(self.llm_config_name)
        return llm
    
    async def _initialize_graph_transformer(self):
        """Initialize the graph transformer with LLM."""
        if self.graph_transformer is None:
            llm = await self._get_llm_for_graph_extraction()
            self.graph_transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=["Person", "Organization", "Location", "Event", "Concept", "Technology", "Product"],
                allowed_relationships=["WORKS_FOR", "LOCATED_IN", "PART_OF", "RELATED_TO", "CREATED", "USES", "PARTICIPATES_IN", "OWNS", "MANAGES"],
                strict_mode=False
            )
            logger.info("Graph transformer initialized with LLM")
    
    async def add_documents(self, documents: List[LangchainDocument]) -> Dict[str, Any]:
        """
        Add documents to the knowledge graph store.
        Converts text to graph structure and stores in Neo4j.
        """
        try:
            await self._initialize_graph_transformer()
            
            results = {
                "processed_documents": 0,
                "extracted_nodes": 0,
                "extracted_relationships": 0,
                "errors": []
            }
            
            for doc in documents:
                try:
                    # Generate unique document ID if not present
                    doc_id = doc.metadata.get('id', str(uuid.uuid4()))
                    
                    # Convert document to LangChain Document format for graph transformer
                    graph_doc = Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata
                    )
                    
                    # Extract graph structure from document
                    logger.info(f"Extracting graph structure from document: {doc_id}")
                    graph_documents = await asyncio.to_thread(
                        self.graph_transformer.convert_to_graph_documents,
                        [graph_doc]
                    )
                    
                    if not graph_documents:
                        logger.warning(f"No graph structure extracted from document: {doc_id}")
                        continue
                    
                    # Store document node
                    with self.driver.session(database=self.database) as session:
                        session.run("""
                            MERGE (d:Document {id: $doc_id})
                            SET d.content = $content,
                                d.filename = $filename,
                                d.created_at = datetime(),
                                d.metadata = $metadata
                        """, 
                        doc_id=doc_id,
                        content=doc.page_content,
                        filename=doc.metadata.get('filename', 'unknown'),
                        metadata=json.dumps(doc.metadata)
                        )
                    
                    # Process extracted graph documents
                    for graph_doc in graph_documents:
                        # Add nodes to Neo4j
                        for node in graph_doc.nodes:
                            with self.driver.session(database=self.database) as session:
                                session.run("""
                                    MERGE (e:Entity {id: $node_id})
                                    SET e.name = $name,
                                        e.type = $type,
                                        e.properties = $properties,
                                        e.created_at = datetime()
                                """,
                                node_id=node.id,
                                name=node.id,  # Node ID is typically the entity name
                                type=node.type,
                                properties=json.dumps(node.properties) if hasattr(node, 'properties') else "{}"
                                )
                                
                                # Link entity to source document
                                session.run("""
                                    MATCH (d:Document {id: $doc_id})
                                    MATCH (e:Entity {id: $node_id})
                                    MERGE (d)-[:MENTIONS]->(e)
                                """,
                                doc_id=doc_id,
                                node_id=node.id
                                )
                            
                            results["extracted_nodes"] += 1
                        
                        # Add relationships to Neo4j
                        for rel in graph_doc.relationships:
                            with self.driver.session(database=self.database) as session:
                                # Create relationship between entities
                                session.run(f"""
                                    MATCH (source:Entity {{id: $source_id}})
                                    MATCH (target:Entity {{id: $target_id}})
                                    MERGE (source)-[r:{rel.type}]->(target)
                                    SET r.properties = $properties,
                                        r.created_at = datetime()
                                """,
                                source_id=rel.source.id,
                                target_id=rel.target.id,
                                properties=json.dumps(rel.properties) if hasattr(rel, 'properties') else "{}"
                                )
                            
                            results["extracted_relationships"] += 1
                    
                    results["processed_documents"] += 1
                    logger.info(f"Successfully processed document: {doc_id}")
                    
                except Exception as e:
                    error_msg = f"Error processing document {doc.metadata.get('id', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            logger.info(f"Knowledge graph ingestion completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error adding documents to knowledge graph: {str(e)}")
            raise
    
    def query_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the knowledge graph.
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, limit=limit)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing graph query: {str(e)}")
            raise
    
    def find_entities(self, entity_name: str = None, entity_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find entities in the knowledge graph by name or type.
        """
        try:
            conditions = []
            params = {"limit": limit}
            
            if entity_name:
                conditions.append("e.name CONTAINS $entity_name")
                params["entity_name"] = entity_name
            
            if entity_type:
                conditions.append("e.type = $entity_type")
                params["entity_type"] = entity_type
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
                MATCH (e:Entity)
                {where_clause}
                RETURN e.id as id, e.name as name, e.type as type, e.properties as properties
                LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Error finding entities: {str(e)}")
            raise
    
    def find_relationships(self, source_entity: str = None, target_entity: str = None, 
                          relationship_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find relationships in the knowledge graph.
        """
        try:
            conditions = []
            params = {"limit": limit}
            
            if source_entity:
                conditions.append("source.name CONTAINS $source_entity")
                params["source_entity"] = source_entity
            
            if target_entity:
                conditions.append("target.name CONTAINS $target_entity")
                params["target_entity"] = target_entity
            
            rel_pattern = f"[r:{relationship_type}]" if relationship_type else "[r]"
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
                MATCH (source:Entity)-{rel_pattern}->(target:Entity)
                {where_clause}
                RETURN source.name as source_name, type(r) as relationship_type, 
                       target.name as target_name, r.properties as properties
                LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Error finding relationships: {str(e)}")
            raise
    
    def find_connected_entities(self, entity_name: str, max_depth: int = 2, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find entities connected to a given entity within a specified depth.
        """
        try:
            query = f"""
                MATCH path = (start:Entity {{name: $entity_name}})-[*1..{max_depth}]-(connected:Entity)
                RETURN DISTINCT connected.name as name, connected.type as type, 
                       length(path) as distance
                ORDER BY distance, connected.name
                LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_name=entity_name, limit=limit)
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Error finding connected entities: {str(e)}")
            raise
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search by finding relevant entities and their connected documents.
        """
        try:
            # Search for entities that match the query
            entity_query = """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $query_text OR e.type CONTAINS $query_text
                WITH e, 
                     CASE 
                         WHEN e.name CONTAINS $query_text THEN 2.0
                         WHEN e.type CONTAINS $query_text THEN 1.0
                         ELSE 0.5
                     END as relevance_score
                MATCH (d:Document)-[:MENTIONS]->(e)
                RETURN d.id as document_id, d.content as content, d.filename as filename,
                       d.metadata as metadata, e.name as entity_name, e.type as entity_type,
                       relevance_score
                ORDER BY relevance_score DESC, d.created_at DESC
                LIMIT $k
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(entity_query, query_text=query, k=k)
                documents = []
                
                for record in result:
                    doc_data = {
                        "id": record["document_id"],
                        "content": record["content"],
                        "filename": record["filename"],
                        "metadata": json.loads(record["metadata"]) if record["metadata"] else {},
                        "score": record["relevance_score"],
                        "matched_entity": record["entity_name"],
                        "entity_type": record["entity_type"]
                    }
                    documents.append(doc_data)
                
                return documents
                
        except Exception as e:
            logger.error(f"Error performing semantic search: {str(e)}")
            raise
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        try:
            stats = {}
            
            with self.driver.session(database=self.database) as session:
                # Count nodes by type
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] AS label, count(n) AS count
                """)
                
                for record in result:
                    stats[f"{record['label'].lower()}_count"] = record["count"]
                
                # Count relationships by type
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) AS type, count(r) AS count
                """)
                
                for record in result:
                    stats[f"{record['type'].lower()}_relationships"] = record["count"]
                
                # Get entity types distribution
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.type AS entity_type, count(e) AS count
                    ORDER BY count DESC
                """)
                
                entity_types = {}
                for record in result:
                    entity_types[record["entity_type"]] = record["count"]
                stats["entity_types"] = entity_types
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete documents and their associated graph structure."""
        try:
            results = {
                "deleted_documents": 0,
                "deleted_entities": 0,
                "deleted_relationships": 0
            }
            
            with self.driver.session(database=self.database) as session:
                for doc_id in document_ids:
                    # Delete document and its relationships
                    result = session.run("""
                        MATCH (d:Document {id: $doc_id})
                        OPTIONAL MATCH (d)-[r]-()
                        DELETE r, d
                        RETURN count(d) as deleted_docs
                    """, doc_id=doc_id)
                    
                    results["deleted_documents"] += result.single()["deleted_docs"]
                
                # Clean up orphaned entities (entities not mentioned by any document)
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE NOT (e)<-[:MENTIONS]-()
                    DETACH DELETE e
                    RETURN count(e) as deleted_entities
                """)
                
                results["deleted_entities"] = result.single()["deleted_entities"]
            
            logger.info(f"Deleted documents and cleaned up graph: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def get_document_count(self) -> int:
        """Get the number of documents in the knowledge graph."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (d:Document) RETURN count(d) AS count")
                return result.single()["count"]
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    def close(self):
        """Close the Neo4j connection."""
        try:
            if hasattr(self, 'driver'):
                self.driver.close()
                logger.info("Neo4j knowledge graph connection closed")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")
