import os
import json
import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from pathlib import Path
import uuid
from collections import defaultdict

from neo4j import GraphDatabase
from langchain.docstore.document import Document as LangchainDocument
import numpy as np

from app.services.base_vector_store import BaseVectorStore
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class Neo4jGraphStore(BaseVectorStore):
    """
    Neo4j-based graph store that represents documents as nodes and relationships as edges.
    
    This implementation creates a knowledge graph where:
    - Documents are represented as nodes with their content and metadata
    - Entities (people, organizations, locations, etc.) are extracted and represented as nodes
    - Keywords are extracted and represented as nodes
    - Relationships between documents and entities/keywords are created
    - Document-to-document relationships are created based on shared entities/keywords
    - Similarity search is performed using graph algorithms
    """
    
    def __init__(self, config: Dict, embedding_service: Optional[EmbeddingService] = None):
        """Initialize the Neo4j graph store with configuration."""
        super().__init__(config)
        self.embedding_service = embedding_service
        
        # Neo4j connection parameters
        self.uri = config.get('neo4j_uri', 'neo4j://localhost:7687')
        self.user = config.get('neo4j_user', 'neo4j')
        self.password = config.get('neo4j_password', '')
        self.database = config.get('neo4j_database', 'neo4j')
        self.dimension = embedding_service.get_dimension() if embedding_service else 384
        
        # Local caches for faster operation
        self.document_cache = {}  # Maps document IDs to document data
        
        # Connect to Neo4j
        self._initialize_database()
        
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
                    logger.info(record["message"])
            
            # Initialize schema (constraints and indexes)
            with self.driver.session(database=self.database) as session:
                # Create constraints
                session.run("""
                    CREATE CONSTRAINT document_id IF NOT EXISTS 
                    FOR (d:Document) REQUIRE d.id IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT entity_name IF NOT EXISTS 
                    FOR (e:Entity) REQUIRE e.name IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT keyword_name IF NOT EXISTS 
                    FOR (k:Keyword) REQUIRE k.name IS UNIQUE
                """)
                
                # Create indexes
                session.run("""
                    CREATE INDEX document_content IF NOT EXISTS 
                    FOR (d:Document) ON (d.content)
                """)
                
                # Create vector index if using embeddings
                if self.embedding_service:
                    # Check if vector index plugin is available
                    try:
                        session.run("""
                            CALL db.index.vector.createNodeIndex(
                                'document_embeddings',
                                'Document',
                                'embedding',
                                $dimension,
                                'cosine'
                            )
                        """, dimension=self.dimension)
                        logger.info(f"Created Neo4j vector index with dimension {self.dimension}")
                    except Exception as e:
                        logger.warning(f"Could not create vector index, vector search will not be available: {e}")
            
            logger.info("Neo4j database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Neo4j database: {str(e)}")
            raise
    
    def _extract_entities_and_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract entities and keywords from text for graph relationships.
        This is a simple implementation - in production, you'd use NER and more sophisticated methods.
        """
        # Simple entity extraction (capitalized words, could be improved with NER)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Simple keyword extraction (remove common words, get significant terms)
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stopwords = {'and', 'the', 'for', 'with', 'that', 'this', 'are', 'from', 'have', 'has', 'not'}
        keywords = [word for word in words if word not in stopwords]
        
        # Remove duplicates and sort
        entities = sorted(set(entities))
        keywords = sorted(set(keywords))
        
        # Limit to most frequent or important (for simplicity, just take top N)
        entities = entities[:20]  # Limit to top 20 entities
        keywords = keywords[:30]  # Limit to top 30 keywords
        
        return entities, keywords
    
    def add_documents(self, documents: List[LangchainDocument]) -> List[str]:
        """Add documents to the graph store."""
        try:
            if not documents:
                return []
            
            document_ids = []
            
            # Process in batches for efficiency
            for doc in documents:
                # Generate document ID
                doc_id = str(uuid.uuid4())
                document_ids.append(doc_id)
                
                # Extract entities and keywords
                text = doc.page_content
                entities, keywords = self._extract_entities_and_keywords(text)
                
                # Cache document data
                self.document_cache[doc_id] = {
                    'content': text,
                    'metadata': doc.metadata,
                    'entities': entities,
                    'keywords': keywords
                }
                
                # Generate embedding if embedding service is available
                embedding = None
                if self.embedding_service:
                    embedding = self.embedding_service.embed_texts([text])[0]
                
                # Create document node with its properties
                with self.driver.session(database=self.database) as session:
                    # Create document node
                    if embedding is not None:
                        session.run("""
                            CREATE (d:Document {
                                id: $id, 
                                content: $content, 
                                metadata: $metadata,
                                embedding: $embedding
                            })
                        """, id=doc_id, content=text, metadata=json.dumps(doc.metadata), embedding=embedding)
                    else:
                        session.run("""
                            CREATE (d:Document {
                                id: $id, 
                                content: $content, 
                                metadata: $metadata
                            })
                        """, id=doc_id, content=text, metadata=json.dumps(doc.metadata))
                    
                    # Create entity nodes and relationships
                    for entity in entities:
                        session.run("""
                            MERGE (e:Entity {name: $name})
                            WITH e
                            MATCH (d:Document {id: $doc_id})
                            MERGE (d)-[:MENTIONS]->(e)
                        """, name=entity, doc_id=doc_id)
                    
                    # Create keyword nodes and relationships
                    for keyword in keywords:
                        session.run("""
                            MERGE (k:Keyword {name: $name})
                            WITH k
                            MATCH (d:Document {id: $doc_id})
                            MERGE (d)-[:CONTAINS]->(k)
                        """, name=keyword, doc_id=doc_id)
                
                # Create document-to-document relationships
                self._create_document_relationships(doc_id, entities, keywords)
            
            logger.info(f"Added {len(documents)} documents to Neo4j graph")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Neo4j graph: {str(e)}")
            raise
    
    def _create_document_relationships(self, doc_id: str, entities: List[str], keywords: List[str]):
        """Create relationships between documents based on shared entities and keywords."""
        try:
            # Find documents sharing entities
            with self.driver.session(database=self.database) as session:
                # Find documents with shared entities and create relationships
                session.run("""
                    MATCH (d1:Document {id: $doc_id})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(d2:Document)
                    WHERE d1 <> d2
                    MERGE (d1)-[r:RELATED_TO {type: 'entity', strength: 0.0}]->(d2)
                    ON MATCH SET r.strength = r.strength + 0.2
                """, doc_id=doc_id)
                
                # Find documents with shared keywords and create or strengthen relationships
                session.run("""
                    MATCH (d1:Document {id: $doc_id})-[:CONTAINS]->(k:Keyword)<-[:CONTAINS]-(d2:Document)
                    WHERE d1 <> d2
                    MERGE (d1)-[r:RELATED_TO {type: 'keyword', strength: 0.0}]->(d2)
                    ON MATCH SET r.strength = r.strength + 0.1
                """, doc_id=doc_id)
                
            logger.debug(f"Created document relationships for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error creating document relationships: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        similarity_threshold: float = 0.3,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """
        Search for similar documents using graph algorithms.
        
        This implementation uses multiple strategies:
        1. Vector similarity (if embedding service available)
        2. Keyword and entity matching
        3. Graph traversal to find related documents
        """
        try:
            # Extract entities and keywords from query
            entities, keywords = self._extract_entities_and_keywords(query)
            
            results = []
            doc_scores = {}  # Track scores by document ID
            
            with self.driver.session(database=self.database) as session:
                # Strategy 1: Direct match on entities and keywords
                if entities or keywords:
                    # Find documents that mention the entities in query
                    if entities:
                        entity_query = """
                            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
                            WHERE e.name IN $entities
                            RETURN d.id AS id, d.content AS content, d.metadata AS metadata,
                                   count(DISTINCT e) AS matching_count
                            ORDER BY matching_count DESC
                            LIMIT 20
                        """
                        result = session.run(entity_query, entities=entities)
                        for record in result:
                            doc_id = record["id"]
                            doc_content = record["content"]
                            doc_metadata = json.loads(record["metadata"])
                            score = min(0.9, 0.5 + (record["matching_count"] / len(entities) * 0.4))
                            
                            # Apply filter if specified
                            if filter and not self._matches_filter(doc_metadata, filter):
                                continue
                                
                            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0), score)
                            results.append((
                                LangchainDocument(page_content=doc_content, metadata=doc_metadata),
                                score
                            ))
                    
                    # Find documents that contain the keywords in query
                    if keywords:
                        keyword_query = """
                            MATCH (d:Document)-[:CONTAINS]->(k:Keyword)
                            WHERE k.name IN $keywords
                            RETURN d.id AS id, d.content AS content, d.metadata AS metadata,
                                   count(DISTINCT k) AS matching_count
                            ORDER BY matching_count DESC
                            LIMIT 20
                        """
                        result = session.run(keyword_query, keywords=keywords)
                        for record in result:
                            doc_id = record["id"]
                            doc_content = record["content"]
                            doc_metadata = json.loads(record["metadata"])
                            score = min(0.8, 0.3 + (record["matching_count"] / len(keywords) * 0.5))
                            
                            # Apply filter if specified
                            if filter and not self._matches_filter(doc_metadata, filter):
                                continue
                                
                            if doc_id in doc_scores:
                                # We've already seen this document, update score if higher
                                doc_scores[doc_id] = max(doc_scores.get(doc_id), score)
                            else:
                                doc_scores[doc_id] = score
                                results.append((
                                    LangchainDocument(page_content=doc_content, metadata=doc_metadata),
                                    score
                                ))
                
                # Strategy 2: Vector similarity search if embedding service is available
                if self.embedding_service:
                    # Generate query embedding
                    query_embedding = self.embedding_service.embed_texts([query])[0]
                    
                    # Vector search in Neo4j using vector index
                    try:
                        vector_query = """
                            CALL db.index.vector.queryNodes('document_embeddings', $k, $embedding)
                            YIELD node, score
                            RETURN node.id AS id, node.content AS content, node.metadata AS metadata, score
                        """
                        result = session.run(vector_query, k=k*2, embedding=query_embedding)
                        
                        for record in result:
                            doc_id = record["id"]
                            doc_content = record["content"]
                            doc_metadata = json.loads(record["metadata"])
                            
                            # Neo4j vector search returns similarity (higher is better)
                            # Normalize to [0,1] range
                            score = min(0.95, max(0.0, record["score"]))
                            
                            # Apply filter if specified
                            if filter and not self._matches_filter(doc_metadata, filter):
                                continue
                                
                            if doc_id in doc_scores:
                                # We've already seen this document, update score if higher
                                doc_scores[doc_id] = max(doc_scores.get(doc_id), score)
                            else:
                                doc_scores[doc_id] = score
                                results.append((
                                    LangchainDocument(page_content=doc_content, metadata=doc_metadata),
                                    score
                                ))
                    except Exception as e:
                        logger.warning(f"Vector search failed, falling back to graph search: {e}")
                
                # Strategy 3: Graph traversal to find related documents
                if entities or keywords:
                    # Find documents related to the ones we've already found
                    doc_ids = list(doc_scores.keys())
                    if doc_ids:
                        related_query = """
                            MATCH (d1:Document)-[r:RELATED_TO]->(d2:Document)
                            WHERE d1.id IN $doc_ids AND NOT d2.id IN $doc_ids
                            RETURN d2.id AS id, d2.content AS content, d2.metadata AS metadata,
                                   r.strength AS strength, d1.id AS source_id
                            ORDER BY strength DESC
                            LIMIT 10
                        """
                        result = session.run(related_query, doc_ids=doc_ids)
                        
                        for record in result:
                            doc_id = record["id"]
                            doc_content = record["content"]
                            doc_metadata = json.loads(record["metadata"])
                            source_id = record["source_id"]
                            
                            # Score is based on relationship strength and source document score
                            source_score = doc_scores.get(source_id, 0.5)
                            rel_strength = record["strength"]
                            score = min(0.7, source_score * 0.7 * (0.5 + rel_strength))
                            
                            # Apply filter if specified
                            if filter and not self._matches_filter(doc_metadata, filter):
                                continue
                                
                            if doc_id in doc_scores:
                                # We've already seen this document, update score if higher
                                doc_scores[doc_id] = max(doc_scores.get(doc_id), score)
                            else:
                                doc_scores[doc_id] = score
                                results.append((
                                    LangchainDocument(page_content=doc_content, metadata=doc_metadata),
                                    score
                                ))
            
            # Update results with the latest scores
            for i, (doc, _) in enumerate(results):
                doc_id = None
                # Extract document ID from metadata if available
                for k, v in doc.metadata.items():
                    if k.lower() == 'id' or k.lower() == 'document_id':
                        doc_id = v
                        break
                
                if not doc_id:
                    # If we can't find ID in metadata, use the content as a key
                    doc_id = doc.page_content[:100]  # Use first 100 chars as ID
                
                results[i] = (doc, doc_scores.get(doc_id, 0.0))
            
            # Sort by score and apply threshold
            results = [(doc, score) for doc, score in results if score >= similarity_threshold]
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            # Return empty list on error
            return []
    

    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the graph store."""
        try:
            if not document_ids:
                return
            
            with self.driver.session(database=self.database) as session:
                # Delete document nodes and their relationships
                session.run("""
                    UNWIND $doc_ids AS doc_id
                    MATCH (d:Document {id: doc_id})
                    OPTIONAL MATCH (d)-[r]-()
                    DELETE r, d
                """, doc_ids=document_ids)
                
                # Clean up orphaned entities and keywords
                session.run("""
                    MATCH (e:Entity)
                    WHERE NOT (e)<-[:MENTIONS]-()
                    DELETE e
                """)
                
                session.run("""
                    MATCH (k:Keyword)
                    WHERE NOT (k)<-[:CONTAINS]-()
                    DELETE k
                """)
            
            # Remove from cache
            for doc_id in document_ids:
                if doc_id in self.document_cache:
                    del self.document_cache[doc_id]
            
            logger.info(f"Deleted {len(document_ids)} documents from Neo4j graph")
            
        except Exception as e:
            logger.error(f"Error deleting documents from Neo4j graph: {str(e)}")
            raise
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (d:Document) RETURN count(d) AS count")
                return result.single()["count"]
        except Exception as e:
            logger.error(f"Error getting document count from Neo4j graph: {str(e)}")
            return 0
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph structure."""
        try:
            stats = {}
            
            with self.driver.session(database=self.database) as session:
                # Get node counts
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] AS label, count(n) AS count
                """)
                
                for record in result:
                    stats[f"{record['label'].lower()}_count"] = record["count"]
                
                # Get relationship counts
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) AS type, count(r) AS count
                """)
                
                for record in result:
                    stats[f"{record['type'].lower()}_count"] = record["count"]
                
                # Get database size estimate
                result = session.run("""
                    CALL db.stats.size() YIELD nodeSize, relationshipSize, propertySize, totalSize
                    RETURN nodeSize, relationshipSize, propertySize, totalSize
                """)
                
                if result.peek():
                    record = result.single()
                    stats["node_size_bytes"] = record["nodeSize"]
                    stats["relationship_size_bytes"] = record["relationshipSize"]
                    stats["property_size_bytes"] = record["propertySize"]
                    stats["total_size_bytes"] = record["totalSize"]
                    stats["total_size_mb"] = record["totalSize"] / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {"error": str(e)}
    
    def close(self):
        """Close the Neo4j connection."""
        try:
            if hasattr(self, 'driver'):
                self.driver.close()
                logger.info("Neo4j connection closed")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")
