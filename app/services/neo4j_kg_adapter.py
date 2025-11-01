import logging
from typing import List, Dict, Any, Optional
import asyncio

from langchain.docstore.document import Document as LangchainDocument

from app.services.base_vector_store import BaseVectorStore
from app.services.neo4j_knowledge_graph_store import Neo4jKnowledgeGraphStore

logger = logging.getLogger(__name__)


class Neo4jKnowledgeGraphAdapter(BaseVectorStore):
    """
    Adapter to make Neo4jKnowledgeGraphStore compatible with BaseVectorStore interface.
    
    This adapter bridges the gap between the async knowledge graph operations
    and the synchronous vector store interface expected by the RAG service.
    """
    
    def __init__(self, config: Dict[str, Any], llm_config_name: str):
        """Initialize the adapter with knowledge graph store."""
        super().__init__(config)
        self.kg_store = Neo4jKnowledgeGraphStore(config, llm_config_name)
        self.llm_config_name = llm_config_name
        
    def add_documents(self, documents: List[LangchainDocument]) -> List[str]:
        """
        Add documents to the knowledge graph store.
        
        This method runs the async add_documents method synchronously.
        """
        try:
            # Run the async method in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(self.kg_store.add_documents(documents))
                
                # Generate document IDs based on the number of processed documents
                doc_ids = []
                for i, doc in enumerate(documents):
                    doc_id = doc.metadata.get('id', f'kg_doc_{i}')
                    doc_ids.append(doc_id)
                
                logger.info(f"Added {result['processed_documents']} documents to knowledge graph")
                return doc_ids
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error adding documents to knowledge graph: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, similarity_threshold: float = 0.7,
                         filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search using the knowledge graph.
        
        This method uses the semantic search capability of the knowledge graph
        to find relevant documents based on entity and relationship matching.
        """
        try:
            # Use the knowledge graph's semantic search
            results = self.kg_store.semantic_search(query, k)
            
            # Convert results to the expected format
            formatted_results = []
            for result in results:
                # Create a document-like structure
                doc_result = {
                    'id': result.get('id', 'unknown'),
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('score', 0.0),
                    'source': result.get('filename', 'unknown'),
                    # Add knowledge graph specific information
                    'matched_entity': result.get('matched_entity', ''),
                    'entity_type': result.get('entity_type', ''),
                    'search_type': 'knowledge_graph'
                }
                
                # Apply metadata filter if provided
                if filter:
                    if self._matches_filter(doc_result['metadata'], filter):
                        formatted_results.append(doc_result)
                else:
                    formatted_results.append(doc_result)
            
            # Apply similarity threshold
            filtered_results = [
                result for result in formatted_results 
                if result['score'] >= similarity_threshold
            ]
            
            logger.info(f"Knowledge graph search returned {len(filtered_results)} results for query: {query}")
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Error performing knowledge graph search: {str(e)}")
            raise
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if metadata matches the filter criteria.
        Supports basic MongoDB-style operators.
        """
        try:
            for key, value in filter_dict.items():
                if key.startswith('$'):
                    # Handle logical operators
                    if key == '$and':
                        if not all(self._matches_filter(metadata, condition) for condition in value):
                            return False
                    elif key == '$or':
                        if not any(self._matches_filter(metadata, condition) for condition in value):
                            return False
                    elif key == '$not':
                        if self._matches_filter(metadata, value):
                            return False
                else:
                    # Handle field-level conditions
                    if key not in metadata:
                        return False
                    
                    field_value = metadata[key]
                    
                    if isinstance(value, dict):
                        # Handle field operators
                        for op, op_value in value.items():
                            if op == '$eq':
                                if field_value != op_value:
                                    return False
                            elif op == '$neq':
                                if field_value == op_value:
                                    return False
                            elif op == '$in':
                                if field_value not in op_value:
                                    return False
                            elif op == '$nin':
                                if field_value in op_value:
                                    return False
                            elif op == '$gt':
                                if not (field_value > op_value):
                                    return False
                            elif op == '$gte':
                                if not (field_value >= op_value):
                                    return False
                            elif op == '$lt':
                                if not (field_value < op_value):
                                    return False
                            elif op == '$lte':
                                if not (field_value <= op_value):
                                    return False
                    else:
                        # Simple equality check
                        if field_value != value:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching filter: {str(e)}")
            return False
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the knowledge graph."""
        try:
            result = self.kg_store.delete_documents(document_ids)
            logger.info(f"Deleted documents from knowledge graph: {result}")
        except Exception as e:
            logger.error(f"Error deleting documents from knowledge graph: {str(e)}")
            raise
    
    def get_document_count(self) -> int:
        """Get the number of documents in the knowledge graph."""
        try:
            return self.kg_store.get_document_count()
        except Exception as e:
            logger.error(f"Error getting document count from knowledge graph: {str(e)}")
            return 0
    
    # Additional knowledge graph specific methods
    def query_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the knowledge graph."""
        return self.kg_store.query_graph(query, limit)
    
    def find_entities(self, entity_name: str = None, entity_type: str = None, 
                     limit: int = 10) -> List[Dict[str, Any]]:
        """Find entities in the knowledge graph."""
        return self.kg_store.find_entities(entity_name, entity_type, limit)
    
    def find_relationships(self, source_entity: str = None, target_entity: str = None,
                          relationship_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Find relationships in the knowledge graph."""
        return self.kg_store.find_relationships(source_entity, target_entity, relationship_type, limit)
    
    def find_connected_entities(self, entity_name: str, max_depth: int = 2, 
                               limit: int = 20) -> List[Dict[str, Any]]:
        """Find entities connected to a given entity."""
        return self.kg_store.find_connected_entities(entity_name, max_depth, limit)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return self.kg_store.get_graph_stats()
    
    def close(self):
        """Close the knowledge graph connection."""
        self.kg_store.close()
