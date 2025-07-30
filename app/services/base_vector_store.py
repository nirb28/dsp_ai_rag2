from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document as LangchainDocument
import logging

logger = logging.getLogger(__name__)

class BaseVectorStore:
    """Base class for vector stores."""
    
    def __init__(self, config):
        """Initialize the vector store with configuration."""
        self.config = config
    
    def add_documents(self, documents: List[LangchainDocument]) -> List[str]:
        """Add documents to the vector store."""
        raise NotImplementedError("Subclasses must implement add_documents")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        similarity_threshold: float = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for similar documents with optional metadata filtering.
        
        Args:
            query: The search query string
            k: Number of documents to return
            similarity_threshold: Minimum similarity score threshold
            filter: MongoDB-style filter conditions for metadata (LangChain convention)
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        raise NotImplementedError("Subclasses must implement similarity_search")
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        raise NotImplementedError("Subclasses must implement delete_documents")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        raise NotImplementedError("Subclasses must implement get_document_count")
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter using MongoDB-style operators.
        
        Supports the following operators:
        - Simple equality: {"key": "value"}
        - $eq: {"key": {"$eq": "value"}}
        - $neq: {"key": {"$neq": "value"}}
        - $gt: {"key": {"$gt": 5}}
        - $lt: {"key": {"$lt": 10}}
        - $gte: {"key": {"$gte": 5}}
        - $lte: {"key": {"$lte": 10}}
        - $in: {"key": {"$in": ["val1", "val2"]}}
        - $nin: {"key": {"$nin": ["val1", "val2"]}}
        - $and: {"$and": [{"key1": "val1"}, {"key2": "val2"}]}
        - $or: {"$or": [{"key1": "val1"}, {"key2": "val2"}]}
        - $not: {"$not": {"key": "value"}}
        """
        if not filter_dict:
            return True
            
        for key, value in filter_dict.items():
            if key == "$and":
                if not isinstance(value, list):
                    return False
                if not all(self._matches_filter(metadata, condition) for condition in value):
                    return False
            elif key == "$or":
                if not isinstance(value, list):
                    return False
                if not any(self._matches_filter(metadata, condition) for condition in value):
                    return False
            elif key == "$not":
                if self._matches_filter(metadata, value):
                    return False
            else:
                # Handle field-specific conditions
                if key not in metadata:
                    return False
                    
                field_value = metadata[key]
                
                if isinstance(value, dict):
                    # Handle operator-based conditions
                    for op, op_value in value.items():
                        if op == "$eq":
                            if field_value != op_value:
                                return False
                        elif op == "$neq":
                            if field_value == op_value:
                                return False
                        elif op == "$gt":
                            if not (isinstance(field_value, (int, float)) and field_value > op_value):
                                return False
                        elif op == "$lt":
                            if not (isinstance(field_value, (int, float)) and field_value < op_value):
                                return False
                        elif op == "$gte":
                            if not (isinstance(field_value, (int, float)) and field_value >= op_value):
                                return False
                        elif op == "$lte":
                            if not (isinstance(field_value, (int, float)) and field_value <= op_value):
                                return False
                        elif op == "$in":
                            if not isinstance(op_value, list) or field_value not in op_value:
                                return False
                        elif op == "$nin":
                            if not isinstance(op_value, list) or field_value in op_value:
                                return False
                        else:
                            logger.warning(f"Unsupported operator: {op}")
                            return False
                else:
                    # Simple equality check
                    if field_value != value:
                        return False
                        
        return True
