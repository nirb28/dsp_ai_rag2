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
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for similar documents."""
        raise NotImplementedError("Subclasses must implement similarity_search")
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        raise NotImplementedError("Subclasses must implement delete_documents")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        raise NotImplementedError("Subclasses must implement get_document_count")
