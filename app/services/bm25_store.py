from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from langchain.docstore.document import Document
import json
import os
import pickle
from rank_bm25 import BM25Okapi  # Pure Python implementation, no model downloads
import logging

from app.services.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

class BM25VectorStore(BaseVectorStore):
    """A BM25-based retrieval system that implements the BaseVectorStore interface."""

    def __init__(self, config: Dict[str, Any], embedding_service=None):
        """Initialize the BM25 store.
        
        Note: embedding_service is ignored for BM25 but included to match the interface.
        """
        super().__init__(config)
        self.config = config
        self.documents = []
        self.bm25 = None
        self.tokenized_corpus = []
        self.index_path = config.get('index_path', './storage/bm25_index')
        self.name = config.get('name', 'bm25_store')
        
        # Create index directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Try to load existing index
        self._load_index()
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and lowercasing."""
        return text.lower().split()
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the BM25 index."""
        if not documents:
            return
            
        # Add documents to our list
        self.documents.extend(documents)
        
        # Update tokenized corpus
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.documents]
        
        # Create new BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Save index
        self._save_index()
        
        logger.info(f"Added {len(documents)} documents to BM25 index. Total documents: {len(self.documents)}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        similarity_threshold: float = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for documents similar to the query using BM25 with optional metadata filtering.
        
        Args:
            query: The search query string
            k: Number of documents to return
            similarity_threshold: Minimum similarity score threshold
            filter: MongoDB-style filter conditions for metadata (LangChain convention)
        """
        if not self.bm25 or not self.documents:
            return []
            
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get scores using BM25
        scores = self.bm25.get_scores(tokenized_query)
        
        # Create document-score pairs and apply filters
        doc_score_pairs = []
        for i in range(len(scores)):
            doc = self.documents[i]
            score = scores[i]
            
            # Apply similarity threshold if specified
            if similarity_threshold is not None and score <= similarity_threshold:
                continue
                
            # Apply metadata filtering if specified
            if filter and not self._matches_filter(doc.metadata, filter):
                continue
                
            doc_score_pairs.append((doc, score))
        
        # Sort by score in descending order
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return doc_score_pairs[:k]
    
    def get_document_count(self) -> int:
        """Return the number of documents in the index."""
        return len(self.documents)
    
    def get_all_documents(self, limit: int = None) -> List[Document]:
        """Return all documents in the index, up to the specified limit."""
        if limit:
            return self.documents[:limit]
        return self.documents
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents by ID. Not efficient - rebuilds the entire index."""
        if not document_ids:
            return
            
        # Keep track of documents to keep
        keep_docs = []
        
        # Filter documents to keep
        for doc in self.documents:
            doc_id = doc.metadata.get('id')
            if not doc_id or doc_id not in document_ids:
                keep_docs.append(doc)
        
        # Update documents list
        self.documents = keep_docs
        
        # Rebuild index
        if self.documents:
            self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.documents]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.tokenized_corpus = []
            self.bm25 = None
        
        # Save index
        self._save_index()
        
        logger.info(f"Deleted {len(document_ids)} documents from BM25 index. Remaining documents: {len(self.documents)}")
    
    def _save_index(self) -> None:
        """Save the BM25 index and documents to disk."""
        try:
            # Create separate files for documents and BM25 index
            with open(f"{self.index_path}_{self.name}_documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
                
            if self.bm25:
                with open(f"{self.index_path}_{self.name}_bm25.pkl", 'wb') as f:
                    pickle.dump(self.bm25, f)
                    
            with open(f"{self.index_path}_{self.name}_tokenized_corpus.pkl", 'wb') as f:
                pickle.dump(self.tokenized_corpus, f)
                
            logger.info(f"Saved BM25 index with {len(self.documents)} documents to {self.index_path}_{self.name}")
        except Exception as e:
            logger.error(f"Error saving BM25 index: {str(e)}")
    
    def _load_index(self) -> None:
        """Load the BM25 index and documents from disk."""
        try:
            docs_path = f"{self.index_path}_{self.name}_documents.pkl"
            bm25_path = f"{self.index_path}_{self.name}_bm25.pkl"
            corpus_path = f"{self.index_path}_{self.name}_tokenized_corpus.pkl"
            
            if os.path.exists(docs_path) and os.path.exists(bm25_path) and os.path.exists(corpus_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                    
                with open(bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
                    
                with open(corpus_path, 'rb') as f:
                    self.tokenized_corpus = pickle.load(f)
                    
                logger.info(f"Loaded BM25 index with {len(self.documents)} documents from {self.index_path}_{self.name}")
            else:
                logger.info(f"No existing BM25 index found at {self.index_path}_{self.name}")
        except Exception as e:
            logger.error(f"Error loading BM25 index: {str(e)}")
            # Reset to empty state if loading fails
            self.documents = []
            self.bm25 = None
            self.tokenized_corpus = []
    

