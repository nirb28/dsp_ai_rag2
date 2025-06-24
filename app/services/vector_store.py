import os
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

import faiss
from langchain.docstore.document import Document as LangchainDocument

from app.config import VectorStoreConfig
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, config: VectorStoreConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self.dimension = embedding_service.get_dimension()
        self.index = None
        self.documents = []
        self.metadata = []
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load FAISS index."""
        try:
            index_path = Path(self.config.index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            
            index_file = index_path / "index.faiss"
            docs_file = index_path / "documents.pkl"
            metadata_file = index_path / "metadata.json"
            
            if index_file.exists() and docs_file.exists():
                # Load existing index
                self.index = faiss.read_index(str(index_file))
                
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.metadata = json.load(f)
                else:
                    self.metadata = [{}] * len(self.documents)
                
                logger.info(f"Loaded existing FAISS index with {len(self.documents)} documents")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                self.documents = []
                self.metadata = []
                logger.info(f"Created new FAISS index with dimension {self.dimension}")
                
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            raise

    def add_documents(self, documents: List[LangchainDocument]) -> List[str]:
        """Add documents to the vector store."""
        try:
            if not documents:
                return []
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            doc_metadata = [doc.metadata for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedding_service.embed_texts(texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            start_id = len(self.documents)
            self.index.add(embeddings_array)
            
            # Store documents and metadata
            self.documents.extend(documents)
            self.metadata.extend(doc_metadata)
            
            # Save index
            self._save_index()
            
            # Generate IDs
            doc_ids = [f"doc_{start_id + i}" for i in range(len(documents))]
            
            logger.info(f"Added {len(documents)} documents to FAISS index")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS index: {str(e)}")
            raise

    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        similarity_threshold: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for similar documents."""
        try:
            if len(self.documents) == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize query vector
            faiss.normalize_L2(query_vector)
            
            # Search
            search_k = min(k * 2, len(self.documents))  # Search more to allow for filtering
            scores, indices = self.index.search(query_vector, search_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                    
                if score < similarity_threshold:
                    continue
                
                doc = self.documents[idx]
                metadata = self.metadata[idx]
                
                # Apply metadata filtering if specified
                if filter_metadata:
                    if not all(metadata.get(key) == value for key, value in filter_metadata.items()):
                        continue
                
                results.append((doc, float(score)))
                
                if len(results) >= k:
                    break
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {str(e)}")
            raise

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        # Note: FAISS doesn't support deletion directly
        # This would require rebuilding the index
        logger.warning("Document deletion not implemented for FAISS. Consider rebuilding the index.")
        return False

    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        return len(self.documents)

    def _save_index(self):
        """Save the FAISS index and associated data."""
        try:
            index_path = Path(self.config.index_path)
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path / "index.faiss"))
            
            # Save documents
            with open(index_path / "documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save metadata
            with open(index_path / "metadata.json", 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise

class VectorStoreManager:
    def __init__(self):
        self.stores: Dict[str, FAISSVectorStore] = {}

    def get_store(self, collection_name: str, config: VectorStoreConfig, embedding_service: EmbeddingService) -> FAISSVectorStore:
        """Get or create a vector store for a collection."""
        if collection_name not in self.stores:
            # Create collection-specific config
            collection_config = VectorStoreConfig(
                type=config.type,
                index_path=f"{config.index_path}/{collection_name}",
                dimension=config.dimension
            )
            self.stores[collection_name] = FAISSVectorStore(collection_config, embedding_service)
        
        return self.stores[collection_name]

    def list_collections(self) -> List[str]:
        """List all available collections."""
        return list(self.stores.keys())

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        if collection_name in self.stores:
            del self.stores[collection_name]
            # Also delete files if needed
            return True
        return False
