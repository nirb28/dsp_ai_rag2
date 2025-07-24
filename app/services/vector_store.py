import os
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import uuid
from enum import Enum

import faiss
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from langchain.docstore.document import Document as LangchainDocument

from app.config import VectorStoreConfig, settings
from app.services.embedding_service import EmbeddingService
from app.services.base_vector_store import BaseVectorStore
from app.services.bm25_store import BM25VectorStore
from app.services.networkx_graph_store import NetworkXGraphStore

logger = logging.getLogger(__name__)

class VectorStore(str, Enum):
    FAISS = "faiss"
    REDIS = "redis"
    BM25 = "bm25"
    NETWORKX = "networkx"

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, config: VectorStoreConfig, embedding_service: EmbeddingService):
        super().__init__(config)
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

class RedisVectorStore(BaseVectorStore):
    def __init__(self, config: VectorStoreConfig, embedding_service: EmbeddingService):
        super().__init__(config)
        self.embedding_service = embedding_service
        self.dimension = embedding_service.get_dimension()
        self.redis_client = None
        self.index_name = config.redis_index_name
        self.vector_field_name = "embedding"
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Redis client and create index if it doesn't exist."""
        try:
            # Connect to Redis using configuration
            redis_url = f"redis://{':' + self.config.redis_password + '@' if self.config.redis_password else ''}{self.config.redis_host}:{self.config.redis_port}"
            
            # Use environment variable if available
            if os.environ.get("REDIS_URL"):
                redis_url = os.environ.get("REDIS_URL")
                
            self.redis_client = redis.from_url(redis_url)
            
            # Check if the index already exists
            existing_indices = self.redis_client.execute_command("FT._LIST")
            
            if self.index_name.encode() not in existing_indices:
                # Create index with vector search capability
                text_field = TextField(name="content")
                metadata_field = TextField(name="metadata", no_stem=True)
                filename_field = TextField(name="filename", no_stem=True)
                doc_id_field = TextField(name="doc_id", no_stem=True)
                
                # Vector field for embeddings
                vector_field = VectorField(
                    self.vector_field_name,
                    "FLAT", {
                        "TYPE": "FLOAT32",
                        "DIM": self.dimension,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
                
                # Create the index
                self.redis_client.ft(self.index_name).create_index(
                    [text_field, metadata_field, filename_field, doc_id_field, vector_field],
                    definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
                )
                logger.info(f"Created new Redis index: {self.index_name}")
            else:
                logger.info(f"Using existing Redis index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error initializing Redis client: {str(e)}")
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
            
            # Prepare pipeline for batch insert
            pipe = self.redis_client.pipeline()
            doc_ids = []
            
            # Insert each document
            for i, (doc, metadata, embedding) in enumerate(zip(documents, doc_metadata, embeddings)):
                # Generate a unique ID
                doc_id = f"doc:{str(uuid.uuid4())}"
                doc_ids.append(doc_id)
                
                # Convert metadata to JSON string
                metadata_json = json.dumps(metadata)
                filename = metadata.get('filename', 'Unknown')
                
                # Store the document with its embedding
                pipe.hset(
                    doc_id,
                    mapping={
                        "content": doc.page_content,
                        "metadata": metadata_json,
                        "filename": filename,
                        "doc_id": doc_id.replace("doc:", ""),
                        self.vector_field_name: np.array(embedding, dtype=np.float32).tobytes()
                    }
                )
            
            # Execute the pipeline
            pipe.execute()
            
            logger.info(f"Added {len(documents)} documents to Redis index")
            return [doc_id.replace("doc:", "") for doc_id in doc_ids]
            
        except Exception as e:
            logger.error(f"Error adding documents to Redis index: {str(e)}")
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
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            query_vector = np.array(query_embedding, dtype=np.float32)
            
            # Build Redis query
            base_query = f"*=>[KNN {k} @{self.vector_field_name} $vector AS score]"
            
            # Add metadata filters if provided
            if filter_metadata:
                for key, value in filter_metadata.items():
                    base_query += f" @metadata:{key}:{value}"
            
            # Create query object
            redis_query = Query(base_query)\
                .sort_by("score")\
                .dialect(2)\
                .return_fields("content", "metadata", "score", "filename")\
                .paging(0, k)
                
            # Execute the query
            query_params = {"vector": query_vector.tobytes()}
            results = self.redis_client.ft(self.index_name).search(redis_query, query_params)
            
            # Process results
            output = []
            for doc in results.docs:
                score = float(doc.score)
                
                # Skip if below threshold
                if score < similarity_threshold:
                    continue
                
                # Extract content and metadata
                content = doc.content
                try:
                    metadata = json.loads(doc.metadata)
                except:
                    metadata = {"filename": doc.filename}
                
                # Create LangChain document
                langchain_doc = LangchainDocument(page_content=content, metadata=metadata)
                
                # Append to results
                output.append((langchain_doc, score))
            
            logger.info(f"Found {len(output)} similar documents for query in Redis")
            return output
            
        except Exception as e:
            logger.error(f"Error searching Redis index: {str(e)}")
            raise
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        try:
            pipe = self.redis_client.pipeline()
            
            for doc_id in document_ids:
                pipe.delete(f"doc:{doc_id}")
            
            pipe.execute()
            logger.info(f"Deleted {len(document_ids)} documents from Redis index")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from Redis: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        try:
            info = self.redis_client.ft(self.index_name).info()
            return info['num_docs']
        except Exception as e:
            logger.error(f"Error getting document count from Redis: {str(e)}")
            return 0

class VectorStoreManager:
    def __init__(self):
        self.stores: Dict[str, Any] = {}  # Changed to Any to support multiple vector store types

    def get_vector_store(self, configuration_name: str, config: VectorStoreConfig, embedding_config: dict) -> Any:
        """Get or create a vector store for a configuration."""
        if configuration_name not in self.stores:
            # Create configuration-specific config with updated paths for configuration
            if config.type == VectorStore.FAISS:
                # For FAISS, we need to update the index path for the specific configuration
                configuration_config = VectorStoreConfig(
                    type=config.type,
                    index_path=f"{config.index_path}/{configuration_name}",
                    dimension=config.dimension,
                    # Include Redis settings to avoid validation errors
                    redis_host=config.redis_host,
                    redis_port=config.redis_port,
                    redis_password=config.redis_password,
                    redis_index_name=config.redis_index_name
                )
                embedding_service = EmbeddingService(embedding_config)
                self.stores[configuration_name] = FAISSVectorStore(configuration_config, embedding_service)
                
            elif config.type == VectorStore.REDIS:
                # For Redis, use configuration name as part of index name
                configuration_config = VectorStoreConfig(
                    type=config.type,
                    index_path=config.index_path,  # Not needed for Redis but keep for consistency
                    dimension=config.dimension,
                    redis_host=config.redis_host,
                    redis_port=config.redis_port,
                    redis_password=config.redis_password,
                    # Use configuration name in the index name
                    redis_index_name=f"{config.redis_index_name}-{configuration_name}"
                )
                embedding_service = EmbeddingService(embedding_config)
                self.stores[configuration_name] = RedisVectorStore(configuration_config, embedding_service)
                
            elif config.type == VectorStore.BM25:
                # For BM25, we need a specific path for the index
                configuration_config = {
                    'type': config.type,
                    'index_path': f"{config.index_path}/bm25",
                    'name': configuration_name  # Pass configuration name to BM25 store
                }
                # Note: BM25 doesn't need an embedding service but we pass it to maintain interface compatibility
                embedding_service = None
                if embedding_config.get('enabled', False):
                    # If embeddings are enabled in config, create the service but it won't be used by BM25
                    embedding_service = EmbeddingService(embedding_config)
                self.stores[configuration_name] = BM25VectorStore(configuration_config, embedding_service)
                logger.info(f"Created BM25 vector store for configuration {configuration_name}")
                
            elif config.type == VectorStore.NETWORKX:
                # For NetworkX, we need a specific path for the graph storage
                configuration_config = {
                    'type': config.type,
                    'index_path': f"{config.index_path}/networkx",
                    'name': configuration_name  # Pass configuration name to NetworkX store
                }
                # NetworkX can optionally use embeddings for enhanced similarity
                embedding_service = None
                if embedding_config.get('enabled', False):
                    embedding_service = EmbeddingService(embedding_config)
                self.stores[configuration_name] = NetworkXGraphStore(configuration_config, embedding_service)
                logger.info(f"Created NetworkX graph store for configuration {configuration_name}")
            else:
                raise ValueError(f"Unsupported vector store type: {config.type}")
                
            logger.info(f"Created new vector store of type {config.type} for configuration {configuration_name}")
        
        return self.stores[configuration_name]

    def list_configurations(self) -> List[str]:
        """List all available configurations."""
        return list(self.stores.keys())

    def configuration_exists(self, configuration_name: str) -> bool:
        """Check if a configuration exists."""
        return configuration_name in self.stores

    def delete_configuration(self, configuration_name: str) -> bool:
        """Delete a configuration."""
        if configuration_name in self.stores:
            # Get the store instance before removing it
            store = self.stores[configuration_name]
            
            # For Redis, we might want to clean up the index
            if isinstance(store, RedisVectorStore):
                # Optionally implement index cleanup if needed
                pass
                
            del self.stores[configuration_name]
            return True
        return False
