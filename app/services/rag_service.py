import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os

from app.config import RAGConfig, settings
from app.models import Document, DocumentStatus, QueryResponse
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreManager, FAISSVectorStore
from app.services.generation_service import GenerationServiceFactory
from app.services.reranker_service import RerankerService
from app.services.context_service import ContextService

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.configurations: Dict[str, RAGConfig] = {}
        self.embedding_services: Dict[str, EmbeddingService] = {}
        self.generation_services: Dict[str, Any] = {}
        self.reranker_services: Dict[str, RerankerService] = {}
        self.context_services: Dict[str, ContextService] = {}
        self._load_configurations()

    def _load_configurations(self):
        """Load configurations from storage."""
        config_file = Path(settings.STORAGE_PATH) / "configurations.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    for collection_name, config_dict in data.items():
                        self.configurations[collection_name] = RAGConfig(**config_dict)
                logger.info(f"Loaded {len(self.configurations)} configurations")
            except Exception as e:
                logger.error(f"Error loading configurations: {str(e)}")

    def _save_configurations(self):
        """Save configurations to storage."""
        try:
            config_file = Path(settings.STORAGE_PATH) / "configurations.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for collection_name, config in self.configurations.items():
                data[collection_name] = config.dict()
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Saved configurations")
        except Exception as e:
            logger.error(f"Error saving configurations: {str(e)}")

    def set_configuration(self, collection_name: str, config: RAGConfig) -> bool:
        """Set configuration for a collection."""
        try:
            self.configurations[collection_name] = config
            self._save_configurations()
            
            # Clear cached services for this collection
            if collection_name in self.embedding_services:
                del self.embedding_services[collection_name]
            if collection_name in self.generation_services:
                del self.generation_services[collection_name]
            if collection_name in self.reranker_services:
                del self.reranker_services[collection_name]
            if collection_name in self.context_services:
                del self.context_services[collection_name]
            
            logger.info(f"Set configuration for collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error setting configuration: {str(e)}")
            return False

    def get_configuration(self, collection_name: str) -> RAGConfig:
        """Get configuration for a collection."""
        if collection_name not in self.configurations:
            # Use default configuration
            self.configurations[collection_name] = RAGConfig(collection_name=collection_name)
            self._save_configurations()
        
        return self.configurations[collection_name]

    def _get_embedding_service(self, collection_name: str) -> EmbeddingService:
        """Get or create embedding service for a collection."""
        if collection_name not in self.embedding_services:
            config = self.get_configuration(collection_name)
            self.embedding_services[collection_name] = EmbeddingService(config.embedding)
        
        return self.embedding_services[collection_name]

    def _get_generation_service(self, collection_name: str):
        """Get or create generation service for a collection."""
        if collection_name not in self.generation_services:
            config = self.get_configuration(collection_name)
            self.generation_services[collection_name] = GenerationServiceFactory.create_service(config.generation)
        
        return self.generation_services[collection_name]
        
    def _get_reranker_service(self, collection_name: str) -> RerankerService:
        """Get or create reranker service for a collection."""
        if collection_name not in self.reranker_services:
            config = self.get_configuration(collection_name)
            self.reranker_services[collection_name] = RerankerService(config.reranking)
        
        return self.reranker_services[collection_name]
        
    def _get_context_service(self, collection_name: str) -> ContextService:
        """Get or create context service for a collection."""
        if collection_name not in self.context_services:
            config = self.get_configuration(collection_name)
            self.context_services[collection_name] = ContextService(config.context_injection)
        
        return self.context_services[collection_name]

    def _get_vector_store(self, collection_name: str) -> FAISSVectorStore:
        """Get or create vector store for a collection."""
        config = self.get_configuration(collection_name)
        embedding_service = self._get_embedding_service(collection_name)
        
        return self.vector_store_manager.get_store(
            collection_name, 
            config.vector_store, 
            embedding_service
        )

    async def upload_document(
        self, 
        file_path: str, 
        filename: str, 
        collection_name: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        process_immediately: bool = True
    ) -> Document:
        """Upload and optionally process a document."""
        try:
            config = self.get_configuration(collection_name)
            
            # Process document
            document = self.document_processor.process_document(
                file_path, filename, collection_name, config.chunking, metadata
            )
            
            if process_immediately:
                await self._index_document(document, collection_name)
                document.status = DocumentStatus.INDEXED
            
            logger.info(f"Uploaded document: {filename} to collection: {collection_name}")
            return document
            
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise

    async def _index_document(self, document: Document, collection_name: str):
        """Index a document in the vector store."""
        try:
            config = self.get_configuration(collection_name)
            vector_store = self._get_vector_store(collection_name)
            
            # Get chunks
            chunks = self.document_processor.get_chunks(document, config.chunking)
            
            # Add to vector store
            vector_store.add_documents(chunks)
            
            logger.info(f"Indexed document: {document.filename} with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise

    async def query(
        self, 
        query: str, 
        collection_name: str = "default",
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        context_items: Optional[List[Dict[str, Any]]] = None
    ) -> QueryResponse:
        """Query the RAG system with optional context injection and reranking.
        
        Args:
            query: The user query string
            collection_name: The collection to search in
            k: Number of results to retrieve (overrides config)
            similarity_threshold: Minimum similarity score for retrieval (overrides config)
            context_items: Additional context to inject (e.g. chat history)
            
        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        
        try:
            config = self.get_configuration(collection_name)
            vector_store = self._get_vector_store(collection_name)
            generation_service = self._get_generation_service(collection_name)
            reranker_service = self._get_reranker_service(collection_name)
            context_service = self._get_context_service(collection_name)
            
            # Use config defaults if not provided
            k = k or config.retrieval_k
            similarity_threshold = similarity_threshold or config.similarity_threshold
            
            # Apply context injection if enabled
            original_query = query
            if context_service.config.enabled:
                query = context_service.inject_context(query, context_items)
                logger.info("Applied context injection to query")
            
            # Retrieve relevant documents
            results = vector_store.similarity_search(
                original_query,  # Always use original query for vector search
                k=k if not reranker_service.config.enabled else max(k, reranker_service.config.top_n),
                similarity_threshold=similarity_threshold
            )
            
            # Prepare context documents
            context_docs = []
            for doc, score in results:
                context_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score
                })
            
            # Apply reranking if enabled
            if reranker_service.config.enabled and context_docs:
                original_count = len(context_docs)
                context_docs = await reranker_service.rerank(original_query, context_docs)
                logger.info(f"Reranked documents: {original_count} → {len(context_docs)}")
                
                # Limit to original k if reranking returned more
                if len(context_docs) > k:
                    context_docs = context_docs[:k]
            
            # Generate response - use the potentially context-injected query for generation
            answer = await generation_service.generate_response(query, context_docs)
            
            processing_time = time.time() - start_time
            
            response = QueryResponse(
                query=original_query,  # Return the original query in the response
                answer=answer,
                sources=context_docs,
                processing_time=processing_time,
                collection_name=collection_name
            )
            
            logger.info(f"Processed query in {processing_time:.2f}s with {len(context_docs)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def get_collections(self) -> List[Dict[str, Any]]:
        """Get information about all collections."""
        collections = []
        
        for collection_name in self.configurations.keys():
            try:
                vector_store = self._get_vector_store(collection_name)
                doc_count = vector_store.get_document_count()
                config = self.get_configuration(collection_name)
                
                collections.append({
                    'name': collection_name,
                    'document_count': doc_count,
                    'config': config.dict()
                })
            except Exception as e:
                logger.error(f"Error getting info for collection {collection_name}: {str(e)}")
        
        return collections

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            # Remove from configurations
            if collection_name in self.configurations:
                del self.configurations[collection_name]
            
            # Clear cached services
            if collection_name in self.embedding_services:
                del self.embedding_services[collection_name]
            if collection_name in self.generation_services:
                del self.generation_services[collection_name]
            if collection_name in self.reranker_services:
                del self.reranker_services[collection_name]
            if collection_name in self.context_services:
                del self.context_services[collection_name]
            
            # Delete from vector store manager
            self.vector_store_manager.delete_collection(collection_name)
            
            self._save_configurations()
            
            logger.info(f"Deleted collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
