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

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.configurations: Dict[str, RAGConfig] = {}
        self.embedding_services: Dict[str, EmbeddingService] = {}
        self.generation_services: Dict[str, Any] = {}
        self.reranker_services: Dict[str, RerankerService] = {}
        self._load_configurations()

    def _load_configurations(self):
        """Load configurations from storage."""
        config_file = Path(settings.STORAGE_PATH) / "configurations.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    for configuration_name, config_dict in data.items():
                        self.configurations[configuration_name] = RAGConfig(**config_dict)
                logger.info(f"Loaded {len(self.configurations)} configurations")
            except Exception as e:
                logger.error(f"Error loading configurations: {str(e)}")

    def _save_configurations(self):
        """Save configurations to storage."""
        try:
            config_file = Path(settings.STORAGE_PATH) / "configurations.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for configuration_name, config in self.configurations.items():
                data[configuration_name] = config.dict()
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Saved configurations")
        except Exception as e:
            logger.error(f"Error saving configurations: {str(e)}")

    def set_configuration(self, configuration_name: str, config: RAGConfig) -> bool:
        """Set configuration for a configuration."""
        try:
            self.configurations[configuration_name] = config
            self._save_configurations()
            
            # Clear cached services for this configuration
            if configuration_name in self.embedding_services:
                del self.embedding_services[configuration_name]
            if configuration_name in self.generation_services:
                del self.generation_services[configuration_name]
            if configuration_name in self.reranker_services:
                del self.reranker_services[configuration_name]
            
            logger.info(f"Set configuration for configuration: {configuration_name}")
            return True
        except Exception as e:
            logger.error(f"Error setting configuration: {str(e)}")
            return False

    def get_configuration(self, configuration_name: str) -> RAGConfig:
        """Get configuration for a configuration."""
        if configuration_name not in self.configurations:
            # Use default configuration
            self.configurations[configuration_name] = RAGConfig(configuration_name=configuration_name)
            self._save_configurations()
        
        return self.configurations[configuration_name]

    def _get_embedding_service(self, configuration_name: str) -> EmbeddingService:
        """Get or create embedding service for a configuration."""
        if configuration_name not in self.embedding_services:
            config = self.get_configuration(configuration_name)
            self.embedding_services[configuration_name] = EmbeddingService(config.embedding)
        
        return self.embedding_services[configuration_name]

    def _get_generation_service(self, configuration_name: str) -> Any:
        """Get or create generation service for a configuration."""
        if configuration_name not in self.generation_services:
            config = self.get_configuration(configuration_name)
            self.generation_services[configuration_name] = GenerationServiceFactory.create_service(config.generation)
        
        return self.generation_services[configuration_name]
        
    def _get_reranker_service(self, configuration_name: str) -> RerankerService:
        """Get or create reranker service for a configuration."""
        if configuration_name not in self.reranker_services:
            config = self.get_configuration(configuration_name)
            self.reranker_services[configuration_name] = RerankerService(config.reranking)
        
        return self.reranker_services[configuration_name]
        
    # _get_context_service method removed

    def _get_vector_store(self, configuration_name: str):
        """Get or create vector store for a configuration."""
        config = self.get_configuration(configuration_name)
        vector_store = self.vector_store_manager.get_vector_store(
            configuration_name, 
            config.vector_store, 
            config.embedding.dict()
        )
        return vector_store

    async def upload_document(
        self, 
        file_path: str, 
        filename: str, 
        configuration_name: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        process_immediately: bool = True
    ) -> Document:
        """Upload and optionally process a document."""
        try:
            config = self.get_configuration(configuration_name)
            
            # Process document into Document object
            document = self.document_processor.process_document(
                file_path,
                filename,
                configuration_name=configuration_name,
                chunking_config=config.chunking,
                metadata=metadata
            )
            
            if process_immediately:
                await self._index_document(document, configuration_name)
                document.status = DocumentStatus.INDEXED
            
            logger.info(f"Uploaded document: {filename} to configuration: {configuration_name}")
            return document
            
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise

    async def _index_document(self, document: Document, configuration_name: str):
        """Index a document in the vector store."""
        try:
            # Process document text into chunks
            config = self.get_configuration(configuration_name)
            chunks = self.document_processor.get_chunks(document, config.chunking)
            
            # Get services
            embedding_service = self._get_embedding_service(configuration_name)
            vector_store = self._get_vector_store(configuration_name)
            
            # Generate embeddings and add to vector store
            embeddings = await embedding_service.embed_texts([c.text for c in chunks])
            vector_store.add_chunks(chunks, embeddings)
            
            # Update document status
            document.status = DocumentStatus.PROCESSED
            return True
        except Exception as e:
            document.status = DocumentStatus.FAILED
            document.error = str(e)
            logger.error(f"Error indexing document: {str(e)}")
            return False

    async def query(
        self, 
        query: str, 
        configuration_name: str = "default",
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        context_items: Optional[List[Dict[str, Any]]] = None
    ) -> QueryResponse:
        """Query the RAG system with optional context injection and reranking.
        
        Args:
            query: The user query string
            configuration_name: The configuration to search in
            k: Number of results to retrieve (overrides config)
            similarity_threshold: Minimum similarity score for retrieval (overrides config)
            context_items: Optional list of additional context items to include with the retrieved documents
            
        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        
        try:
            # Get configuration
            config = self.get_configuration(configuration_name)
            vector_store = self._get_vector_store(configuration_name)
            embedding_service = self._get_embedding_service(configuration_name)
            generation_service = self._get_generation_service(configuration_name)
            reranker_service = self._get_reranker_service(configuration_name)
            
            # Use config defaults if not provided
            k = k or config.retrieval_k
            similarity_threshold = similarity_threshold or config.similarity_threshold
            
            # Context injection functionality removed
            
            # Retrieve relevant documents
            results = vector_store.similarity_search(
                query,
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
                context_docs = await reranker_service.rerank(query, context_docs)
                logger.info(f"Reranked documents: {original_count} → {len(context_docs)}")
                
                # Limit to original k if reranking returned more
                if len(context_docs) > k:
                    context_docs = context_docs[:k]
                    
            # Add additional context items if provided
            if context_items:
                logger.info(f"Adding {len(context_items)} additional context items")
                for item in context_items:
                    # Ensure consistency with retrieved docs format
                    if 'similarity_score' not in item:
                        item['similarity_score'] = 1.0  # Give highest priority to manually injected content
                    context_docs.append(item)
            
            # Generate response - use the potentially context-injected query for generation
            answer = await generation_service.generate_response(query, context_docs)
            
            processing_time = time.time() - start_time
            
            response = QueryResponse(
                query=query,
                answer=answer,
                sources=context_docs,
                processing_time=processing_time,
                configuration_name=configuration_name
            )
            
            logger.info(f"Processed query in {processing_time:.2f}s with {len(context_docs)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def get_configurations(self) -> List[Dict[str, Any]]:
        """Get information about all configurations."""
        configurations = []
        
        for configuration_name in self.configurations.keys():
            try:
                vector_store = self._get_vector_store(configuration_name)
                doc_count = vector_store.get_document_count()
                config = self.get_configuration(configuration_name)
                
                configurations.append({
                    'name': configuration_name,
                    'document_count': doc_count,
                    'config': config.dict()
                })
            except Exception as e:
                logger.error(f"Error getting info for configuration {configuration_name}: {str(e)}")
        
        return configurations

    def delete_configuration(self, configuration_name: str) -> bool:
        """Delete a configuration."""
        try:
            # Remove from configurations
            if configuration_name in self.configurations:
                del self.configurations[configuration_name]
            
            # Clear cached services
            if configuration_name in self.embedding_services:
                del self.embedding_services[configuration_name]
            if configuration_name in self.generation_services:
                del self.generation_services[configuration_name]
            if configuration_name in self.reranker_services:
                del self.reranker_services[configuration_name]
            
            # Delete from vector store manager
            self.vector_store_manager.delete_configuration(configuration_name)
            
            self._save_configurations()
            
            logger.info(f"Deleted configuration: {configuration_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting configuration: {str(e)}")
            return False
