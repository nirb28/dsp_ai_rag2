import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os

from app.config import RAGConfig, LLMConfig, LLMProvider, settings, process_env_vars_in_model
from app.model_schemas import Document, DocumentStatus, QueryResponse
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreManager, FAISSVectorStore
from app.services.generation_service import GenerationServiceFactory
from app.services.reranker_service import RerankerService
from app.services.query_expansion_service import QueryExpansionService
from app.services.security_service import SecurityService

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.configurations: Dict[str, RAGConfig] = {}
        self.embedding_services: Dict[str, EmbeddingService] = {}
        self.generation_services: Dict[str, Any] = {}
        self.reranker_services: Dict[str, RerankerService] = {}
        self.llm_configurations: Dict[str, LLMConfig] = {}
        self.query_expansion_service = QueryExpansionService()
        self._load_configurations()
        self._load_llm_configurations()
        # Note: MCP servers are now handled by the separate MCP server application


    def _load_configurations(self):
        """Load configurations from storage."""
        config_file = Path(settings.STORAGE_PATH) / "configurations.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    for configuration_name, config_dict in data.items():
                        # Create the RAGConfig and process environment variables
                        # Remove any existing configuration_name field from the config_dict
                        if 'configuration_name' in config_dict:
                            del config_dict['configuration_name']
                        config = RAGConfig(**config_dict)
                        config = process_env_vars_in_model(config)
                        self.configurations[configuration_name] = config
                logger.info(f"Loaded {len(self.configurations)} configurations")
            except Exception as e:
                logger.error(f"Error loading configurations: {str(e)}")
                raise

    def _save_configurations(self):
        """Save configurations to storage."""
        try:
            config_file = Path(settings.STORAGE_PATH) / "configurations.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for configuration_name, config in self.configurations.items():
                # Save configuration without adding redundant name field
                config_dict = config.dict()
                data[configuration_name] = config_dict
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Saved configurations")
        except Exception as e:
            logger.error(f"Error saving configurations: {str(e)}")

    def set_configuration(self, configuration_name: str, config: RAGConfig) -> bool:
        """Set configuration for a configuration."""
        try:
            # Process environment variables in the configuration
            config = process_env_vars_in_model(config)
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
        """Get configuration for a configuration.
        
        Args:
            configuration_name: Name of the configuration to retrieve
            
        Returns:
            The configuration if it exists with environment variables resolved
            
        Raises:
            KeyError: If the configuration does not exist
        """
        if configuration_name not in self.configurations:
            # Configuration doesn't exist
            raise KeyError(f"Configuration '{configuration_name}' not found")
        
        # Get the configuration and ensure environment variables are processed
        # This ensures any environment variables added after initial loading are processed
        config = self.configurations[configuration_name]
        return process_env_vars_in_model(config)

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
    
    def _get_security_service(self, configuration_name: str) -> Optional[SecurityService]:
        """Get security service for a configuration if security is enabled."""
        config = self.get_configuration(configuration_name)
        if config.security and config.security.enabled:
            return SecurityService(config.security)
        return None
        
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
            
    async def upload_text_content(
        self, 
        content: str,
        filename: str,
        configuration_name: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        process_immediately: bool = True
    ) -> Document:
        """Upload and optionally process raw text content without requiring a file.
        
        Args:
            content: The raw text content to process
            filename: A name to identify the document
            configuration_name: The configuration to use
            metadata: Optional metadata for the document
            process_immediately: Whether to index the document immediately
            
        Returns:
            Document object with the processed content
        """
        try:
            config = self.get_configuration(configuration_name)
            
            # Process text content into Document object
            document = self.document_processor.process_text_content(
                content,
                filename,
                configuration_name=configuration_name,
                chunking_config=config.chunking,
                metadata=metadata
            )
            
            if process_immediately:
                await self._index_document(document, configuration_name)
                document.status = DocumentStatus.INDEXED
            
            logger.info(f"Uploaded text content as document: {filename} to configuration: {configuration_name}")
            return document
            
        except Exception as e:
            logger.error(f"Error uploading text content: {str(e)}")
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
            embeddings = embedding_service.embed_texts([c.page_content for c in chunks])
            vector_store.add_documents(chunks)
            
            # Update document status
            document.status = DocumentStatus.INDEXED
            return True
        except Exception as e:
            document.status = DocumentStatus.FAILED
            document.error = str(e)
            logger.error(f"Error indexing document: {str(e)}")
            return False
    
    def validate_security_and_merge_filters(
        self, 
        configuration_name: str, 
        authorization_header: Optional[str],
        request_filter: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Validate security and merge filters from JWT claims with request filters.
        
        Args:
            configuration_name: Name of the configuration
            authorization_header: Authorization header from request
            request_filter: Original filter from the request
            
        Returns:
            Merged filters (request + JWT metadata filters)
            
        Raises:
            HTTPException: If security validation fails
        """
        security_service = self._get_security_service(configuration_name)
        
        if not security_service:
            # Security is disabled, return original filter
            return request_filter
        
        # Validate the request
        is_valid, jwt_claims = security_service.validate_request(authorization_header)
        
        if not is_valid:
            # This should not happen as validate_request raises HTTPException on failure
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
        
        # Extract metadata filters from JWT claims
        jwt_filters = security_service.extract_metadata_filters(jwt_claims)
        
        # Merge filters
        merged_filters = security_service.merge_filters(request_filter, jwt_filters)
        
        if jwt_filters:
            logger.debug(f"Applied JWT metadata filters for configuration '{configuration_name}': {jwt_filters}")
        
        return merged_filters

    async def query(
        self, 
        query: str, 
        configuration_name: str = "default",
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        context_items: Optional[List[Dict[str, Any]]] = None,
        config_override: Optional[RAGConfig] = None,
        system_prompt: Optional[str] = None,
        query_expansion: Optional[Dict[str, Any]] = None,
        filter_after_reranking: bool = True,
        filter: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """Query the RAG system with optional context injection and reranking.
    
    Args:
        query: The user query string
        configuration_name: The configuration to search in
        k: Number of results to retrieve (overrides config)
        similarity_threshold: Minimum similarity score for retrieval (overrides config)
        context_items: Optional list of additional context items to include with the retrieved documents
        config_override: Optional RAGConfig object to override the configuration settings for this query
                        (can be used to override generation endpoint, embedding endpoint, or vector store)
        filter_after_reranking: Whether to apply score threshold filtering after reranking
        filter: LangChain-style metadata filter for document retrieval
        
    Returns:
        QueryResponse with answer and sources
    """
        start_time = time.time()
        
        try:
            # Get configuration, use override if provided
            config = config_override if config_override else self.get_configuration(configuration_name)
            
            # Get the standard vector store for the configuration name
            vector_store = self._get_vector_store(configuration_name)
            
            # Use services based on overridden config if provided, otherwise use standard services
            if config_override:
                logger.info(f"Using configuration overrides for query")
                # Create services from override config
                from app.services.embedding_service import EmbeddingService
                from app.services.generation_service import GenerationServiceFactory
                from app.services.reranker_service import RerankerService
                
                embedding_service = EmbeddingService(config.embedding)
                generation_service = GenerationServiceFactory.create_service(config.generation)
                reranker_service = RerankerService(config.reranking if hasattr(config, 'reranking') else None)
            else:
                # Use standard services
                embedding_service = self._get_embedding_service(configuration_name)
                generation_service = self._get_generation_service(configuration_name)
                reranker_service = self._get_reranker_service(configuration_name)
                
            # Use config defaults if not provided
            k = k or config.retrieval_k
            similarity_threshold = similarity_threshold or config.similarity_threshold
            
            # Handle query expansion if requested
            queries_to_search = [query]  # Always include original query
            expansion_metadata = None
            if query_expansion and query_expansion.get('enabled', True):
                try:
                    llm_config_name = query_expansion.get('llm_config_name')
                    if llm_config_name:
                        llm_config = self.get_llm_configuration(llm_config_name)
                        strategy = query_expansion.get('strategy', 'fusion')
                        num_queries = query_expansion.get('num_queries', 3)
                        include_metadata = query_expansion.get('include_metadata', False)
                        
                        if include_metadata:
                            expanded_queries, expansion_metadata = await self.query_expansion_service.expand_query_with_metadata(
                                query, llm_config, strategy, num_queries
                            )
                        else:
                            expanded_queries = await self.query_expansion_service.expand_query(
                                query, llm_config, strategy, num_queries
                            )
                        
                        queries_to_search = expanded_queries
                        logger.info(f"Using {len(queries_to_search)} queries for retrieval (including original)")
                except Exception as e:
                    logger.error(f"Query expansion failed: {str(e)}. Using original query only.")
                    queries_to_search = [query]
            
            # Retrieve relevant documents using all queries
            all_results = []
            
            # Pass the freshly created embedding service to override the one in the vector store
            if hasattr(vector_store, 'embedding_service'):
                # Store the original embedding service
                original_embedding_service = vector_store.embedding_service
                # Temporarily replace with our current embedding service
                vector_store.embedding_service = embedding_service
            
            for q in queries_to_search:
                results = vector_store.similarity_search(
                    q,
                    k=k if not reranker_service.config.enabled else max(k, reranker_service.config.top_n),
                    similarity_threshold=similarity_threshold,
                    filter=filter
                )
                
                # Add query source information to results
                query_results = [(doc, score, q) for doc, score in results]
                all_results.append(query_results)
            
            # Restore the original embedding service if we changed it
            if hasattr(vector_store, 'embedding_service') and 'original_embedding_service' in locals():
                vector_store.embedding_service = original_embedding_service
            
            # Merge and deduplicate results if multiple queries were used
            if len(queries_to_search) > 1:
                merged_results = self._merge_query_results(all_results, k * 2)  # Get more for reranking
            else:
                merged_results = all_results[0] if all_results else []
            
            # Prepare context documents
            context_docs = []
            for item in merged_results:
                if len(item) == 3:  # (doc, score, query)
                    doc, score, source_query = item
                    context_docs.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': score,
                        'source_query': source_query if len(queries_to_search) > 1 else None
                    })
                else:  # (doc, score)
                    doc, score = item
                    context_docs.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': score,
                        'source_query': None
                    })
            
            # Apply reranking if enabled
            if reranker_service.config.enabled and context_docs:
                original_count = len(context_docs)
                context_docs = await reranker_service.rerank(query, context_docs, filter_after_reranking=filter_after_reranking)
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
            answer = await generation_service.generate_response(query, context_docs, system_prompt=system_prompt)
            
            processing_time = time.time() - start_time
            
            # Enhance expansion metadata with query results summary if metadata was requested
            if expansion_metadata and len(queries_to_search) > 1:
                # Create query results summary by analyzing the raw results before merging
                query_results_summary = []
                
                for i, q in enumerate(queries_to_search):
                    # Get raw results for this specific query from all_results
                    query_raw_results = all_results[i] if i < len(all_results) else []
                    
                    # Count results from this query and get top score
                    results_count = len(query_raw_results)
                    top_score = 0.0
                    
                    if query_raw_results:
                        # Extract scores from the raw results
                        if len(query_raw_results[0]) == 3:  # (doc, score, query)
                            scores = [item[1] for item in query_raw_results]
                        else:  # (doc, score)
                            scores = [item[1] for item in query_raw_results]
                        top_score = max(scores) if scores else 0.0
                    
                    query_results_summary.append({
                        "query": q,
                        "is_original": q == query,
                        "results_count": results_count,
                        "top_similarity_score": top_score
                    })
                
                expansion_metadata["query_results_summary"] = query_results_summary
                expansion_metadata["total_unique_results"] = len(context_docs)
                expansion_metadata["total_raw_results"] = sum(len(results) for results in all_results)
            
            response = QueryResponse(
                query=query,
                answer=answer,
                sources=context_docs,
                processing_time=processing_time,
                configuration_name=configuration_name,
                query_expansion_metadata=expansion_metadata
            )
            
            logger.info(f"Processed query in {processing_time:.2f}s with {len(context_docs)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def get_configuration_names(self) -> List[str]:
        """Get just the names of available configurations.
        
        Returns:
            List of configuration names as strings
        """
        return list(self.configurations.keys())

    def get_configurations(self) -> List[Dict[str, Any]]:
        """Get information about all configurations without initializing services."""
        configurations = []
        
        for configuration_name in self.configurations.keys():
            try:
                config = self.get_configuration(configuration_name)
                
                # Don't try to connect to vector store or get document count
                # Just return the configuration information
                configurations.append({
                    'configuration_name': configuration_name,  # Add the new field
                    'document_count': 0,  # Default to 0 since we're not connecting to the store
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

    def reload_configurations(self) -> bool:
        """Reload configurations from file."""
        try:
            # Clear existing configurations
            self.configurations = {}
            
            # Clear vector store manager cache
            
            # Reload configurations from file
            self._load_configurations()
            
            logger.info(f"Reloaded configurations. Found {len(self.configurations)} configurations.")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading configurations: {str(e)}")
            return False
            
    async def duplicate_configuration(
        self,
        source_configuration_name: str,
        target_configuration_name: str,
        include_documents: bool = False
    ) -> Dict[str, Any]:
        """Duplicate a configuration with option to include vector store contents.
        
        Args:
            source_configuration_name: Name of the source configuration
            target_configuration_name: Name of the target configuration
            include_documents: Whether to copy documents from source to target
            
        Returns:
            Dictionary with information about the duplication
        """
        try:
            # Check if source configuration exists
            if source_configuration_name not in self.configurations:
                raise KeyError(f"Source configuration '{source_configuration_name}' not found")
                
            # Check if target configuration already exists
            if target_configuration_name in self.configurations:
                raise ValueError(f"Target configuration '{target_configuration_name}' already exists")
                
            # Get source configuration
            source_config = self.configurations[source_configuration_name]
            
            # Create a deep copy of the configuration
            import copy
            target_config = copy.deepcopy(source_config)
            
            # Set the new configuration
            self.configurations[target_configuration_name] = target_config
            self._save_configurations()
            
            documents_copied = 0
            
            # Copy documents if requested
            if include_documents:
                # Get source vector store
                source_vector_store = self._get_vector_store(source_configuration_name)
                
                # Get target vector store (this will create it)
                target_vector_store = self._get_vector_store(target_configuration_name)
                
                # For FAISS vector store, we can copy documents directly
                if hasattr(source_vector_store, 'documents') and hasattr(target_vector_store, 'add_documents'):
                    # Get documents from source
                    documents = source_vector_store.documents
                    
                    if documents:
                        # Add documents to target
                        target_vector_store.add_documents(documents)
                        documents_copied = len(documents)
                        
            return {
                "source_configuration_name": source_configuration_name,
                "target_configuration_name": target_configuration_name,
                "config": target_config.dict(),
                "documents_copied": documents_copied,
                "message": f"Configuration '{source_configuration_name}' duplicated to '{target_configuration_name}' successfully"
            }
            
        except Exception as e:
            logger.info(f"Error duplicating configuration: {str(e)}")
            raise
    
    def _load_llm_configurations(self):
        """Load LLM configurations from storage."""
        config_file = Path(settings.STORAGE_PATH) / "llm_configurations.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    for name, config_dict in data.items():
                        config = LLMConfig(**config_dict)
                        config = process_env_vars_in_model(config)
                        self.llm_configurations[name] = config
                logger.info(f"Loaded {len(self.llm_configurations)} LLM configurations")
            except Exception as e:
                logger.error(f"Error loading LLM configurations: {str(e)}")
                # Don't raise here, just log the error
    
    def _save_llm_configurations(self):
        """Save LLM configurations to storage."""
        try:
            config_file = Path(settings.STORAGE_PATH) / "llm_configurations.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for name, config in self.llm_configurations.items():
                # Save configuration without sensitive data in logs
                config_dict = config.dict()
                # Don't log API keys
                if 'api_key' in config_dict and config_dict['api_key']:
                    config_dict['api_key'] = "***REDACTED***"
                data[name] = config.dict()  # Save actual config with API key
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Saved LLM configurations")
        except Exception as e:
            logger.error(f"Error saving LLM configurations: {str(e)}")
    
    def set_llm_configuration(self, name: str, config: LLMConfig) -> bool:
        """Set LLM configuration."""
        try:
            # Process environment variables in the configuration
            config = process_env_vars_in_model(config)
            self.llm_configurations[name] = config
            self._save_llm_configurations()
            
            logger.info(f"Set LLM configuration: {name}")
            return True
        except Exception as e:
            logger.error(f"Error setting LLM configuration: {str(e)}")
            return False
    
    def get_llm_configuration(self, name: str) -> LLMConfig:
        """Get LLM configuration by name."""
        if name not in self.llm_configurations:
            raise KeyError(f"LLM configuration '{name}' not found")
        
        # Process environment variables
        config = self.llm_configurations[name]
        return process_env_vars_in_model(config)
    
    def get_llm_configurations(self) -> List[Dict[str, Any]]:
        """Get all LLM configurations."""
        configurations = []
        for name, config in self.llm_configurations.items():
            config_dict = config.dict()
            # Don't expose API keys in list responses
            if 'api_key' in config_dict and config_dict['api_key']:
                config_dict['api_key'] = "***REDACTED***"
            configurations.append(config_dict)
        return configurations
    
    def delete_llm_configuration(self, name: str) -> bool:
        """Delete LLM configuration."""
        try:
            if name in self.llm_configurations:
                del self.llm_configurations[name]
                self._save_llm_configurations()
                logger.info(f"Deleted LLM configuration: {name}")
                return True
            else:
                logger.warning(f"LLM configuration '{name}' not found for deletion")
                return False
        except Exception as e:
            logger.error(f"Error deleting LLM configuration: {str(e)}")
            return False
    
    def reload_llm_configurations(self) -> bool:
        """Reload LLM configurations from file."""
        try:
            self.llm_configurations = {}
            self._load_llm_configurations()
            logger.info(f"Reloaded LLM configurations. Found {len(self.llm_configurations)} configs.")
            return True
        except Exception as e:
            logger.error(f"Error reloading LLM configurations: {str(e)}")
            return False

    def _merge_query_results(self, all_results: List[List], k: int) -> List:
        """Merge results from multiple queries, removing duplicates and ranking by score."""
        import logging
        logger = logging.getLogger(__name__)
        # Flatten all results
        merged = []
        seen_content = set()
        pre_merge_count = sum(len(r) for r in all_results)
        print(all_results)
        logger.debug(f"[Merge] Pre-merge results: {pre_merge_count} total documents")
        for query_results in all_results:
            for item in query_results:
                if len(item) == 3:  # (doc, score, query)
                    doc, score, source_query = item
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        merged.append((doc, score, source_query))
                else:  # (doc, score)
                    doc, score = item
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        merged.append((doc, score))
        logger.debug(f"[Merge] Deduplicated {pre_merge_count} results down to {len(merged)} unique documents.")
        # Sort by similarity score (descending)
        merged.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"[Merge] Post-sort results: {len(merged)} documents")
        # Return top k results
        logger.debug(f"[Merge] Returning top {k} results after sorting.")
        return merged[:k]

    
    async def retrieve(
        self,
        query: str,
        configuration_name: str = "default",
        k: int = 5,
        similarity_threshold: float = 0.0,
        query_expansion: Optional[Dict[str, Any]] = None,
        filter_after_reranking: bool = True,
        filter: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Retrieve documents with optional query expansion.
        
        Args:
            query: The search query
            configuration_name: Configuration to use
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
            query_expansion: Optional query expansion configuration
            filter_after_reranking: Whether to apply score threshold filtering after reranking
            filter: LangChain-style metadata filter for document retrieval
            
        Returns:
            Tuple of (documents list, expansion metadata dict or None)
        """
        try:
            # Get configuration and services
            config = self.get_configuration(configuration_name)
            vector_store = self._get_vector_store(configuration_name)
            embedding_service = self._get_embedding_service(configuration_name)
            
            # Handle query expansion if requested
            queries_to_search = [query]  # Always include original query
            expansion_metadata = None
            if query_expansion and query_expansion.get('enabled', True):
                try:
                    llm_config_name = query_expansion.get('llm_config_name')
                    if llm_config_name:
                        llm_config = self.get_llm_configuration(llm_config_name)
                        strategy = query_expansion.get('strategy', 'fusion')
                        num_queries = query_expansion.get('num_queries', 3)
                        include_metadata = query_expansion.get('include_metadata', False)
                        
                        if include_metadata:
                            expanded_queries, expansion_metadata = await self.query_expansion_service.expand_query_with_metadata(
                                query, llm_config, strategy, num_queries
                            )
                        else:
                            expanded_queries = await self.query_expansion_service.expand_query(
                                query, llm_config, strategy, num_queries
                            )
                        
                        queries_to_search = expanded_queries
                        logger.info(f"Using {len(queries_to_search)} queries for retrieval (including original)")
                except Exception as e:
                    logger.error(f"Query expansion failed: {str(e)}. Using original query only.")
                    queries_to_search = [query]
            
            # Retrieve documents using all queries
            all_results = []
            
            # Override embedding service if needed
            if hasattr(vector_store, 'embedding_service'):
                original_embedding_service = vector_store.embedding_service
                vector_store.embedding_service = embedding_service
            
            for q in queries_to_search:
                results = vector_store.similarity_search(
                    q,
                    k=k,
                    similarity_threshold=similarity_threshold,
                    filter=filter
                )
                logger.debug(f"[Retrieve] Retrieved {len(results)} results for query '{q}'. Top 3 scores: {[r[1] for r in results[:3]] if results else []}")
                # Add query source information to results
                query_results = [(doc, score, q) for doc, score in results]
                all_results.append(query_results)
            
            # Restore original embedding service
            if hasattr(vector_store, 'embedding_service') and 'original_embedding_service' in locals():
                vector_store.embedding_service = original_embedding_service
            
            # Merge and deduplicate results if multiple queries were used
            if len(queries_to_search) > 1:
                logger.debug(f"[Query] Merging and deduplicating results from {len(queries_to_search)} queries. Raw total: {sum(len(r) for r in all_results)}.")
                merged_results = self._merge_query_results(all_results, k)
                logger.debug(f"[Query] Results after merging/deduplication: {len(merged_results)}")
            else:
                merged_results = all_results[0] if all_results else []
                logger.debug(f"[Query] Single-query, skipping merge. Results: {len(merged_results)}")
            
            # Prepare response documents
            documents = []
            for item in merged_results:
                if len(item) == 3:  # (doc, score, query)
                    doc, score, source_query = item
                    documents.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': score,
                        'source_query': source_query if len(queries_to_search) > 1 else None
                    })
                else:  # (doc, score)
                    doc, score = item
                    documents.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': score,
                        'source_query': None
                    })
            
            # Enhance expansion metadata with query results summary if metadata was requested
            if expansion_metadata and len(queries_to_search) > 1:
                # Create query results summary by analyzing the raw results before merging
                query_results_summary = []
                
                for i, q in enumerate(queries_to_search):
                    # Get raw results for this specific query from all_results
                    query_raw_results = all_results[i] if i < len(all_results) else []
                    
                    # Count results from this query and get top score
                    results_count = len(query_raw_results)
                    top_score = 0.0
                    
                    if query_raw_results:
                        # Extract scores from the raw results
                        if len(query_raw_results[0]) == 3:  # (doc, score, query)
                            scores = [item[1] for item in query_raw_results]
                        else:  # (doc, score)
                            scores = [item[1] for item in query_raw_results]
                        top_score = max(scores) if scores else 0.0
                    
                    query_results_summary.append({
                        "query": q,
                        "is_original": q == query,
                        "results_count": results_count,
                        "top_similarity_score": top_score
                    })
                
                expansion_metadata["query_results_summary"] = query_results_summary
                expansion_metadata["total_unique_results"] = len(documents)
                expansion_metadata["total_raw_results"] = sum(len(results) for results in all_results)
            
            return documents, expansion_metadata
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
