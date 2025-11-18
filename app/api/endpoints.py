import os
import tempfile
import time
import json
import uuid
import numpy as np
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query, Header
from fastapi.responses import JSONResponse
from fastapi import Body
import logging
from langchain.docstore.document import Document as LangchainDocument
# Import the documentation router
from app.api.documentation import router as documentation_router

from app.model_schemas.base_models import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    RetrieveRequest, 
    RetrieveResponse, 
    QueryRequest, 
    QueryResponse, 
    ConfigurationNamesResponse,
    ConfigurationRequest,
    ConfigurationResponse,
    ConfigurationsResponse,
    ConfigurationInfo,
    DuplicateConfigurationRequest,
    DuplicateConfigurationResponse,
    TextDocumentsUploadRequest,
    TextDocumentsUploadResponse,
    LLMConfigRequest,
    LLMConfigResponse,
    LLMConfigListResponse,
    DeleteConfigurationResponse,
    HealthResponse,
    ErrorResponse
)
from app.config import RAGConfig, LLMConfig, LLMProvider, settings
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreManager, FAISSVectorStore, RedisVectorStore

logger = logging.getLogger(__name__)

# Add method to FAISSVectorStore to get all documents
if not hasattr(FAISSVectorStore, 'get_all_documents'):
    def get_all_documents(self, limit=10):
        """Get all documents in the vector store.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of documents
        """
        if not hasattr(self, 'documents') or not self.documents:
            return []
            
        return self.documents[:limit]
        
    FAISSVectorStore.get_all_documents = get_all_documents

# Add method to RedisVectorStore to get all documents
if not hasattr(RedisVectorStore, 'get_all_documents'):
    def get_all_documents(self, limit: int = 10) -> List[LangchainDocument]:
      try:
        cursor = 0
        documents = []
        #prefix = f"doc:{self.index_name}:*"  # Use configuration-specific prefix
        prefix = f"doc:{{{self.index_name}}}:*"
        while True:
            cursor, keys = self.redis_client.scan(cursor, match=prefix, count=limit)
            for key in keys:
                doc_data = self.redis_client.hgetall(key)
                if not doc_data:
                    continue
                content = doc_data.get(b"content", b"").decode("utf-8")
                metadata = json.loads(doc_data.get(b"metadata", b"{}").decode("utf-8"))
                langchain_doc = LangchainDocument(page_content=content, metadata=metadata)
                documents.append(langchain_doc)
                if len(documents) >= limit:               
                    return documents
            if cursor == 0:
                break
        return documents
      except Exception as e:
        logger.error(f"Error retrieving documents from Redis: {str(e)}")
        raise
    RedisVectorStore.get_all_documents = get_all_documents

router = APIRouter()

# Include documentation router
router.include_router(documentation_router, prefix="/documentation")

# Global RAG service instance - will be set by main.py
rag_service = None

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services={
            "rag_service": "running",
            "vector_store": "faiss",
            "embedding": "sentence-transformers",
            "generation": "groq"
        }
    )

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    configuration_name: str = Form(default="default"),
    metadata: Optional[str] = Form(default=None),
    process_immediately: bool = Form(default=True)
):
    """Upload a document to a collection."""
    temp_file_path = None
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower().lstrip('.')
        if file_extension not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed types: {settings.ALLOWED_FILE_TYPES}"
            )
        
        # Validate file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                import json
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process document
        document = await rag_service.upload_document(
            file_path=temp_file_path,
            filename=file.filename,
            configuration_name=configuration_name,
            metadata=doc_metadata,
            process_immediately=process_immediately
        )
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            status=document.status,
            message="Document uploaded successfully to configuration '{configuration_name}",
            configuration_name=configuration_name
        )
            
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        raise
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Error removing temporary file: {str(e)}")


@router.post("/upload/text", response_model=TextDocumentsUploadResponse)
async def upload_text_documents(request: TextDocumentsUploadRequest):
    """Upload multiple text documents without requiring file attachments.
    
    This endpoint allows uploading one or more text documents directly as JSON,
    without the need to create and upload actual files.
    
    Args:
        request: The request containing text documents and configuration options
        
    Returns:
        Information about the uploaded documents
    """
    try:
        configuration_name = request.configuration_name
        process_immediately = request.process_immediately
        
        # Validate configuration exists
        try:
            rag_service.get_configuration(configuration_name)
        except KeyError:
            raise HTTPException(
                status_code=404, 
                detail=f"Configuration '{configuration_name}' not found"
            )
        
        # Process each document
        results = []
        for doc in request.documents:
            try:
                # Generate a unique filename if none is provided
                filename = doc.filename
                if not filename:
                    filename = f"text_doc_{str(uuid.uuid4())[:8]}.txt"
                
                # Process text content
                document = await rag_service.upload_text_content(
                    content=doc.content,
                    filename=filename,
                    configuration_name=configuration_name,
                    metadata=doc.metadata,
                    process_immediately=process_immediately
                )
                
                results.append(DocumentUploadResponse(
                    document_id=document.id,
                    filename=document.filename,
                    status=document.status,
                    message="Document uploaded successfully",
                    configuration_name=configuration_name
                ))
                
            except Exception as e:
                logger.error(f"Error processing text document {filename}: {str(e)}")
                results.append(DocumentUploadResponse(
                    document_id="",
                    filename=filename,
                    status=DocumentStatus.FAILED,
                    message=f"Error: {str(e)}",
                    configuration_name=configuration_name
                ))
        
        return TextDocumentsUploadResponse(
            documents=results,
            total_count=len(results),
            configuration_name=configuration_name,
            message=f"Processed {len(results)} text documents"
        )
        
    except Exception as e:
        logger.error(f"Error in text upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text documents: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Query documents in a configuration.
    
    Uses reranking if configured for the configuration.
    Supports JWT Bearer token authentication if security is enabled.
    """
    try:
        # Validate security and merge filters
        merged_filter = rag_service.validate_security_and_merge_filters(
            configuration_name=request.configuration_name,
            authorization_header=authorization,
            request_filter=request.filter
        )
        
        # Update request filter with merged filter (includes JWT metadata filters)
        request.filter = merged_filter
        
        # Get original configuration
        config = rag_service.get_configuration(request.configuration_name)
        temp_config = None
        
        # Apply config overrides if provided
        system_prompt = None
        if request.config:
            # Extract system_prompt if present in config overrides
            if 'system_prompt' in request.config:
                system_prompt = request.config.pop('system_prompt')
                logger.info(f"Using custom system prompt for query request")
                
            # Create a temporary configuration with overrides
            temp_config_dict = config.dict()
            temp_config_dict.update(request.config)
            try:
                from app.config import RAGConfig
                temp_config = RAGConfig(**temp_config_dict)
                # Use temporary config for this request
                config = temp_config
                logger.info(f"Using configuration overrides for query request")
            except Exception as e:
                logger.warning(f"Invalid config override: {str(e)}")
                # If the config override is invalid, restore the system_prompt to the request.config for debugging
                if system_prompt is not None:
                    request.config['system_prompt'] = system_prompt
        
        # Check if reranking and context injection are enabled for this configuration
        reranking_enabled = hasattr(config, 'reranking') and config.reranking and config.reranking.enabled
        context_enabled = hasattr(config, 'context_injection') and config.context_injection and config.context_injection.enabled
        
        if context_enabled:
            logger.info(f"Context injection enabled for configuration '{request.configuration_name}'")
        if reranking_enabled:
            logger.info(f"Reranking enabled for configuration '{request.configuration_name}' using model: {config.reranking.model}")
        
        # Convert context items to list of dictionaries if provided
        context_items = None
        if request.context_items:
            context_items = [item.dict() for item in request.context_items]
            logger.info(f"Using {len(context_items)} context items for query")
        
        # Prepare query expansion if provided
        query_expansion_dict = None
        if request.query_expansion:
            query_expansion_dict = {
                'enabled': request.query_expansion.enabled,
                'strategy': request.query_expansion.strategy,
                'llm_config_name': request.query_expansion.llm_config_name,
                'num_queries': request.query_expansion.num_queries,
                'include_metadata': request.query_expansion.include_metadata
            }
        
        response = await rag_service.query(
            query=request.query,
            configuration_name=request.configuration_name,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
            context_items=request.context_items,
            config_override=config if temp_config else None,
            system_prompt=system_prompt,
            query_expansion=query_expansion_dict,
            filter_after_reranking=request.filter_after_reranking,
            filter=request.filter
        )
        
        # Filter metadata if requested
        if not request.include_metadata:
            for source in response.sources:
                source['metadata'] = {}
        
        # Log debug information if requested
        if request.debug:
            from app.services.debug_utils import write_debug_log
            
            # Prepare request and response payloads
            request_dict = request.dict()
            # Handle password and sensitive fields in request
            if request_dict.get('config') and request_dict['config'].get('api_key'):
                request_dict['config']['api_key'] = "***REDACTED***"
            
            # Prepare response payload
            response_dict = response.dict()
            
            # Write debug log
            log_path = write_debug_log(
                category="query",
                request_payload=request_dict,
                response_payload=response_dict,
                configuration_name=request.configuration_name
            )
            
            logger.info(f"Debug log written to {log_path}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/configurations", response_model=ConfigurationResponse)
async def add_configuration(request: ConfigurationRequest):
    """Add or update a configuration with custom settings."""
    try:
        # Validate and create RAG config
        config = RAGConfig(**request.config)
        
        # Set configuration
        success = rag_service.set_configuration(request.configuration_name, config)
        
        if success:
            return ConfigurationResponse(
                configuration_name=request.configuration_name,
                config=config.dict(),
                message=f"Configuration '{request.configuration_name}' created/updated successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
            
    except Exception as e:
        logger.error(f"Error creating/updating configuration: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")

@router.get("/configurations/{configuration_name}")
async def get_configuration(configuration_name: str):
    """Get a specific configuration."""
    try:
        config = rag_service.get_configuration(configuration_name)
        return {
            "configuration_name": configuration_name,
            "config": config.dict()
        }
    except KeyError:
        # Return 404 when configuration doesn't exist
        raise HTTPException(status_code=404, detail=f"Configuration '{configuration_name}' not found")
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/debug/configurations")
async def debug_configurations():
    """Debug endpoint to list all configurations and their details"""
    try:
        # Get list of configurations from RAG service
        rag_configs = rag_service.get_configurations()
        
        # Get configurations known to the vector store manager
        vector_store_configs = rag_service.vector_store_manager.list_configurations()
        
        # Return detailed info
        return {
            "rag_configs": rag_configs,
            "vector_store_configs": vector_store_configs,
            "configuration_names": list(rag_service.configurations.keys()),
            "postman_test_exists": rag_service.vector_store_manager.configuration_exists("postman_test"),
            "postman_test_in_rag_configs": "postman_test" in rag_service.configurations,
            "stores_keys": list(rag_service.vector_store_manager.stores.keys())
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return {"error": str(e)}

@router.get("/configurations", response_model=Union[ConfigurationsResponse, ConfigurationNamesResponse])
async def list_configurations(names_only: bool = False):
    """List all configurations and their information.
    
    Args:
        names_only: If True, returns only the configuration names without their details
    """
    try:
        if names_only:
            # Get only configuration names
            names = rag_service.get_configuration_names()
            return ConfigurationNamesResponse(
                names=names,
                total_count=len(names)
            )
        else:
            # Get full configuration details
            configurations_data = rag_service.get_configurations()
            
            configurations = []
            for data in configurations_data:
                configurations.append(ConfigurationInfo(
                    configuration_name=data['configuration_name'],
                    document_count=data['document_count'],
                    created_at=data.get('created_at'),
                    last_updated=data.get('last_updated'),
                    config=data['config']
                ))
            
            return ConfigurationsResponse(
                configurations=configurations,
                total_count=len(configurations)
            )
        
    except Exception as e:
        logger.error(f"Error listing configurations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/configurations/{configuration_name}", response_model=DeleteConfigurationResponse)
async def delete_configuration(configuration_name: str):
    """Delete a configuration."""
    try:
        success = rag_service.delete_configuration(configuration_name)
        if success:
            return DeleteConfigurationResponse(
                success=True,
                message=f"Configuration '{configuration_name}' deleted successfully",
                configuration_name=configuration_name
            )
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete configuration '{configuration_name}'")
    except Exception as e:
        logger.error(f"Error deleting configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configurations/duplicate", response_model=DuplicateConfigurationResponse)
async def duplicate_configuration(request: DuplicateConfigurationRequest):
    """Duplicate a configuration with option to include vector store contents.
    
    This endpoint allows creating a copy of an existing configuration with a new name.
    Optionally, it can also copy all documents from the source configuration's vector store.
    
    Args:
        request: The request containing source and target configuration names and options
        
    Returns:
        Information about the duplicated configuration
    """
    try:
        result = await rag_service.duplicate_configuration(
            source_configuration_name=request.source_configuration_name,
            target_configuration_name=request.target_configuration_name,
            include_documents=request.include_documents
        )
        
        return DuplicateConfigurationResponse(
            source_configuration_name=result["source_configuration_name"],
            target_configuration_name=result["target_configuration_name"],
            config=result["config"],
            documents_copied=result["documents_copied"],
            message=result["message"]
        )
        
    except KeyError as e:
        logger.error(f"Configuration not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Invalid configuration duplication request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error duplicating configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error duplicating configuration: {str(e)}")

@router.get("/debug/{configuration_name}")
async def debug_configuration(configuration_name: str, limit: int = 10, show_vectors: bool = False, show_text: bool = True):
    """Debug endpoint to inspect the vectors and chunks in a configuration.
    
    Args:
        configuration_name: Name of the configuration to inspect
        limit: Maximum number of chunks to return
        show_vectors: Whether to include embedding vectors in the response
        show_text: Whether to include the full text of chunks in the response
        
    Returns:
        Dictionary containing chunks and their metadata from the configuration
    """
    try:
        # Check if configuration exists
        if configuration_name not in rag_service.configurations:
            raise HTTPException(status_code=404, detail=f"Configuration '{configuration_name}' not found")
        
        # Get the vector store for this configuration
        vector_store = rag_service._get_vector_store(configuration_name)
        
        # Get document count
        document_count = vector_store.get_document_count()
        
        if document_count == 0:
            return {
                "configuration_name": configuration_name,
                "chunks": [],
                "total_count": 0,
                "message": "Configuration is empty. No chunks found."
            }
        
        # Get sample chunks
        chunks = vector_store.get_all_documents(limit=limit)
        
        # Try to access index vectors directly if show_vectors is True
        vectors = None
        vector_dimensions = 0
        if show_vectors and hasattr(vector_store, 'index') and vector_store.index is not None:
            try:
                # Get the first few vectors directly from FAISS
                limited_index = min(limit, vector_store.index.ntotal)
                if limited_index > 0:
                    vectors = [vector_store.index.reconstruct(i).tolist() for i in range(limited_index)]
                    if vectors and len(vectors) > 0:
                        vector_dimensions = len(vectors[0])
            except Exception as e:
                logger.warning(f"Could not extract vectors from FAISS index: {e}")
        
        # Prepare response
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "metadata": chunk.metadata,
                "text_length": len(chunk.page_content),
                "text_preview": chunk.page_content[:100] + '...' if len(chunk.page_content) > 100 else chunk.page_content
            }
            
            # Include full text only if show_text is True
            if show_text:
                chunk_data["text"] = chunk.page_content
            
            # Add embedding from our direct extraction if available
            if show_vectors and vectors and i < len(vectors):
                chunk_data["embedding"] = {
                    "dimensions": vector_dimensions,
                    "preview": vectors[i][:5] if vector_dimensions > 5 else vectors[i]
                }
            
            chunks_data.append(chunk_data)
        
        return {
            "configuration_name": configuration_name,
            "chunks": chunks_data,
            "total_count": document_count,
            "shown_count": len(chunks_data),
            "has_embeddings": vectors is not None,
            "embedding_dimensions": vector_dimensions,
            "config": rag_service.configurations[configuration_name].dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error debugging configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/retrieve", response_model=RetrieveResponse, 
          description="Retrieve relevant documents from one or more configurations with optional fusion",
          responses={
              200: {
                  "description": "Successfully retrieved documents",
                  "content": {
                      "application/json": {
                          "examples": {
                              "single_config": {
                                  "summary": "Basic retrieval from single configuration",
                                  "value": {
                                      "query": "What is RAG?",
                                      "documents": [
                                          {
                                              "content": "RAG (Retrieval Augmented Generation) is a technique...",
                                              "similarity_score": 0.92,
                                              "metadata": {"source": "introduction.pdf", "page": 1}
                                          }
                                      ],
                                      "processing_time": 0.456,
                                      "configuration_name": "default",
                                      "total_found": 1
                                  }
                              },
                              "multi_config": {
                                  "summary": "Retrieval from multiple configurations with RRF fusion",
                                  "value": {
                                      "query": "What is RAG?",
                                      "documents": [
                                          {
                                              "content": "RAG (Retrieval Augmented Generation) is a technique...",
                                              "similarity_score": 0.92,
                                              "source_configuration": "research_papers",
                                              "rrf_score": 0.0151,
                                              "metadata": {"source": "introduction.pdf", "page": 1}
                                          },
                                          {
                                              "content": "Retrieval Augmented Generation combines search with LLMs...",
                                              "similarity_score": 0.85,
                                              "source_configuration": "knowledge_base",
                                              "rrf_score": 0.0143,
                                              "metadata": {"source": "llm_techniques.md", "section": "RAG"}
                                          }
                                      ],
                                      "processing_time": 0.789,
                                      "configuration_names": ["research_papers", "knowledge_base"],
                                      "total_found": 2,
                                      "fusion_method": "rrf"
                                  }
                              }
                          }
                      }
                  }
              },
              404: {"description": "Configuration not found"},
              500: {"description": "Internal server error"}
          }
)

async def retrieve_documents(
    request: RetrieveRequest = Body(
        ...,
        examples={
            "FullFeatureRequest": {
                "summary": "Full feature example with multi-config, fusion, reranking, and query expansion",
                "description": "Retrieves documents using multiple configurations, RRF fusion, reranking, and query expansion with metadata.",
                "value": {
                    "query": "What is Computer Vision?",
                    "configuration_names": ["batch_rl-docs_test", "batch_ml_ai_basics_test"],
                    "fusion_method": "rrf",
                    "k": 10,
                    "include_metadata": False,
                    "similarity_threshold": 0.1,
                    "use_reranking": True,
                    "filter_after_reranking": False,
                    "include_vectors": False,
                    "query_expansion": {
                        "enabled": True,
                        "strategy": "multi_query",
                        "llm_config_name": "nvidia-llama3-8b",
                        "num_queries": 4,
                        "include_metadata": True
                    }
                }
            }
        }
    ),
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Retrieve relevant documents from a configuration without generating a response.
    
    This endpoint allows direct access to the vector retrieval functionality without LLM generation.
    It supports retrieving from multiple vector stores and combining results using fusion methods.
    Supports JWT Bearer token authentication if security is enabled.
    
    Args:
        request: The retrieval request containing query and options
        authorization: Authorization header for security validation
    
    Returns:
        Documents retrieved from the vector store(s), optionally reranked or fused
    """
    try:
        start_time = time.time()
        
        # Debug logging for retrieve request
        if request.debug:
            logger.debug("="*80)
            logger.debug("RETRIEVE REQUEST DEBUG")
            logger.debug("="*80)
            logger.debug(f"Query: {request.query}")
            logger.debug(f"Configuration Name: {request.configuration_name}")
            logger.debug(f"Configuration Names (multi): {request.configuration_names}")
            logger.debug(f"K: {request.k}")
            logger.debug(f"Similarity Threshold: {request.similarity_threshold}")
            logger.debug(f"Filter: {request.filter}")
            logger.debug(f"Filter After Reranking: {request.filter_after_reranking}")
            logger.debug(f"Use Reranking: {request.use_reranking}")
            logger.debug(f"Fusion Method: {request.fusion_method}")
            logger.debug(f"Include Metadata: {request.include_metadata}")
            logger.debug(f"Include Vectors: {request.include_vectors}")
            logger.debug(f"Query Expansion: {request.query_expansion}")
            logger.debug(f"Config Overrides: {request.config}")
            logger.debug("="*80)
        
        # Determine configuration names for security validation
        config_names = []
        if request.configuration_names and len(request.configuration_names) > 0:
            config_names = request.configuration_names
        else:
            config_names = [request.configuration_name]
        
        # Validate security for the primary configuration and merge filters
        # For multi-config requests, we use the first configuration's security settings
        primary_config_name = config_names[0]
        merged_filter = rag_service.validate_security_and_merge_filters(
            configuration_name=primary_config_name,
            authorization_header=authorization,
            request_filter=request.filter
        )
        
        # Update request filter with merged filter (includes JWT metadata filters)
        request.filter = merged_filter
        
        # Check for multiple configurations
        multi_config = len(config_names) > 1
        
        if request.configuration_names and len(request.configuration_names) > 0:
            # Multiple configurations specified
            config_names = request.configuration_names
            multi_config = True
        else:
            # Single configuration
            config_names = [request.configuration_name]
        
        # Validate that all requested configurations exist
        for config_name in config_names:
            # Check if configuration exists in RAG service but not in vector store manager
            if config_name in rag_service.configurations and not rag_service.vector_store_manager.configuration_exists(config_name):
                logger.info(f"Configuration '{config_name}' exists in RAG service but not in vector store manager. Initializing vector store.")
                # Initialize the vector store by accessing it
                try:
                    # This will create the vector store if it doesn't exist
                    rag_service._get_vector_store(config_name)
                except Exception as e:
                    logger.error(f"Error initializing vector store for {config_name}: {str(e)}")
            
            # Validate configuration exists in vector store manager
            if not rag_service.vector_store_manager.configuration_exists(config_name):
                raise HTTPException(
                    status_code=404, 
                    detail=f"Configuration '{config_name}' not found"
                )
        
        # Prepare to store results from each configuration
        all_results = []
        used_configs = []
        all_expansion_metadata = []
        
        # Prepare query expansion if provided
        query_expansion_dict = None
        if request.query_expansion:
            query_expansion_dict = {
                'enabled': request.query_expansion.enabled,
                'strategy': request.query_expansion.strategy,
                'llm_config_name': request.query_expansion.llm_config_name,
                'num_queries': request.query_expansion.num_queries,
                'include_metadata': request.query_expansion.include_metadata
            }
        
        # Process each configuration
        for config_name in config_names:
            # Get configuration, with optional overrides from the request
            config = rag_service.get_configuration(config_name)
            temp_config = None
            
            if request.config:
                # Create a temporary configuration with overrides
                temp_config_dict = config.dict()
                temp_config_dict.update(request.config)
                try:
                    temp_config = RAGConfig(**temp_config_dict)
                    # Use temporary config for this request
                    config = temp_config
                except Exception as e:
                    logger.warning(f"Invalid config override: {str(e)}")
            
            # Get query parameters
            k = request.k
            similarity_threshold = request.similarity_threshold if request.similarity_threshold is not None else config.similarity_threshold
            
            # Use the new retrieve method with query expansion support
            config_documents, config_expansion_metadata = await rag_service.retrieve(
                query=request.query,
                configuration_name=config_name,
                k=k,
                similarity_threshold=similarity_threshold,
                query_expansion=query_expansion_dict,
                filter_after_reranking=request.filter_after_reranking,
                filter=request.filter
            )
            
            # Add source configuration to each document
            for document in config_documents:
                document['source_configuration'] = config_name
                
                # Filter metadata if not requested
                if not request.include_metadata and 'metadata' in document:
                    document['metadata'] = {}
            
            # Store results from this configuration
            if config_documents:
                all_results.append(config_documents)
                used_configs.append(config_name)
                
                # Store expansion metadata if available
                if config_expansion_metadata:
                    config_expansion_metadata['configuration_name'] = config_name
                    all_expansion_metadata.append(config_expansion_metadata)
        
        # Combine results if multiple configurations
        documents = []
        fusion_method_used = None
        
        if multi_config and len(all_results) > 1:
            # Import fusion methods
            from app.services.utils import reciprocal_rank_fusion, simple_fusion
            
            # Determine fusion method
            fusion_method = request.fusion_method or "rrf"
            fusion_method_used = fusion_method
            
            if fusion_method == "rrf":
                # Use Reciprocal Rank Fusion
                documents = reciprocal_rank_fusion(
                    all_results, 
                    k=request.rrf_k_constant,
                    key_field='content'
                )
            elif fusion_method == "simple":
                # Use simple score averaging
                documents = simple_fusion(
                    all_results,
                    key_field='content'
                )
            else:
                # Default: just concatenate and sort by score
                for result_list in all_results:
                    documents.extend(result_list)
                documents = sorted(documents, key=lambda x: x['similarity_score'], reverse=True)
                fusion_method_used = "concatenate"
        elif len(all_results) == 1:
            # Single configuration results
            documents = all_results[0]
        
        # Apply reranking if requested
        if request.use_reranking and documents:
            # Get reranker from the first configuration (for multi-config) or the only configuration
            primary_config = used_configs[0] if used_configs else request.configuration_name
            reranker_service = rag_service._get_reranker_service(primary_config)
            
            if reranker_service and reranker_service.config.enabled:
                documents = await reranker_service.rerank(request.query, documents, filter_after_reranking=request.filter_after_reranking)
        
        # Include embedding vectors if requested
        if request.include_vectors and documents:
            try:
                # Use embedding service from the first configuration
                primary_config = used_configs[0] if used_configs else request.configuration_name
                embedding_service = rag_service._get_embedding_service(primary_config)
                
                # Get the embeddings for the document contents
                texts = [doc['content'] for doc in documents]
                embeddings = embedding_service.embed_texts(texts)
                
                # Add embeddings to documents
                for i, doc in enumerate(documents):
                    if i < len(embeddings):
                        doc['vector'] = embeddings[i]
            except Exception as e:
                logger.warning(f"Failed to include vectors: {str(e)}")
        
        # Limit to top K if needed (in case fusion returned more)
        if len(documents) > request.k:
            documents = documents[:request.k]
            
        processing_time = time.time() - start_time
        
        # Determine which configuration name to return
        result_config_name = None
        if not multi_config:
            result_config_name = request.configuration_name
        
        # Prepare combined expansion metadata
        combined_expansion_metadata = None
        if all_expansion_metadata:
            if len(all_expansion_metadata) == 1:
                # Single configuration metadata
                combined_expansion_metadata = all_expansion_metadata[0]
            else:
                # Multiple configurations metadata
                combined_expansion_metadata = {
                    "configurations": all_expansion_metadata,
                    "total_configurations": len(all_expansion_metadata)
                }
        
        response = RetrieveResponse(
            query=request.query,
            documents=documents,
            processing_time=processing_time,
            configuration_name=result_config_name,
            configuration_names=used_configs if multi_config else None,
            total_found=len(documents),
            fusion_method=fusion_method_used,
            query_expansion_metadata=combined_expansion_metadata
        )
        
        # Log debug information if requested
        if request.debug:
            from app.services.debug_utils import write_debug_log
            
            # Prepare request and response payloads
            request_dict = request.dict()
            # Handle password and sensitive fields in request
            if request_dict.get('config') and request_dict['config'].get('api_key'):
                request_dict['config']['api_key'] = "***REDACTED***"
            # Redact API key in query_expansion if present
            if request_dict.get('query_expansion') and request_dict['query_expansion'].get('llm_config_name'):
                # We don't have direct access to the API key here, just note that it's handled securely
                pass
            
            # Prepare response payload
            response_dict = response.dict()
            
            # Get configuration name for logging
            config_name = result_config_name
            if multi_config and used_configs:
                config_name = "multi_" + "_".join(used_configs[:2])  # Use first two configs in name if multi
                if len(used_configs) > 2:
                    config_name += f"_plus{len(used_configs)-2}more"
            
            # Write debug log
            log_path = write_debug_log(
                category="retrieve",
                request_payload=request_dict,
                response_payload=response_dict,
                configuration_name=config_name
            )
            
            logger.info(f"Debug log written to {log_path}")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# LLM Configuration Management Endpoints

@router.post("/llm-configs/reload")
async def reload_llm_configs():
    """Reload LLM configurations from file."""
    success = rag_service.reload_llm_configurations()
    if success:
        return {"status": "success", "message": "LLM configurations reloaded."}
    else:
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to reload LLM configurations."})


@router.post("/llm-configs", response_model=LLMConfigResponse)
async def create_llm_configuration(request: LLMConfigRequest):
    """Create or update an LLM configuration for query expansion."""
    try:
        # Convert provider string to enum
        try:
            provider = LLMProvider(request.provider)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {request.provider}. Must be one of: {', '.join([p.value for p in LLMProvider])}"
            )
        
        # Create LLM configuration
        llm_config = LLMConfig(
            name=request.name,
            provider=provider,
            model=request.model,
            endpoint=request.endpoint,
            api_key=request.api_key,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            timeout=request.timeout
        )
        
        # Save configuration
        success = rag_service.set_llm_configuration(request.name, llm_config)
        
        if success:
            # Prepare response (without API key for security)
            response_config = llm_config.dict()
            if response_config.get('api_key'):
                response_config['api_key'] = "***REDACTED***"
            
            return LLMConfigResponse(
                name=llm_config.name,
                provider=llm_config.provider.value,
                model=llm_config.model,
                endpoint=llm_config.endpoint,
                system_prompt=llm_config.system_prompt,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                top_p=llm_config.top_p,
                top_k=llm_config.top_k,
                timeout=llm_config.timeout,
                message=f"LLM configuration '{request.name}' created/updated successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save LLM configuration")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating LLM configuration: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")


@router.get("/llm-configs", response_model=LLMConfigListResponse)
async def list_llm_configurations():
    """List all LLM configurations."""
    try:
        configurations = rag_service.get_llm_configurations()
        return LLMConfigListResponse(
            configurations=configurations,
            total_count=len(configurations)
        )
    except Exception as e:
        logger.error(f"Error listing LLM configurations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/llm-configs/{config_name}", response_model=LLMConfigResponse)
async def get_llm_configuration(config_name: str):
    """Get a specific LLM configuration."""
    try:
        llm_config = rag_service.get_llm_configuration(config_name)
        
        return LLMConfigResponse(
            name=llm_config.name,
            provider=llm_config.provider.value,
            model=llm_config.model,
            endpoint=llm_config.endpoint,
            system_prompt=llm_config.system_prompt,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            top_p=llm_config.top_p,
            top_k=llm_config.top_k,
            timeout=llm_config.timeout,
            message=f"LLM configuration '{config_name}' retrieved successfully"
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"LLM configuration '{config_name}' not found")
    except Exception as e:
        logger.error(f"Error getting LLM configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/llm-configs/{config_name}")
async def delete_llm_configuration(config_name: str):
    """Delete an LLM configuration."""
    try:
        success = rag_service.delete_llm_configuration(config_name)
        if success:
            return {"message": f"LLM configuration '{config_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"LLM configuration '{config_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting LLM configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Preset application endpoint has been removed

@router.delete("/documents")
async def delete_all_documents(configuration_name: str = Query(...), confirm: bool = Query(False)):
    """Delete all documents and their chunks from a configuration.
    
    Args:
        configuration_name: The configuration to delete documents from
        confirm: Must be set to true to confirm the deletion
        
    Returns:
        Results of the deletion operation
    """
    try:
        # Require explicit confirmation to prevent accidental deletion
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Deletion requires confirmation. Set confirm=true query parameter."
            )
            
        vector_store = rag_service._get_vector_store(configuration_name)
        
        # For FAISS store
        if hasattr(vector_store, 'documents'):
            # Get the count before deletion
            doc_count = len(vector_store.documents)
            
            # Clear documents and metadata
            vector_store.documents = []
            
            if hasattr(vector_store, 'metadata'):
                vector_store.metadata = []
                
            # Create new empty index
            if hasattr(vector_store, 'embedding_service') and hasattr(vector_store, 'dimension'):
                import faiss
                vector_store.index = faiss.IndexFlatIP(vector_store.dimension)
                
            # Save empty index
            if hasattr(vector_store, '_save_index'):
                vector_store._save_index()
                
            return {
                "message": f"Successfully deleted all documents ({doc_count}) from configuration '{configuration_name}'",
                "deleted_count": doc_count,
                "configuration_name": configuration_name
            }
            
        # For Redis store
        elif hasattr(vector_store, 'redis_client') and hasattr(vector_store, 'index_name'):
            # Get all keys
            all_keys = vector_store.redis_client.keys(f"doc:*")
            key_count = len(all_keys)
            
            if key_count > 0:
                # Delete all keys
                pipe = vector_store.redis_client.pipeline()
                for key in all_keys:
                    pipe.delete(key)
                pipe.execute()
                
                # Recreate index
                try:
                    vector_store.redis_client.ft(vector_store.index_name).dropindex()
                    # The _initialize_client method would normally be called to recreate the index
                    if hasattr(vector_store, '_initialize_client'):
                        vector_store._initialize_client()
                except:
                    logger.warning(f"Could not drop Redis index: {vector_store.index_name}")
                
            return {
                "message": f"Successfully deleted all documents ({key_count}) from configuration '{configuration_name}'",
                "deleted_count": key_count,
                "configuration_name": configuration_name
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unsupported vector store type for deletion: {type(vector_store).__name__}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting all documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting all documents: {str(e)}")


@router.get("/documents")
async def list_documents(configuration_name: str = Query(...), include_metadata: bool = True):
    """List all unique documents in a configuration.
    
    Args:
        configuration_name: The configuration to list documents from
        include_metadata: Whether to include detailed metadata in the response (default: True)
        
    Returns:
        List of unique document IDs and their metadata
    """
    try:
        vector_store = rag_service._get_vector_store(configuration_name)
        
        # Get all documents
        all_documents = vector_store.get_all_documents(limit=10000)  # Use a high limit to get most documents
        
        # Track unique document IDs and collect metadata
        unique_documents = {}
        for doc in all_documents:
            doc_id = doc.metadata.get('document_id')
            if not doc_id:
                continue
                
            if doc_id not in unique_documents:
                # Initialize with basic metadata from first chunk
                unique_documents[doc_id] = {
                    'document_id': doc_id,
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'chunk_count': 1
                }
                
                # Add any other useful metadata if include_metadata is True
                if include_metadata:
                    for key, value in doc.metadata.items():
                        if key not in ['document_id', 'chunk']:
                            unique_documents[doc_id][key] = value
            else:
                # Increment chunk count for existing documents
                unique_documents[doc_id]['chunk_count'] += 1
        
        return {
            'configuration_name': configuration_name,
            'document_count': len(unique_documents),
            'documents': list(unique_documents.values())
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str, configuration_name: str = Query(...), include_vectors: bool = False):
    """Retrieve all chunks of a specific document.
    
    Args:
        document_id: The ID of the document to retrieve chunks for
        configuration_name: The configuration where the document is stored
        include_vectors: Whether to include embedding vectors in the response (default: False)
        
    Returns:
        Dictionary containing all chunks of the document
    """
    try:
        vector_store = rag_service._get_vector_store(configuration_name)
        
        # Get all documents
        all_documents = vector_store.get_all_documents(limit=1000)  # Adjust limit as needed
        
        # Filter by document_id in metadata
        document_chunks = [
            {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in all_documents
            if doc.metadata.get('document_id') == document_id
        ]
        
        # Include embedding vectors if requested
        if include_vectors and document_chunks:
            try:
                # Get embedding service for the configuration
                embedding_service = rag_service._get_embedding_service(configuration_name)
                
                # Get the embeddings for the document contents
                texts = [chunk['content'] for chunk in document_chunks]
                embeddings = embedding_service.embed_texts(texts)
                
                # Add embeddings to chunks
                for i, chunk in enumerate(document_chunks):
                    if i < len(embeddings):
                        chunk['vector'] = embeddings[i]
            except Exception as e:
                logger.warning(f"Failed to include vectors: {str(e)}")
        
        return {
            'document_id': document_id,
            'configuration_name': configuration_name,
            'chunk_count': len(document_chunks),
            'chunks': document_chunks
        }
        
    except Exception as e:
        logger.error(f"Error retrieving document chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document chunks: {str(e)}")


@router.delete("/documents/{document_id}/chunks")
async def delete_document_chunks(document_id: str, configuration_name: str = Query(...)):
    """Delete all chunks of a specific document.
    
    Args:
        document_id: The ID of the document to delete chunks for
        configuration_name: The configuration where the document is stored
        
    Returns:
        Dictionary with deletion results
    """
    try:
        vector_store = rag_service._get_vector_store(configuration_name)
        
        # For FAISS store
        if hasattr(vector_store, 'documents'):
            # Find indices of documents to delete
            indices_to_delete = [
                i for i, doc in enumerate(vector_store.documents)
                if doc.metadata.get('document_id') == document_id
            ]
            
            if not indices_to_delete:
                return {"message": f"No chunks found for document ID: {document_id}", "deleted": 0}
            
            # Create a new list of documents excluding the ones to delete
            vector_store.documents = [
                doc for i, doc in enumerate(vector_store.documents) 
                if i not in indices_to_delete
            ]
            
            # Update metadata list to match documents
            if hasattr(vector_store, 'metadata') and len(vector_store.metadata) >= len(indices_to_delete):
                vector_store.metadata = [
                    meta for i, meta in enumerate(vector_store.metadata)
                    if i not in indices_to_delete
                ]
            
            # Rebuild the index (simplified approach - rebuild from scratch)
            if hasattr(vector_store, 'embedding_service') and vector_store.documents:
                # Extract texts
                texts = [doc.page_content for doc in vector_store.documents]
                
                # Generate new embeddings
                embeddings = vector_store.embedding_service.embed_texts(texts)
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                # Normalize embeddings for cosine similarity
                import faiss
                faiss.normalize_L2(embeddings_array)
                
                # Create new index
                vector_store.index = faiss.IndexFlatIP(vector_store.dimension)
                vector_store.index.add(embeddings_array)
            
            # Save index
            if hasattr(vector_store, '_save_index'):
                vector_store._save_index()
            
            return {
                "message": f"Successfully deleted {len(indices_to_delete)} chunks for document ID: {document_id}",
                "deleted": len(indices_to_delete)
            }
            
        # For Redis store
        elif hasattr(vector_store, 'redis_client'):
            # Redis supports metadata filtering
            all_keys = vector_store.redis_client.keys(f"doc:*")
            doc_ids_to_delete = []
            
            for key in all_keys:
                doc_data = vector_store.redis_client.hgetall(key)
                if doc_data and b'metadata' in doc_data:
                    try:
                        metadata = json.loads(doc_data[b'metadata'].decode('utf-8'))
                        if metadata.get('document_id') == document_id:
                            doc_ids_to_delete.append(key.decode('utf-8').replace('doc:', ''))
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid metadata JSON for key: {key}")
                        continue
            
            if doc_ids_to_delete:
                vector_store.delete_documents(doc_ids_to_delete)
                return {
                    "message": f"Successfully deleted {len(doc_ids_to_delete)} chunks for document ID: {document_id}",
                    "deleted": len(doc_ids_to_delete)
                }
            else:
                return {"message": f"No chunks found for document ID: {document_id}", "deleted": 0}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unsupported vector store type for deletion: {type(vector_store).__name__}"
            )
        
    except Exception as e:
        logger.error(f"Error deleting document chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document chunks: {str(e)}")


@router.post("/configurations/reload")
async def reload_configurations():
    """Reload configurations from file.
    
    This endpoint forces the server to reload all configurations from the storage file,
    discarding any in-memory changes that haven't been saved.
    """
    try:
        success = rag_service.reload_configurations()
        
        if success:
            return {"message": "Configurations reloaded successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload configurations")
            
    except Exception as e:
        logger.error(f"Error reloading configurations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
