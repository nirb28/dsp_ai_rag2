import os
import tempfile
import time
import json
import numpy as np
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
import logging

# Import the documentation router
from app.api.documentation import router as documentation_router

from app.models import (
    DocumentUploadResponse, QueryRequest, QueryResponse, 
    ConfigurationRequest, ConfigurationResponse, HealthResponse,
    ErrorResponse, ConfigurationsResponse, ConfigurationInfo,
    ConfigurationNamesResponse, RetrieveRequest, RetrieveResponse
)
from app.config import RAGConfig, settings
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreManager, FAISSVectorStore

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

router = APIRouter()

# Include documentation router
router.include_router(documentation_router, prefix="/documentation")

# Global RAG service instance
rag_service = RAGService()

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
        
        try:
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
                configuration_name=configuration_name,
                message=f"Document uploaded successfully to configuration '{configuration_name}'"
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents in a configuration.
    
    Uses reranking if configured for the configuration.
    """
    try:
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
        
        response = await rag_service.query(
            query=request.query,
            configuration_name=request.configuration_name,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
            context_items=request.context_items,
            config_override=config if temp_config else None,
            system_prompt=system_prompt
        )
        
        # Filter metadata if requested
        if not request.include_metadata:
            for source in response.sources:
                source['metadata'] = {}
        
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
        config.configuration_name = request.configuration_name
        
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
                    name=data['name'],
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

@router.delete("/configurations/{configuration_name}")
async def delete_configuration(configuration_name: str):
    """Delete a configuration."""
    try:
        success = rag_service.delete_configuration(configuration_name)
        if success:
            return {"message": f"Configuration '{configuration_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Configuration '{configuration_name}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """Retrieve relevant documents from a configuration without generating a response.
    
    This endpoint allows direct access to the vector retrieval functionality without LLM generation.
    It's useful for debugging, testing, or when only the context documents are needed.
    
    Args:
        request: The retrieval request containing query and options
    
    Returns:
        Documents retrieved from the vector store, optionally reranked
    """
    try:
        start_time = time.time()
        configuration_name = request.configuration_name
        
        # Check if configuration exists in RAG service but not in vector store manager
        if configuration_name in rag_service.configurations and not rag_service.vector_store_manager.configuration_exists(configuration_name):
            logger.info(f"Configuration '{configuration_name}' exists in RAG service but not in vector store manager. Initializing vector store.")
            # Initialize the vector store by accessing it
            try:
                # This will create the vector store if it doesn't exist
                rag_service._get_vector_store(configuration_name)
            except Exception as e:
                logger.error(f"Error initializing vector store for {configuration_name}: {str(e)}")
        
        # Validate configuration exists in vector store manager
        if not rag_service.vector_store_manager.configuration_exists(configuration_name):
            raise HTTPException(
                status_code=404, 
                detail=f"Configuration '{configuration_name}' not found"
            )
        
        # Get the vector store for this configuration
        vector_store = rag_service._get_vector_store(configuration_name)
        
        # Get configuration, with optional overrides from the request
        config = rag_service.get_configuration(configuration_name)
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
        
        # Get embeddings service for this configuration
        embedding_service = rag_service._get_embedding_service(configuration_name)
        
        # Get reranker service if needed
        reranker_service = None
        if request.use_reranking:
            reranker_service = rag_service._get_reranker_service(configuration_name)
        
        # Get query parameters
        k = request.k
        similarity_threshold = request.similarity_threshold if request.similarity_threshold is not None else config.retrieval.similarity_threshold
        
        # Retrieve documents
        results = vector_store.similarity_search(
            request.query,
            k=k,
            similarity_threshold=similarity_threshold
        )
        
        # Process documents for response
        documents = []
        for doc, score in results:
            document = {
                'content': doc.page_content,
                'similarity_score': score
            }
            
            if request.include_metadata:
                document['metadata'] = doc.metadata
                
            documents.append(document)
        
        # Apply reranking if requested
        if request.use_reranking and reranker_service and reranker_service.config.enabled and documents:
            documents = await reranker_service.rerank(request.query, documents)
        
        # Include embedding vectors if requested
        if request.include_vectors and documents:
            try:
                # Get the embeddings for the document contents
                texts = [doc['content'] for doc in documents]
                embeddings = embedding_service.embed_texts(texts)
                
                # Add embeddings to documents
                for i, doc in enumerate(documents):
                    if i < len(embeddings):
                        doc['vector'] = embeddings[i]
            except Exception as e:
                logger.warning(f"Failed to include vectors: {str(e)}")
        
        processing_time = time.time() - start_time
        
        return RetrieveResponse(
            query=request.query,
            documents=documents,
            processing_time=processing_time,
            configuration_name=configuration_name,
            total_found=len(documents)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
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
async def list_documents(configuration_name: str = Query(...)):
    """List all unique documents in a configuration.
    
    Args:
        configuration_name: The configuration to list documents from
        
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
                
                # Add any other useful metadata
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
async def get_document_chunks(document_id: str, configuration_name: str = Query(...)):
    """Retrieve all chunks of a specific document.
    
    Args:
        document_id: The ID of the document to retrieve chunks for
        configuration_name: The configuration where the document is stored
        
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
