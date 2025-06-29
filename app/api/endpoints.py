import os
import tempfile
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging

from app.models import (
    DocumentUploadResponse, QueryRequest, QueryResponse, 
    ConfigurationRequest, ConfigurationResponse, HealthResponse,
    ErrorResponse, CollectionsResponse, CollectionInfo
)
from app.config import RAGConfig, DEFAULT_CONFIGS, settings
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
    collection_name: str = Form(default="default"),
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
                collection_name=collection_name,
                metadata=doc_metadata,
                process_immediately=process_immediately
            )
            
            return DocumentUploadResponse(
                document_id=document.id,
                filename=document.filename,
                status=document.status,
                collection_name=collection_name,
                message=f"Document uploaded successfully to collection '{collection_name}'"
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
    """Query documents in a collection.
    
    Supports optional context items for context injection (e.g., chat history)
    and uses reranking if configured for the collection.
    """
    try:
        # Check if reranking and context injection are enabled for this collection
        config = rag_service.get_configuration(request.collection_name)
        reranking_enabled = hasattr(config, 'reranking') and config.reranking and config.reranking.enabled
        context_enabled = hasattr(config, 'context_injection') and config.context_injection and config.context_injection.enabled
        
        if context_enabled:
            logger.info(f"Context injection enabled for collection '{request.collection_name}'")
        if reranking_enabled:
            logger.info(f"Reranking enabled for collection '{request.collection_name}' using model: {config.reranking.model}")
        
        # Convert context items to list of dictionaries if provided
        context_items = None
        if request.context_items:
            context_items = [item.dict() for item in request.context_items]
            logger.info(f"Using {len(context_items)} context items for query")
        
        response = await rag_service.query(
            query=request.query,
            collection_name=request.collection_name,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
            context_items=context_items
        )
        
        # Filter metadata if requested
        if not request.include_metadata:
            for source in response.sources:
                source['metadata'] = {}
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/configure", response_model=ConfigurationResponse)
async def configure_collection(request: ConfigurationRequest):
    """Configure a collection with custom settings."""
    try:
        # Validate and create RAG config
        config = RAGConfig(**request.config)
        config.collection_name = request.collection_name
        
        # Set configuration
        success = rag_service.set_configuration(request.collection_name, config)
        
        if success:
            return ConfigurationResponse(
                collection_name=request.collection_name,
                config=config.dict(),
                message=f"Configuration updated for collection '{request.collection_name}'"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
            
    except Exception as e:
        logger.error(f"Error configuring collection: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")

@router.get("/configure/{collection_name}")
async def get_configuration(collection_name: str):
    """Get configuration for a collection."""
    try:
        config = rag_service.get_configuration(collection_name)
        return {
            "collection_name": collection_name,
            "config": config.dict()
        }
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all collections and their information."""
    try:
        collections_data = rag_service.get_collections()
        
        collections = []
        for data in collections_data:
            collections.append(CollectionInfo(
                name=data['name'],
                document_count=data['document_count'],
                created_at=data.get('created_at'),
                last_updated=data.get('last_updated'),
                config=data['config']
            ))
        
        return CollectionsResponse(
            collections=collections,
            total_count=len(collections)
        )
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection."""
    try:
        success = rag_service.delete_collection(collection_name)
        
        if success:
            return {"message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/presets")
async def get_configuration_presets():
    """Get available configuration presets."""
    return {
        "presets": {
            name: config.dict() for name, config in DEFAULT_CONFIGS.items()
        },
        "description": {
            "fast_processing": "Optimized for speed with smaller chunks and faster models",
            "high_quality": "Optimized for quality with larger chunks and better models",
            "balanced": "Balanced approach between speed and quality"
        }
    }

@router.get("/configurations")
async def list_all_configurations():
    """List all configurations across all collections."""
    try:
        # Get all configurations from the RAG service
        configurations = {}
        
        for collection_name, config in rag_service.configurations.items():
            configurations[collection_name] = config.dict()
        
        return {
            "configurations": configurations,
            "total_count": len(configurations)
        }
        
    except Exception as e:
        logger.error(f"Error listing configurations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/debug/{collection_name}")
async def debug_collection(collection_name: str, limit: int = 10, show_vectors: bool = False):
    """Debug endpoint to inspect the vectors and chunks in a collection.
    
    Args:
        collection_name: Name of the collection to inspect
        limit: Maximum number of chunks to return
        show_vectors: Whether to include embedding vectors in the response
        
    Returns:
        Dictionary containing chunks and their metadata from the collection
    """
    try:
        # Check if collection exists
        if collection_name not in rag_service.configurations:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Get vector store
        vector_store = rag_service._get_vector_store(collection_name)
        
        # Get document count
        document_count = vector_store.get_document_count()
        
        if document_count == 0:
            return {
                "collection_name": collection_name,
                "chunks": [],
                "total_count": 0,
                "message": "Collection is empty. No chunks found."
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
                "text": chunk.page_content,
                "metadata": chunk.metadata,
                "text_length": len(chunk.page_content),
                "text_preview": chunk.page_content[:100] + '...' if len(chunk.page_content) > 100 else chunk.page_content
            }
            
            # Add embedding from our direct extraction if available
            if show_vectors and vectors and i < len(vectors):
                chunk_data["embedding"] = {
                    "dimensions": vector_dimensions,
                    "preview": vectors[i][:5] if vector_dimensions > 5 else vectors[i]
                }
            
            chunks_data.append(chunk_data)
        
        return {
            "collection_name": collection_name,
            "chunks": chunks_data,
            "total_count": document_count,
            "shown_count": len(chunks_data),
            "has_embeddings": vectors is not None,
            "embedding_dimensions": vector_dimensions,
            "config": rag_service.configurations[collection_name].dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error debugging collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/configure/preset/{preset_name}")
async def apply_configuration_preset(preset_name: str, collection_name: str = "default"):
    """Apply a configuration preset to a collection."""
    try:
        if preset_name not in DEFAULT_CONFIGS:
            raise HTTPException(
                status_code=404, 
                detail=f"Preset '{preset_name}' not found. Available presets: {list(DEFAULT_CONFIGS.keys())}"
            )
        
        config = DEFAULT_CONFIGS[preset_name]
        config.collection_name = collection_name
        
        success = rag_service.set_configuration(collection_name, config)
        
        if success:
            return ConfigurationResponse(
                collection_name=collection_name,
                config=config.dict(),
                message=f"Applied preset '{preset_name}' to collection '{collection_name}'"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to apply preset")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying preset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
