"""Unified OpenAI-compatible API endpoints (single /v1/chat/completions for all configs)."""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, status
from fastapi.responses import StreamingResponse

from app.model_schemas.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    ErrorDetail
)
from app.services.openai_chat_service import OpenAIChatService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


def create_unified_openai_router(rag_service: RAGService) -> APIRouter:
    """Create a unified OpenAI-compatible router (model parameter determines configuration).
    
    This provides a standard OpenAI-style API where all requests go to /v1/chat/completions
    and the 'model' parameter in the request body determines which RAG configuration to use.
    
    Args:
        rag_service: RAG service instance
        
    Returns:
        FastAPI router with unified OpenAI-compatible endpoints
    """
    router = APIRouter(
        prefix="/v1",
        tags=["OpenAI Compatible (Unified)"]
    )
    
    openai_chat_service = OpenAIChatService(rag_service)
    
    @router.post(
        "/chat/completions",
        response_model=ChatCompletionResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Authentication failed"},
            404: {"model": ErrorResponse, "description": "Model (configuration) not found"},
            500: {"model": ErrorResponse, "description": "Internal server error"}
        },
        summary="Create chat completion",
        description="""
        Create a chat completion using RAG. The 'model' parameter in the request body
        specifies which RAG configuration to use.
        
        This endpoint mimics the standard OpenAI chat completions API, making it easy
        to use with existing OpenAI client libraries and tools.
        
        **Example:**
        ```json
        {
          "model": "malts_faq",
          "messages": [
            {"role": "user", "content": "What are malts?"}
          ]
        }
        ```
        
        **RAG-specific features:**
        - Automatically retrieves relevant documents based on the user query
        - Supports metadata filtering for document retrieval
        - Includes source documents in the response (optional)
        - Supports conversation history as context
        
        **Authentication:**
        If security is enabled for the configuration, include a JWT token in the
        Authorization header: `Bearer <token>`
        """
    )
    async def create_chat_completion(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(None, description="Bearer token for authentication")
    ):
        """Create a chat completion using the specified model (configuration)."""
        try:
            # Extract configuration name from model parameter
            configuration_name = request.model
            
            if not configuration_name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Model parameter is required"
                )
            
            # Verify configuration exists
            try:
                rag_service.get_configuration(configuration_name)
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model (configuration) '{configuration_name}' not found. "
                           f"Use GET /v1/models to see available models."
                )
            
            # Handle streaming
            if request.stream:
                return StreamingResponse(
                    openai_chat_service.create_chat_completion_stream(
                        request=request,
                        configuration_name=configuration_name,
                        authorization_header=authorization
                    ),
                    media_type="text/event-stream"
                )
            
            # Non-streaming response
            response = await openai_chat_service.create_chat_completion(
                request=request,
                configuration_name=configuration_name,
                authorization_header=authorization
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in unified chat completion endpoint: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @router.get(
        "/models",
        summary="List available models",
        description="""
        List all available models (RAG configurations).
        
        Returns a list of all RAG configurations that can be used as the 'model'
        parameter in chat completion requests.
        
        Compatible with OpenAI's list models API.
        """
    )
    async def list_models():
        """List all available models (configurations)."""
        try:
            # Get all configuration names
            config_names = rag_service.get_configuration_names()
            
            # Build OpenAI-compatible response
            models = []
            for config_name in config_names:
                models.append({
                    "id": config_name,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "rag-system",
                    "permission": [],
                    "root": config_name,
                    "parent": None
                })
            
            return {
                "object": "list",
                "data": models
            }
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @router.get(
        "/models/{model_id}",
        summary="Retrieve model information",
        description="""
        Retrieve information about a specific model (RAG configuration).
        
        Compatible with OpenAI's retrieve model API.
        """
    )
    async def retrieve_model(model_id: str):
        """Retrieve information about a specific model."""
        try:
            # Verify configuration exists
            try:
                rag_service.get_configuration(model_id)
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_id}' not found"
                )
            
            return {
                "id": model_id,
                "object": "model",
                "created": 1677610602,
                "owned_by": "rag-system",
                "permission": [],
                "root": model_id,
                "parent": None
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving model: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    return router
