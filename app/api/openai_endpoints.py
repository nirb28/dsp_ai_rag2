"""OpenAI-compatible API endpoints for chat completions."""
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

# Global RAG service instance
rag_service = RAGService()
openai_chat_service = OpenAIChatService(rag_service)


def create_openai_router(configuration_name: str) -> APIRouter:
    """Create an OpenAI-compatible router for a specific configuration.
    
    Args:
        configuration_name: Name of the RAG configuration
        
    Returns:
        FastAPI router with OpenAI-compatible endpoints
    """
    router = APIRouter(
        prefix=f"/{configuration_name}/v1",
        tags=[f"OpenAI Compatible - {configuration_name}"]
    )
    
    @router.post(
        "/chat/completions",
        response_model=ChatCompletionResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Authentication failed"},
            404: {"model": ErrorResponse, "description": "Configuration not found"},
            500: {"model": ErrorResponse, "description": "Internal server error"}
        },
        summary=f"Create chat completion for {configuration_name}",
        description=f"""
        Create a chat completion using the '{configuration_name}' RAG configuration.
        
        This endpoint is OpenAI-compatible and can be used as a drop-in replacement
        for OpenAI's chat completions API. It integrates RAG (Retrieval Augmented Generation)
        to provide context-aware responses based on your document collection.
        
        **RAG-specific features:**
        - Automatically retrieves relevant documents based on the user query
        - Supports metadata filtering for document retrieval
        - Includes source documents in the response (optional)
        - Supports conversation history as context
        
        **Authentication:**
        If security is enabled for this configuration, include a JWT token in the
        Authorization header: `Bearer <token>`
        """
    )
    async def create_chat_completion(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(None, description="Bearer token for authentication")
    ):
        """Create a chat completion for this configuration."""
        try:
            # Verify configuration exists
            try:
                rag_service.get_configuration(configuration_name)
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Configuration '{configuration_name}' not found"
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
            logger.error(f"Error in chat completion endpoint: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @router.get(
        "/models",
        summary=f"List models for {configuration_name}",
        description=f"List available models (returns the configuration name)"
    )
    async def list_models():
        """List available models for this configuration."""
        return {
            "object": "list",
            "data": [
                {
                    "id": configuration_name,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "rag-system",
                    "permission": [],
                    "root": configuration_name,
                    "parent": None
                }
            ]
        }
    
    @router.get(
        "/models/{model_id}",
        summary=f"Retrieve model info for {configuration_name}",
        description=f"Retrieve information about the model (configuration)"
    )
    async def retrieve_model(model_id: str):
        """Retrieve model information."""
        if model_id != configuration_name:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found"
            )
        
        return {
            "id": configuration_name,
            "object": "model",
            "created": 1677610602,
            "owned_by": "rag-system",
            "permission": [],
            "root": configuration_name,
            "parent": None
        }
    
    return router


def get_all_openai_routers(rag_service_instance: RAGService) -> list[APIRouter]:
    """Get OpenAI-compatible routers for all configurations.
    
    Args:
        rag_service_instance: RAG service instance
        
    Returns:
        List of FastAPI routers
    """
    routers = []
    
    # Get all configuration names
    config_names = rag_service_instance.get_configuration_names()
    
    for config_name in config_names:
        router = create_openai_router(config_name)
        routers.append(router)
        logger.info(f"Created OpenAI-compatible router for configuration: {config_name}")
    
    return routers
