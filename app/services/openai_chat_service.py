"""Service for handling OpenAI-compatible chat completions with RAG."""
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import json

from app.model_schemas.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatMessage,
    UsageInfo
)
from app.services.rag_service import RAGService
from app.config import RAGConfig

logger = logging.getLogger(__name__)


class OpenAIChatService:
    """Service for handling OpenAI-compatible chat completions with RAG integration."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
    def _extract_user_query(self, messages: List[ChatMessage]) -> str:
        """Extract the user query from the messages.
        
        Args:
            messages: List of chat messages
            
        Returns:
            The last user message content
        """
        # Find the last user message
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        
        # If no user message found, use the last message
        if messages:
            return messages[-1].content
        
        raise ValueError("No user message found in the conversation")
    
    def _extract_system_prompt(self, messages: List[ChatMessage]) -> Optional[str]:
        """Extract system prompt from messages.
        
        Args:
            messages: List of chat messages
            
        Returns:
            System prompt if found, None otherwise
        """
        for message in messages:
            if message.role == "system":
                return message.content
        return None
    
    def _extract_context_items(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Extract conversation history as context items.
        
        Args:
            messages: List of chat messages
            
        Returns:
            List of context items (excluding the last user message and system messages)
        """
        context_items = []
        
        # Skip system messages and the last user message
        for i, message in enumerate(messages[:-1]):
            if message.role != "system":
                context_items.append({
                    "content": message.content,
                    "role": message.role,
                    "timestamp": datetime.now(),
                    "metadata": {"message_index": i}
                })
        
        return context_items
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        configuration_name: str,
        authorization_header: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Create a chat completion using RAG.
        
        Args:
            request: Chat completion request
            configuration_name: RAG configuration to use
            authorization_header: Optional authorization header for security
            
        Returns:
            Chat completion response
        """
        start_time = time.time()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        
        try:
            # Extract query and context from messages
            user_query = self._extract_user_query(request.messages)
            system_prompt = self._extract_system_prompt(request.messages)
            context_items = self._extract_context_items(request.messages)
            
            logger.info(f"Processing chat completion for configuration '{configuration_name}': {user_query[:100]}...")
            
            # Build config override if temperature/max_tokens specified
            config_override = None
            if request.rag_config or request.temperature != 0.7 or request.max_tokens:
                # Get base config
                base_config = self.rag_service.get_configuration(configuration_name)
                config_dict = base_config.dict()
                
                # Apply overrides
                if request.rag_config:
                    config_dict.update(request.rag_config)
                
                # Override generation parameters
                if config_dict.get("generation"):
                    if request.temperature is not None:
                        config_dict["generation"]["temperature"] = request.temperature
                    if request.max_tokens is not None:
                        config_dict["generation"]["max_tokens"] = request.max_tokens
                    if request.top_p is not None:
                        config_dict["generation"]["top_p"] = request.top_p
                
                config_override = RAGConfig(**config_dict)
            
            # Validate security and merge filters
            merged_filter = self.rag_service.validate_security_and_merge_filters(
                configuration_name=configuration_name,
                authorization_header=authorization_header,
                request_filter=request.filter
            )
            
            # Call RAG service
            query_response = await self.rag_service.query(
                query=user_query,
                configuration_name=configuration_name,
                k=request.k,
                similarity_threshold=request.similarity_threshold,
                context_items=context_items if context_items else None,
                config_override=config_override,
                system_prompt=system_prompt,
                filter_after_reranking=request.filter_after_reranking,
                filter=merged_filter
            )
            
            # Build OpenAI-compatible response
            assistant_message = ChatMessage(
                role="assistant",
                content=query_response.answer
            )
            
            choice = ChatCompletionChoice(
                index=0,
                message=assistant_message,
                finish_reason="stop"
            )
            
            # Estimate token usage
            prompt_text = " ".join([msg.content for msg in request.messages])
            prompt_tokens = self._estimate_tokens(prompt_text)
            completion_tokens = self._estimate_tokens(query_response.answer)
            
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            processing_time = time.time() - start_time
            
            response = ChatCompletionResponse(
                id=completion_id,
                object="chat.completion",
                created=int(datetime.now().timestamp()),
                model=configuration_name,
                choices=[choice],
                usage=usage,
                processing_time=processing_time
            )
            
            # Include sources if requested
            if request.include_sources:
                response.sources = query_response.sources
            
            logger.info(f"Chat completion completed in {processing_time:.2f}s for configuration '{configuration_name}'")
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating chat completion: {str(e)}")
            raise
    
    async def create_chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        configuration_name: str,
        authorization_header: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion using RAG.
        
        Args:
            request: Chat completion request
            configuration_name: RAG configuration to use
            authorization_header: Optional authorization header for security
            
        Yields:
            Server-sent event formatted chunks
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        
        try:
            # For now, we'll get the full response and stream it token by token
            # In a production system, you'd want to integrate with streaming LLM APIs
            
            # Get the full response first
            response = await self.create_chat_completion(
                request=request,
                configuration_name=configuration_name,
                authorization_header=authorization_header
            )
            
            answer = response.choices[0].message.content
            
            # Stream the answer word by word
            words = answer.split()
            
            for i, word in enumerate(words):
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": configuration_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": word + " " if i < len(words) - 1 else word
                            },
                            "finish_reason": None
                        }
                    ]
                }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send final chunk with finish_reason
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": configuration_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {str(e)}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
