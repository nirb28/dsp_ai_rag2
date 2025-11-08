"""OpenAI-compatible API models for chat completions."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime


class ChatMessage(BaseModel):
    """A chat message in the conversation."""
    role: Literal["system", "user", "assistant", "function"] = Field(
        ..., 
        description="The role of the message author"
    )
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="The name of the author of this message")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call information")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(
        default="default",
        description="Model to use (maps to RAG configuration name)"
    )
    messages: List[ChatMessage] = Field(
        ..., 
        description="List of messages in the conversation"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    n: Optional[int] = Field(
        default=1,
        ge=1,
        description="Number of completions to generate"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the response"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Stop sequences"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum tokens to generate"
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None,
        description="Logit bias"
    )
    user: Optional[str] = Field(
        None,
        description="Unique identifier for the end-user"
    )
    
    # RAG-specific extensions
    rag_config: Optional[Dict[str, Any]] = Field(
        None,
        description="RAG-specific configuration overrides"
    )
    k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum similarity score for retrieval"
    )
    filter: Optional[Dict[str, Any]] = Field(
        None,
        description="LangChain-style metadata filter for document retrieval"
    )
    filter_after_reranking: Optional[bool] = Field(
        default=True,
        description="Whether to apply score threshold filtering after reranking"
    )
    include_sources: Optional[bool] = Field(
        default=True,
        description="Whether to include source documents in the response"
    )


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int = Field(..., description="The index of this choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: Optional[str] = Field(
        None,
        description="The reason the completion finished"
    )
    logprobs: Optional[Dict[str, Any]] = Field(
        None,
        description="Log probabilities"
    )


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatCompletionChoice] = Field(..., description="List of completion choices")
    usage: UsageInfo = Field(..., description="Token usage information")
    system_fingerprint: Optional[str] = Field(
        None,
        description="System fingerprint"
    )
    
    # RAG-specific extensions
    sources: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Source documents used for RAG (if include_sources=True)"
    )
    processing_time: Optional[float] = Field(
        None,
        description="Processing time in seconds"
    )


class ChatCompletionChunk(BaseModel):
    """A chunk of a streaming chat completion."""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(..., description="List of completion choices")
    system_fingerprint: Optional[str] = Field(
        None,
        description="System fingerprint"
    )


class ErrorDetail(BaseModel):
    """Error detail information."""
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: Optional[str] = Field(None, description="Parameter that caused the error")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""
    error: ErrorDetail = Field(..., description="Error details")
