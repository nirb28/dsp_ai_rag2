from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime
from enum import Enum

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"

class Document(BaseModel):
    id: str
    filename: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.UPLOADED
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    configuration_name: str
    file_size: int
    file_type: str
    error: Optional[str] = None

class DocumentUploadRequest(BaseModel):
    configuration_name: str = "default"
    metadata: Optional[Dict[str, Any]] = None
    process_immediately: bool = True

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str
    configuration_name: str

class ContextItem(BaseModel):
    """A single context item for context injection."""
    content: str
    role: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryExpansionRequest(BaseModel):
    """Query expansion configuration for requests."""
    enabled: bool = Field(default=True, description="Whether to enable query expansion")
    strategy: str = Field(default="fusion", description="Query expansion strategy: 'fusion' or 'multi_query'")
    llm_config_name: str = Field(..., description="Name of the LLM configuration to use")
    num_queries: int = Field(default=3, ge=1, le=10, description="Number of expanded queries to generate")
    include_metadata: bool = Field(default=False, description="Whether to include query expansion metadata in the response")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    configuration_name: str = "default"
    k: Optional[int] = Field(default=5, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0)
    include_metadata: bool = True
    context_items: Optional[List[ContextItem]] = Field(default=None, description="Additional context for context injection (e.g. chat history)")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Optional partial config overrides for generation endpoint, embedding endpoint, or vector store")
    filter_after_reranking: bool = Field(default=True, description="Whether to apply score threshold filtering after reranking")
    query_expansion: Optional[QueryExpansionRequest] = Field(default=None, description="Optional query expansion configuration")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="LangChain-style metadata filter for document retrieval")
    debug: bool = Field(default=False, description="Whether to log detailed request and response payloads for debugging")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    configuration_name: str
    query_expansion_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata about query expansion if enabled")

class ConfigurationRequest(BaseModel):
    config: Dict[str, Any]
    configuration_name: str = "default"

class ConfigurationResponse(BaseModel):
    configuration_name: str
    config: Dict[str, Any]
    message: str
    
class DeleteConfigurationResponse(BaseModel):
    success: bool
    message: str
    configuration_name: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ConfigurationInfo(BaseModel):
    configuration_name: str
    document_count: int
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    config: Dict[str, Any]

class ConfigurationsResponse(BaseModel):
    configurations: List[ConfigurationInfo]
    total_count: int
    
class ConfigurationNamesResponse(BaseModel):
    """Simple response model for configuration names only"""
    names: List[str]
    total_count: int

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    configuration_name: Optional[str] = "default"
    configuration_names: Optional[List[str]] = None
    k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0)
    include_metadata: bool = True
    use_reranking: bool = False
    include_vectors: bool = False
    config: Optional[Dict[str, Any]] = None  # Optional partial config overrides
    fusion_method: Optional[str] = "rrf"  # Options: "rrf", "simple"
    rrf_k_constant: int = Field(default=60, ge=1)  # Constant for RRF calculation
    filter_after_reranking: bool = Field(default=True, description="Whether to apply score threshold filtering after reranking")
    query_expansion: Optional[QueryExpansionRequest] = Field(default=None, description="Optional query expansion configuration")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="LangChain-style metadata filter for document retrieval")
    debug: bool = Field(default=False, description="Whether to log detailed request and response payloads for debugging")

class RetrieveResponse(BaseModel):
    query: str
    documents: List[Dict[str, Any]]
    processing_time: float
    configuration_name: Optional[str] = None
    configuration_names: Optional[List[str]] = None
    total_found: int
    fusion_method: Optional[str] = None
    query_expansion_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata about query expansion if enabled")


class TextDocument(BaseModel):
    """Model for a text document to be uploaded without a file."""
    content: str = Field(..., min_length=1)
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextDocumentsUploadRequest(BaseModel):
    """Request model for uploading multiple text documents."""
    documents: List[TextDocument] = Field(..., min_items=1)
    configuration_name: str = "default"
    process_immediately: bool = True


class TextDocumentsUploadResponse(BaseModel):
    """Response model for multiple text documents upload."""
    documents: List[DocumentUploadResponse]
    total_count: int
    configuration_name: str
    message: str


class DuplicateConfigurationRequest(BaseModel):
    """Request model for duplicating a configuration."""
    source_configuration_name: str
    target_configuration_name: str
    include_documents: bool = False


class DuplicateConfigurationResponse(BaseModel):
    """Response model for configuration duplication."""
    source_configuration_name: str
    target_configuration_name: str
    config: Dict[str, Any]
    documents_copied: int = 0
    message: str


class LLMConfigRequest(BaseModel):
    """Request model for creating/updating LLM configurations."""
    name: str = Field(..., description="Unique name for this LLM configuration")
    provider: str = Field(..., description="The LLM provider type (groq, triton, openai_compatible)")
    model: str = Field(..., description="Model name")
    endpoint: str = Field(..., description="API endpoint URL")
    api_key: Optional[str] = Field(default=None, description="API key for the provider (if required)")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for query expansion")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Top-k sampling")
    timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")


class LLMConfigResponse(BaseModel):
    """Response model for LLM configuration operations."""
    name: str
    provider: str
    model: str
    endpoint: str
    system_prompt: Optional[str] = None
    temperature: float
    max_tokens: int
    top_p: float
    top_k: Optional[int] = None
    timeout: int
    message: str


class LLMConfigListResponse(BaseModel):
    """Response model for listing LLM configurations."""
    configurations: List[Dict[str, Any]]
    total_count: int


class MCPServerStatusResponse(BaseModel):
    """Response model for MCP server status."""
    configuration_name: str
    enabled: bool
    running: bool
    protocols: List[str]
    endpoints: Dict[str, str]  # protocol -> endpoint mapping
    tools: List[Dict[str, Any]]
    client_count: int = 0
    uptime_seconds: Optional[float] = None
    message: str


class MCPServerListResponse(BaseModel):
    """Response model for listing MCP servers."""
    servers: List[MCPServerStatusResponse]
    total_count: int


class MCPToolExecutionRequest(BaseModel):
    """Request model for MCP tool execution."""
    tool_name: str
    parameters: Dict[str, Any]


class MCPToolExecutionResponse(BaseModel):
    """Response model for MCP tool execution."""
    tool_name: str
    result: Any
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class MCPServerStartRequest(BaseModel):
    """Request model for starting MCP server."""
    configuration_name: str
    force_restart: bool = False


class MCPServerStopRequest(BaseModel):
    """Request model for stopping MCP server."""
    configuration_name: str


class MCPServerStartStopResponse(BaseModel):
    """Response model for MCP server start/stop operations."""
    configuration_name: str
    action: str  # "started" or "stopped"
    success: bool
    message: str
    endpoints: Optional[Dict[str, str]] = None
