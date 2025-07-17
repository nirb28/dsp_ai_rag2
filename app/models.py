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

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    configuration_name: str = "default"
    k: Optional[int] = Field(default=5, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = True
    context_items: Optional[List[ContextItem]] = Field(default=None, description="Additional context for context injection (e.g. chat history)")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Optional partial config overrides for generation endpoint, embedding endpoint, or vector store")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    configuration_name: str

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
    similarity_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    include_metadata: bool = True
    use_reranking: bool = False
    include_vectors: bool = False
    config: Optional[Dict[str, Any]] = None  # Optional partial config overrides
    fusion_method: Optional[str] = "rrf"  # Options: "rrf", "simple"
    rrf_k_constant: int = Field(default=60, ge=1)  # Constant for RRF calculation

class RetrieveResponse(BaseModel):
    query: str
    documents: List[Dict[str, Any]]
    processing_time: float
    configuration_name: Optional[str] = None
    configuration_names: Optional[List[str]] = None
    total_found: int
    fusion_method: Optional[str] = None


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
