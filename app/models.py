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
    collection_name: str
    file_size: int
    file_type: str

class DocumentUploadRequest(BaseModel):
    collection_name: str = "default"
    metadata: Optional[Dict[str, Any]] = None
    process_immediately: bool = True

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str
    collection_name: str

class ContextItem(BaseModel):
    """A single context item for context injection."""
    content: str
    role: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    collection_name: str = "default"
    k: Optional[int] = Field(default=5, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = True
    context_items: Optional[List[ContextItem]] = Field(default=None, description="Additional context for context injection (e.g. chat history)")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    collection_name: str

class ConfigurationRequest(BaseModel):
    config: Dict[str, Any]
    collection_name: str = "default"

class ConfigurationResponse(BaseModel):
    collection_name: str
    config: Dict[str, Any]
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    config: Dict[str, Any]

class CollectionsResponse(BaseModel):
    collections: List[CollectionInfo]
    total_count: int

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    collection_name: str = "default"
    k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    include_metadata: bool = True
    use_reranking: bool = False
    include_vectors: bool = False
    config: Optional[Dict[str, Any]] = None  # Optional partial config overrides

class RetrieveResponse(BaseModel):
    query: str
    documents: List[Dict[str, Any]]
    processing_time: float
    collection_name: str
    total_found: int
