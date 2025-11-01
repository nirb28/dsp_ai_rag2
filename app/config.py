import os
import json
import logging
import re
from enum import Enum
from typing import Dict, Any, Optional, List, Union, TypeVar, Type, Callable
from pydantic import BaseModel, Field, model_validator, validator, create_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Type variable for generic model types
T = TypeVar('T', bound=BaseModel)

# Regular expression to match environment variable patterns like ${VAR_NAME}
ENV_VAR_PATTERN = re.compile(r'\${([A-Za-z0-9_]+)}')

def resolve_env_vars(value: str) -> str:
    """Replace environment variable references in a string with their values.
    
    Args:
        value: String that may contain environment variable references like ${VAR_NAME}
        
    Returns:
        String with environment variables replaced with their values
    """
    if not isinstance(value, str):
        return value
        
    def replace_env_var(match):
        var_name = match.group(1)
        env_value = os.getenv(var_name)
        if env_value is None:
            logging.warning(f"Environment variable '{var_name}' not found in .env file")
            return match.group(0)  # Return the original ${VAR_NAME} if not found
        return env_value
        
    return ENV_VAR_PATTERN.sub(replace_env_var, value)

def process_env_vars_in_model(model: T) -> T:
    """Process all string fields in a Pydantic model to substitute environment variables.
    
    Args:
        model: A Pydantic model instance
        
    Returns:
        The same model with environment variables resolved in string fields
    """
    data = model.dict()
    
    # Process all string values in the model data
    for field_name, field_value in data.items():
        if isinstance(field_value, str):
            data[field_name] = resolve_env_vars(field_value)
        elif isinstance(field_value, dict):
            # Handle nested dictionaries
            for k, v in field_value.items():
                if isinstance(v, str):
                    field_value[k] = resolve_env_vars(v)
        elif isinstance(field_value, BaseModel):
            # Handle nested models
            data[field_name] = process_env_vars_in_model(field_value)
    
    # Create a new model instance with the processed data
    return model.__class__(**data)

class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE_TEXT = "recursive_text"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"

class VectorStore(str, Enum):
    FAISS = "faiss"
    REDIS = "redis"
    BM25 = "bm25"  # Keyword-based search using BM25 algorithm
    NEO4J_KNOWLEDGE_GRAPH = "neo4j_knowledge_graph"  # True knowledge graph using LangGraph
    ELASTICSEARCH = "elasticsearch"  # Elasticsearch vector store

class EmbeddingModel(str, Enum):
    SENTENCE_TRANSFORMERS_ALL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_TRANSFORMERS_ALL_MPNET = "sentence-transformers/all-mpnet-base-v2"
    OPENAI_TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TRITON_EMBEDDING = "triton-embedding"  # Add Triton embedding model
    LOCAL_MODEL_SERVER = "local-model-server"  # Local model server endpoint

class LLMProvider(str, Enum):
    """The provider of the LLM service"""
    GROQ = "groq"
    TRITON = "triton"
    OPENAI_COMPATIBLE = "openai_compatible"  # For locally deployed OpenAI-compatible endpoints

# Common model names for reference - not used for validation
COMMON_MODELS = {
    "groq": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
    "triton": ["Llama-3.1-70B-Instruct", "llama3-vllm"],
    "openai_compatible": ["gpt-3.5-turbo", "gpt-4", "llama3", "mistral", "mixtral"]  # Common models for OpenAI-compatible APIs
}
    
class RerankerModel(str, Enum):
    """Models that can be used for reranking retrieved chunks via endpoints."""
    NONE = "none"  # No reranking
    COHERE_RERANK = "cohere-rerank"  # Cohere Rerank API
    LOCAL_MODEL_SERVER = "local-model-server"  # Local model server endpoint

class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_TEXT
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    separators: Optional[list[str]] = None

class VectorStoreConfig(BaseModel):
    type: VectorStore = VectorStore.FAISS
    index_path: str = Field(default="./storage/faiss_index", description="Path for FAISS index files")
    dimension: int = Field(default=384, ge=1, le=4096)
    # Redis specific settings
    redis_host: str = Field(default="localhost", description="Redis server hostname")
    redis_port: int = Field(default=10000, description="Redis server port")
    redis_username: Optional[str] = Field(default=None, description="Redis username if authentication is required")
    redis_password: Optional[str] = Field(default=None, description="Redis password if authentication is required")
    redis_index_name: str = Field(default="document-index", description="Redis search index name")
    # Neo4j specific settings
    neo4j_uri: str = Field(default="neo4j://localhost:7687", description="Neo4j server URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: Optional[str] = Field(default=None, description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    # Knowledge graph specific settings
    kg_llm_config_name: Optional[str] = Field(default=None, description="LLM configuration name for knowledge graph extraction")
    # Elasticsearch specific settings
    es_url: str = Field(default="http://localhost:9200", description="Elasticsearch server URL")
    es_index_name: str = Field(default="documents", description="Elasticsearch index name")
    es_user: Optional[str] = Field(default=None, description="Elasticsearch username")
    es_password: Optional[str] = Field(default=None, description="Elasticsearch password")
    es_api_key: Optional[str] = Field(default=None, description="Elasticsearch API key (alternative to username/password)")
    es_api_key_id: Optional[str] = Field(default=None, description="Elasticsearch API key ID (optional, for key identification)")
    es_use_index_suffix: bool = Field(default=True, description="Whether to append configuration name as suffix to index name")
    es_search_type: str = Field(default="vector", description="Search type: fulltext, vector, semantic, hybrid, or query_dsl")
    es_fulltext_field: str = Field(default="content", description="Field name for fulltext/BM25 search")
    es_semantic_field: str = Field(default="text", description="Field name for semantic search (content_vector or ml.tokens)")
    es_semantic_inference_id: str = Field(default=".elser_model_2", description="Inference ID for semantic search model")
    es_query_dsl_template: Optional[Dict[str, Any]] = Field(default=None, description="Custom Elasticsearch Query DSL template with $QUERY$ placeholder")
    # Score normalization setting
    normalize_similarity_scores: bool = Field(default=False, description="Whether to normalize similarity scores to 0-1 range")

class EmbeddingConfig(BaseModel):
    model: Union[EmbeddingModel, str] = EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM
    batch_size: int = Field(default=32, ge=1, le=128)
    server_url: Optional[str] = Field(default="http://localhost:8000", description="URL for the model server or inference server")
    
    @validator('model')
    def validate_model(cls, v):
        # If it's already an enum value, return it
        if isinstance(v, EmbeddingModel):
            return v
        
        # If it's a string, try to convert to enum or keep as string
        try:
            return EmbeddingModel(v)
        except ValueError:
            # If not a valid enum value, allow it as a custom string with a warning
            logging.warning(f"Using custom embedding model name: {v} (not in EmbeddingModel enum)")
            return v

class RerankerConfig(BaseModel):
    """Configuration for the reranking step in the retrieval process."""
    enabled: bool = Field(default=False, description="Whether to use reranking")
    model: Union[RerankerModel, str] = Field(default=RerankerModel.NONE, description="Reranker model to use")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of initial results to rerank")
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Min reranking score to include")
    server_url: Optional[str] = Field(default="http://localhost:9001", description="URL for the model server or inference server")
    
    @validator('model')
    def validate_model(cls, v):
        # If it's already an enum value, return it
        if isinstance(v, RerankerModel):
            return v
        
        # If it's a string, try to convert to enum or keep as string
        try:
            return RerankerModel(v)
        except ValueError:
            # If not a valid enum value, allow it as a custom string with a warning
            logging.warning(f"Using custom reranker model name: {v} (not in RerankerModel enum)")
            return v
    
    @model_validator(mode='after')
    def validate_reranking(self):
        """If reranking is enabled, model must not be NONE."""
        if self.enabled and self.model == RerankerModel.NONE:
            raise ValueError("If reranking is enabled, a reranker model must be selected")
        return self

class GenerationConfig(BaseModel):
    model: str = Field(default="llama3-8b-8192", description="Model name as a string - not restricted to enum values")
    provider: LLMProvider = Field(default=LLMProvider.GROQ, description="The LLM provider type")
    api_key: Optional[str] = Field(default=None, description="API key for the provider (if required)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    server_url: Optional[str] = Field(default="http://localhost:8000", description="URL for the inference server")
    system_prompt: Optional[str] = Field(default=None, description="Default system prompt for the LLM")
    
    @model_validator(mode='after')
    def validate_provider(self):
        """Ensure provider is valid"""
        if self.provider not in [LLMProvider.GROQ, LLMProvider.TRITON, LLMProvider.OPENAI_COMPATIBLE]:
            raise ValueError(f"Invalid provider: {self.provider}. Must be one of: {', '.join([p.value for p in LLMProvider])}")
        return self

class LLMConfig(BaseModel):
    """Configuration for LLM used in query expansion."""
    name: str = Field(..., description="Unique name for this LLM configuration")
    provider: LLMProvider = Field(..., description="The LLM provider type")
    model: str = Field(..., description="Model name")
    endpoint: str = Field(..., description="API endpoint URL")
    api_key: Optional[str] = Field(default=None, description="API key for the provider (if required)")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for query expansion")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Top-k sampling")
    timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")
    
    @model_validator(mode='after')
    def validate_provider(self):
        """Ensure provider is valid"""
        if self.provider not in [LLMProvider.GROQ, LLMProvider.TRITON, LLMProvider.OPENAI_COMPATIBLE]:
            raise ValueError(f"Invalid provider: {self.provider}. Must be one of: {', '.join([p.value for p in LLMProvider])}")
        return self

class MCPProtocol(str, Enum):
    """Available MCP server protocols."""
    HTTP = "http"  # HTTP JSON-RPC
    SSE = "sse"  # Server-Sent Events
    STDIO = "stdio"  # Standard Input/Output

class MCPToolType(str, Enum):
    """Available MCP tool types."""
    RETRIEVE = "retrieve"  # Document retrieval tool

class MCPToolConfig(BaseModel):
    """Configuration for individual MCP tools."""
    type: MCPToolType = Field(..., description="Type of the MCP tool")
    enabled: bool = Field(default=True, description="Whether this tool is enabled")
    name: str = Field(..., description="Tool name exposed to MCP clients")
    description: str = Field(..., description="Tool description for LLMs")
    parameters_schema: Optional[Dict[str, Any]] = Field(default=None, description="Custom JSON schema for tool parameters")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Default similarity threshold for this tool")
    include_metadata: bool = Field(default=True, description="Whether to include document metadata in results")
    
class MCPServerConfig(BaseModel):
    """Configuration for MCP server settings."""
    enabled: bool = Field(default=False, description="Whether to enable MCP server for this configuration")
    startup_enabled: bool = Field(default=False, description="Whether to automatically start MCP server when configuration is loaded")
    name: str = Field(..., description="Human-readable name for this MCP server instance")
    description: str = Field(default="RAG document retrieval server", description="Description of the MCP server")
    
    # Protocol settings
    protocols: List[MCPProtocol] = Field(default=[MCPProtocol.HTTP], description="Enabled MCP protocols")
    http_host: str = Field(default="localhost", description="HTTP server host")
    http_port: int = Field(default=8080, ge=1024, le=65535, description="HTTP server port")
    sse_path: str = Field(default="/sse", description="SSE endpoint path")
    
    # Tool configuration
    tools: List[MCPToolConfig] = Field(default_factory=list, description="Available MCP tools")
    
    # Server metadata
    version: str = Field(default="1.0.0", description="MCP server version")
    author: str = Field(default="DSP AI RAG", description="Server author")
    license: Optional[str] = Field(default="MIT", description="Server license")
    
    # Security inheritance
    inherit_security: bool = Field(default=True, description="Whether to inherit security settings from RAG configuration")
    
    @model_validator(mode='after')
    def validate_protocols(self):
        """Ensure at least one protocol is enabled."""
        if not self.protocols or len(self.protocols) == 0:
            raise ValueError("At least one MCP protocol must be enabled")
        return self
    
    @model_validator(mode='after') 
    def validate_tools(self):
        """Ensure at least one tool is configured."""
        if not self.tools or len(self.tools) == 0:
            # Set default tools if none specified
            self.tools = [
                MCPToolConfig(
                    type=MCPToolType.RETRIEVE,
                    name="retrieve_documents",
                    description="Retrieve relevant documents based on a query"
                )
            ]
        return self

class SecurityType(str, Enum):
    """Available security authentication types."""
    JWT_BEARER = "jwt_bearer"  # JWT Bearer token authentication
    API_KEY = "api_key"  # API Key authentication (future)
    OAUTH2 = "oauth2"  # OAuth2 authentication (future)

class QueryExpansionStrategy(str, Enum):
    """Available query expansion strategies."""
    FUSION = "fusion"  # Generate multiple variations and fuse results
    MULTI_QUERY = "multi_query"  # Generate multiple related queries

class QueryExpansionConfig(BaseModel):
    """Configuration for query expansion in requests."""
    enabled: bool = Field(default=True, description="Whether to enable query expansion")
    strategy: QueryExpansionStrategy = Field(default=QueryExpansionStrategy.FUSION, description="Query expansion strategy")
    llm_config_name: str = Field(..., description="Name of the LLM configuration to use")
    num_queries: int = Field(default=3, ge=1, le=10, description="Number of expanded queries to generate")

class SecurityConfig(BaseModel):
    """Configuration for security authentication."""
    enabled: bool = Field(default=False, description="Whether to enable security authentication")
    type: SecurityType = Field(default=SecurityType.JWT_BEARER, description="Type of security authentication")
    
    # JWT Bearer token configuration
    jwt_secret_key: Optional[str] = Field(default=None, description="Secret key for JWT token validation")
    jwt_algorithm: str = Field(default="HS256", description="Algorithm used for JWT token validation")
    jwt_issuer: Optional[str] = Field(default=None, description="Expected issuer of JWT tokens")
    jwt_audience: Optional[str] = Field(default=None, description="Expected audience of JWT tokens")
    jwt_require_exp: bool = Field(default=True, description="Whether to require expiration claim in JWT")
    jwt_require_iat: bool = Field(default=True, description="Whether to require issued at claim in JWT")
    jwt_leeway: int = Field(default=0, ge=0, description="Leeway in seconds for token expiration validation")
    
    # API Key configuration (for future use)
    api_key_header: str = Field(default="X-API-Key", description="Header name for API key authentication")
    api_keys: Optional[List[str]] = Field(default=None, description="List of valid API keys")
    
    @model_validator(mode='after')
    def validate_security_config(self):
        """Validate security configuration based on type."""
        if self.enabled:
            if self.type == SecurityType.JWT_BEARER:
                if not self.jwt_secret_key:
                    raise ValueError("jwt_secret_key is required when JWT Bearer authentication is enabled")
            elif self.type == SecurityType.API_KEY:
                if not self.api_keys or len(self.api_keys) == 0:
                    raise ValueError("api_keys list is required when API Key authentication is enabled")
        return self

class RAGConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    retrieval_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # New features
    reranking: Optional[RerankerConfig] = Field(default_factory=RerankerConfig)
    security: Optional[SecurityConfig] = Field(default_factory=SecurityConfig)
    mcp_server: Optional[MCPServerConfig] = Field(default=None, description="MCP server configuration")

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")  # Added for Cohere reranking
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./storage")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    ALLOWED_FILE_TYPES: list[str] = ["pdf", "txt", "docx", "pptx"]
    # Model loading settings
    LOCAL_MODELS_PATH: str = os.getenv("LOCAL_MODELS_PATH", "./models")
    # Triton model settings
    TRITON_LLM_MODEL: str = os.getenv("TRITON_LLM_MODEL", "llama3-vllm")
    # Model server settings
    MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER_URL", "http://localhost:9001")
    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = Settings()

# Configuration is now handled dynamically without predefined presets
