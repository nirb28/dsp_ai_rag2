from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, model_validator
import os
from dotenv import load_dotenv

load_dotenv()

class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE_TEXT = "recursive_text"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"

class VectorStore(str, Enum):
    FAISS = "faiss"
    REDIS = "redis"

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
    """Models that can be used for reranking retrieved chunks."""
    NONE = "none"  # No reranking
    COHERE_RERANK = "cohere-rerank"  # Cohere Rerank API
    BGE_RERANKER = "bge-reranker-large"  # BGE Reranker
    SENTENCE_TRANSFORMERS_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # SentenceTransformers cross-encoder

class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_TEXT
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    separators: Optional[list[str]] = None

class VectorStoreConfig(BaseModel):
    type: VectorStore = VectorStore.FAISS
    index_path: str = Field(default="./storage/faiss_index", description="Path for FAISS index files")
    dimension: int = Field(default=384, ge=128, le=1536, description="Embedding dimension")
    # Redis specific configuration
    redis_host: str = Field(default="localhost", description="Redis server hostname")
    redis_port: int = Field(default=6379, description="Redis server port")
    redis_password: Optional[str] = Field(default=None, description="Redis password if authentication is required")
    redis_index_name: str = Field(default="document-index", description="Redis search index name")

class EmbeddingConfig(BaseModel):
    model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM
    batch_size: int = Field(default=32, ge=1, le=128)
    server_url: Optional[str] = Field(default="http://localhost:8000", description="URL for the model server or inference server")

class RerankerConfig(BaseModel):
    """Configuration for the reranking step in the retrieval process."""
    enabled: bool = Field(default=False, description="Whether to use reranking")
    model: RerankerModel = Field(default=RerankerModel.NONE, description="Reranker model to use")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of initial results to rerank")
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Min reranking score to include")
    
    @model_validator(mode='after')
    def validate_reranking(self):
        """If reranking is enabled, model must not be NONE."""
        if self.enabled and self.model == RerankerModel.NONE:
            raise ValueError("If reranking is enabled, a reranker model must be selected")
        return self

class GenerationConfig(BaseModel):
    model: str = Field(default="llama3-8b-8192", description="Model name as a string - not restricted to enum values")
    provider: LLMProvider = Field(default=LLMProvider.GROQ, description="The LLM provider type")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    server_url: Optional[str] = Field(default="http://localhost:8000", description="URL for the inference server")
    
    @model_validator(mode='after')
    def validate_provider(self):
        """Ensure provider is valid"""
        if self.provider not in [LLMProvider.GROQ, LLMProvider.TRITON, LLMProvider.OPENAI_COMPATIBLE]:
            raise ValueError(f"Invalid provider: {self.provider}. Must be one of: {', '.join([p.value for p in LLMProvider])}")
        return self

class RAGConfig(BaseModel):
    configuration_name: str = Field(default="default", min_length=1)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    retrieval_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # New features
    reranking: Optional[RerankerConfig] = Field(default_factory=RerankerConfig)

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
    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = Settings()

# Configuration is now handled dynamically without predefined presets
