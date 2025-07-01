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

class EmbeddingModel(str, Enum):
    SENTENCE_TRANSFORMERS_ALL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_TRANSFORMERS_ALL_MPNET = "sentence-transformers/all-mpnet-base-v2"
    OPENAI_TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TRITON_EMBEDDING = "triton-embedding"  # Add Triton embedding model
    LOCAL_MODEL_SERVER = "local-model-server"  # Local model server endpoint

class GenerationModel(str, Enum):
    GROQ_LLAMA3_8B = "llama3-8b-8192"
    GROQ_LLAMA3_70B = "llama3-70b-8192"
    GROQ_MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GROQ_GEMMA_7B = "gemma-7b-it"
    TRITON_LLAMA_3_70B = "meta-llama Llama-3.1-70B-Instruct"  # Add Triton LLM model
    
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
    index_path: str = Field(default="./storage/faiss_index")
    dimension: int = Field(default=384, ge=128, le=1536)

class EmbeddingConfig(BaseModel):
    model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM
    batch_size: int = Field(default=32, ge=1, le=128)
    model_server_url: Optional[str] = Field(default=None, description="URL for the local model server")

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

class ContextInjectionConfig(BaseModel):
    """Configuration for injecting additional context into prompts."""
    enabled: bool = Field(default=False, description="Whether to use context injection")
    max_items: int = Field(default=5, ge=1, le=20, description="Maximum context items to include")
    max_tokens_per_item: int = Field(default=500, ge=50, le=2000, description="Max tokens per context item")
    position: str = Field(default="before_query", description="Where to inject context in the prompt")
    separator: str = Field(default="\n\n", description="Separator between context items")
    context_prefix: str = Field(default="Context: ", description="Prefix for the injected context")
    
    # Optional field for static context that should always be included
    static_context: Optional[str] = Field(default=None, description="Static context to always inject")

class GenerationConfig(BaseModel):
    model: GenerationModel = GenerationModel.GROQ_LLAMA3_8B
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)

class RAGConfig(BaseModel):
    collection_name: str = Field(default="default", min_length=1)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    retrieval_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # New features
    reranking: Optional[RerankerConfig] = Field(default_factory=RerankerConfig)
    context_injection: Optional[ContextInjectionConfig] = Field(default_factory=ContextInjectionConfig)

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")  # Added for Cohere reranking
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./storage")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    ALLOWED_FILE_TYPES: list[str] = ["pdf", "txt", "docx", "pptx"]
    # Model loading settings
    PREFER_LOCAL_MODELS: bool = os.getenv("PREFER_LOCAL_MODELS", "True").lower() == "true"
    LOCAL_MODELS_PATH: str = os.getenv("LOCAL_MODELS_PATH", "./models")
    # Local model server settings
    MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")
    # Triton server settings
    TRITON_SERVER_URL: str = os.getenv("TRITON_SERVER_URL", "http://localhost:8000")
    TRITON_EMBEDDING_MODEL: str = os.getenv("TRITON_EMBEDDING_MODEL", "embedding-model")
    TRITON_LLM_MODEL: str = os.getenv("TRITON_LLM_MODEL", "meta-llama Llama-3.1-70B-Instruct")

settings = Settings()

# Default configurations for quick setup
DEFAULT_CONFIGS = {
    "fast_processing": RAGConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=500,
            chunk_overlap=50
        ),
        embedding=EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM,
            batch_size=64
        ),
        generation=GenerationConfig(
            model=GenerationModel.GROQ_LLAMA3_8B,
            temperature=0.5,
            max_tokens=512
        ),
        reranking=RerankerConfig(enabled=False),
        context_injection=ContextInjectionConfig(enabled=False)
    ),
    "high_quality": RAGConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_TEXT,
            chunk_size=1500,
            chunk_overlap=300
        ),
        embedding=EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MPNET,
            batch_size=16
        ),
        generation=GenerationConfig(
            model=GenerationModel.GROQ_LLAMA3_70B,
            temperature=0.3,
            max_tokens=2048,
            top_k=40
        ),
        reranking=RerankerConfig(
            enabled=True,
            model=RerankerModel.SENTENCE_TRANSFORMERS_CROSS_ENCODER,
            top_n=15,
            score_threshold=0.2
        ),
        context_injection=ContextInjectionConfig(enabled=False)
    ),
    "balanced": RAGConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_TEXT,
            chunk_size=1000,
            chunk_overlap=200
        ),
        embedding=EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM,
            batch_size=32
        ),
        generation=GenerationConfig(
            model=GenerationModel.GROQ_MIXTRAL_8X7B,
            temperature=0.7,
            max_tokens=1024
        ),
        reranking=RerankerConfig(enabled=False),
        context_injection=ContextInjectionConfig(enabled=False)
    ),
    "with_context": RAGConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_TEXT,
            chunk_size=1000,
            chunk_overlap=200
        ),
        embedding=EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM,
            batch_size=32
        ),
        generation=GenerationConfig(
            model=GenerationModel.GROQ_MIXTRAL_8X7B,
            temperature=0.7,
            max_tokens=1024
        ),
        reranking=RerankerConfig(
            enabled=True,
            model=RerankerModel.BGE_RERANKER
        ),
        context_injection=ContextInjectionConfig(
            enabled=True,
            max_items=3,
            max_tokens_per_item=300,
            position="before_query",
            context_prefix="Previous conversation: ",
            separator="\n"
        )
    )
}
