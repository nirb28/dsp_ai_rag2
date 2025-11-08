# Model schemas package initialization
# Import and re-export models for easier imports

from .base_models import (
    DocumentStatus, Document, DocumentUploadRequest, DocumentUploadResponse,
    ContextItem, QueryExpansionRequest, QueryRequest, QueryResponse,
    ConfigurationRequest, ConfigurationResponse, DeleteConfigurationResponse,
    HealthResponse, ErrorResponse, ConfigurationInfo, ConfigurationsResponse,
    ConfigurationNamesResponse, RetrieveRequest, RetrieveResponse,
    TextDocument, TextDocumentsUploadRequest, TextDocumentsUploadResponse,
    DuplicateConfigurationRequest, DuplicateConfigurationResponse,
    LLMConfigRequest, LLMConfigResponse, LLMConfigListResponse
)

# Import models from submodules
from .model_server_models import (
    EmbeddingRequest, EmbeddingResponse,
    RerankerRequest, RerankerResponse,
    ClassificationRequest, ClassificationResult, ClassificationResponse
)

from .lora_models import (
    LoRAJobRequest, LoRAJob, LoRAJobStatus, LoRAJobProgress,
    LoRATrainingConfig, LoRAGenerationRequest, LoRAGenerationResponse,
    LoRAJobResponse, LoRAJobListResponse
)

from .openai_models import (
    ChatMessage, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionChoice, ChatCompletionChunk, UsageInfo,
    ErrorDetail, ErrorResponse as OpenAIErrorResponse
)
