from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime

class LoRAJobStatus(str, Enum):
    """Status of a LoRA fine-tuning job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

class LoRATrainingConfig(BaseModel):
    """Configuration for LoRA fine-tuning."""
    base_model: str = Field(..., description="Base model to fine-tune (Hugging Face model ID)")
    adapter_name: str = Field(..., description="Name for the LoRA adapter")
    training_data: List[Dict[str, str]] = Field(
        ..., 
        description="List of prompt-completion pairs for training",
        example=[{"prompt": "What is machine learning?", "completion": "Machine learning is..."}]
    )
    lora_r: int = Field(8, description="LoRA attention dimension")
    lora_alpha: int = Field(16, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.05, description="LoRA dropout rate")
    batch_size: int = Field(4, description="Training batch size")
    learning_rate: float = Field(3e-4, description="Learning rate")
    num_epochs: int = Field(3, description="Number of training epochs")
    sequence_length: int = Field(512, description="Maximum sequence length for training")
    target_modules: Optional[List[str]] = Field(
        None, 
        description="List of modules to apply LoRA to (e.g., ['q_proj', 'v_proj'])"
    )
    use_fp16: bool = Field(False, description="Use FP16 precision")
    use_bf16: bool = Field(False, description="Use BF16 precision")
    use_int8: bool = Field(False, description="Use INT8 quantization")

class LoRAJobRequest(BaseModel):
    """Request to create a LoRA fine-tuning job."""
    name: str = Field(..., description="User-provided name for the job")
    config: LoRATrainingConfig = Field(..., description="Training configuration")

class LoRAJobProgress(BaseModel):
    """Progress information for a LoRA fine-tuning job."""
    current_step: int = Field(0, description="Current training step")
    total_steps: int = Field(0, description="Total number of training steps")
    loss: Optional[float] = Field(None, description="Current training loss")
    learning_rate: Optional[float] = Field(None, description="Current learning rate")
    epoch: Optional[float] = Field(None, description="Current epoch")
    message: Optional[str] = Field(None, description="Status message")

class LoRAJob(BaseModel):
    """LoRA fine-tuning job."""
    id: str = Field(..., description="Unique job ID")
    name: str = Field(..., description="User-provided job name")
    config: LoRATrainingConfig = Field(..., description="Training configuration")
    status: LoRAJobStatus = Field(..., description="Current job status")
    progress: LoRAJobProgress = Field(default_factory=LoRAJobProgress, description="Job progress")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Job last updated timestamp")
    error: Optional[str] = Field(None, description="Error message if job failed")

class LoRAJobResponse(BaseModel):
    """Response for LoRA job operations."""
    job_id: str = Field(..., description="Job ID")
    name: str = Field(..., description="Job name")
    status: LoRAJobStatus = Field(..., description="Job status")
    progress: LoRAJobProgress = Field(..., description="Job progress")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Job last updated timestamp")
    message: str = Field(..., description="Response message")
    error: Optional[str] = Field(None, description="Error message if any")

class LoRAJobListResponse(BaseModel):
    """Response for listing LoRA jobs."""
    jobs: List[LoRAJob] = Field(..., description="List of jobs")
    total_count: int = Field(..., description="Total number of jobs")

class LoRAJobStatusUpdate(BaseModel):
    """Status update for a LoRA fine-tuning job."""
    job_id: str = Field(..., description="Job ID")
    status: LoRAJobStatus = Field(..., description="New job status")
    progress: Optional[LoRAJobProgress] = Field(None, description="Updated progress")
    error: Optional[str] = Field(None, description="Error message if job failed")

class LoRAGenerationRequest(BaseModel):
    """Request for text generation using a fine-tuned model."""
    adapter_name: str = Field(..., description="Name of the LoRA adapter to use")
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(100, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.0, description="Repetition penalty")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")

class LoRAGenerationResponse(BaseModel):
    """Response for text generation using a fine-tuned model."""
    generated_text: str = Field(..., description="Generated text")
    adapter_name: str = Field(..., description="Adapter used for generation")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
