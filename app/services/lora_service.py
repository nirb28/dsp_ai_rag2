import logging
import os
import json
import uuid
import time
import asyncio
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import BackgroundTasks

# Import models from the dedicated models file
from app.model_schemas.lora_models import (
    LoRAJobRequest, LoRAJob, LoRAJobStatus, LoRAJobProgress,
    LoRATrainingConfig, LoRAGenerationRequest, LoRAGenerationResponse
)

logger = logging.getLogger(__name__)

class LoRAService:
    """Service for managing LoRA fine-tuning jobs."""

    def __init__(self):
        """Initialize the LoRA service."""
        self.jobs: Dict[str, LoRAJob] = {}
        self.running_jobs = set()
        self.canceled_jobs = set()
        self.job_executor = ThreadPoolExecutor(max_workers=1)  # Limit to 1 concurrent job due to CPU resources
        
        # Create necessary directories if they don't exist
        self.adapters_dir = Path("adapters")
        self.jobs_dir = Path("jobs")
        self.adapters_dir.mkdir(exist_ok=True)
        self.jobs_dir.mkdir(exist_ok=True)
        
        # Load existing jobs from disk
        self._load_jobs()

    def _load_jobs(self) -> None:
        """Load existing jobs from disk."""
        try:
            for job_file in self.jobs_dir.glob("*.json"):
                try:
                    with open(job_file, "r") as f:
                        job_data = json.load(f)
                    
                    # Convert string dates back to datetime objects
                    job_data["created_at"] = datetime.fromisoformat(job_data["created_at"])
                    job_data["updated_at"] = datetime.fromisoformat(job_data["updated_at"])
                    
                    job = LoRAJob(**job_data)
                    self.jobs[job.id] = job
                    
                    # Add to running_jobs if the job is still marked as running
                    # (might happen if the server was stopped during training)
                    if job.status == LoRAJobStatus.RUNNING:
                        # Set to FAILED since the job was interrupted
                        job.status = LoRAJobStatus.FAILED
                        job.error = "Job was interrupted due to server shutdown"
                        job.updated_at = datetime.now()
                        self._save_job(job)
                        
                except Exception as e:
                    logger.error(f"Error loading job from {job_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading jobs: {str(e)}")

    def _save_job(self, job: LoRAJob) -> None:
        """Save job to disk."""
        job_file = self.jobs_dir / f"{job.id}.json"
        try:
            # Convert job to dict and handle datetime conversion
            job_dict = job.dict()
            job_dict["created_at"] = job.created_at.isoformat()
            job_dict["updated_at"] = job.updated_at.isoformat()
            
            with open(job_file, "w") as f:
                json.dump(job_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving job {job.id}: {str(e)}")

    async def create_job(self, request: LoRAJobRequest, background_tasks: BackgroundTasks) -> LoRAJob:
        """Create and start a new LoRA fine-tuning job."""
        # Validate that base model exists or can be downloaded
        # This would be a more comprehensive check in production
        if not request.config.base_model:
            raise ValueError("Base model must be specified")
            
        # Validate that adapter name is unique
        adapter_path = self.adapters_dir / request.config.adapter_name
        if adapter_path.exists():
            raise ValueError(f"Adapter with name '{request.config.adapter_name}' already exists")
            
        # Validate training data
        if not request.config.training_data or len(request.config.training_data) == 0:
            raise ValueError("Training data must not be empty")
            
        for item in request.config.training_data:
            if "prompt" not in item or "completion" not in item:
                raise ValueError("Each training data item must have 'prompt' and 'completion' fields")

        # Create job
        job_id = str(uuid.uuid4())
        now = datetime.now()
        job = LoRAJob(
            id=job_id,
            name=request.name,
            config=request.config,
            status=LoRAJobStatus.PENDING,
            created_at=now,
            updated_at=now
        )
        
        # Save job to memory and disk
        self.jobs[job_id] = job
        self._save_job(job)
        
        # Start job in background
        background_tasks.add_task(self._run_job, job_id)
        
        return job

    async def _run_job(self, job_id: str) -> None:
        """Run a LoRA fine-tuning job in the background."""
        if job_id in self.running_jobs:
            logger.warning(f"Job {job_id} is already running")
            return
            
        self.running_jobs.add(job_id)
        job = self.jobs[job_id]
        
        try:
            # Update job status
            job.status = LoRAJobStatus.RUNNING
            job.updated_at = datetime.now()
            self._save_job(job)
            
            # Submit the training task to the executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.job_executor,
                self._train_model,
                job_id
            )
            
            # Check if the job was canceled
            if job_id in self.canceled_jobs:
                self.canceled_jobs.remove(job_id)
                job.status = LoRAJobStatus.CANCELED
                job.updated_at = datetime.now()
                self._save_job(job)
            else:
                # Job completed successfully
                job.status = LoRAJobStatus.COMPLETED
                job.updated_at = datetime.now()
                self._save_job(job)
                
        except Exception as e:
            logger.error(f"Error running job {job_id}: {str(e)}")
            # Update job status
            if job_id in self.canceled_jobs:
                self.canceled_jobs.remove(job_id)
                job.status = LoRAJobStatus.CANCELED
            else:
                job.status = LoRAJobStatus.FAILED
                job.error = str(e)
            
            job.updated_at = datetime.now()
            self._save_job(job)
        finally:
            self.running_jobs.discard(job_id)

    def _train_model(self, job_id: str) -> None:
        """Train a model using LoRA fine-tuning.
        
        This is the actual training implementation that runs in a thread.
        """
        job = self.jobs[job_id]
        config = job.config
        
        try:
            # Import libraries here to avoid loading them until needed
            import torch
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                DataCollatorForLanguageModeling
            )
            from peft import get_peft_model, LoraConfig, TaskType
            from datasets import Dataset
            
            # Prepare the output directory
            adapter_dir = self.adapters_dir / config.adapter_name
            adapter_dir.mkdir(exist_ok=True)
            
            # Update progress
            self._update_job_progress(job_id, {
                "message": "Loading base model and tokenizer...",
                "current_step": 0,
                "total_steps": 0
            })
            
            # Check for cancellation
            if job_id in self.canceled_jobs:
                logger.info(f"Job {job_id} was canceled before model loading")
                return
                
            # Load model and tokenizer
            try:
                # Determine model loading configuration based on user settings
                device_map = "auto"
                load_in_8bit = config.use_int8
                torch_dtype = torch.float32
                
                if config.use_fp16:
                    torch_dtype = torch.float16
                elif config.use_bf16:
                    torch_dtype = torch.bfloat16
                
                # Load the model with the appropriate configuration
                model = AutoModelForCausalLM.from_pretrained(
                    config.base_model,
                    device_map=device_map,
                    load_in_8bit=load_in_8bit,
                    torch_dtype=torch_dtype,
                )
                
                tokenizer = AutoTokenizer.from_pretrained(config.base_model)
                
                # Ensure padding token is set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                
            except Exception as e:
                raise RuntimeError(f"Failed to load model and tokenizer: {str(e)}")
                
            # Update progress
            self._update_job_progress(job_id, {
                "message": "Preparing training data...",
                "current_step": 0,
                "total_steps": 0
            })
            
            # Check for cancellation
            if job_id in self.canceled_jobs:
                logger.info(f"Job {job_id} was canceled before data preparation")
                return
            
            # Prepare training data
            # Format: instruction-tuning format combining prompt and completion
            def prepare_training_data(data_items):
                formatted_data = []
                
                for item in data_items:
                    # Instruction format
                    text = f"### Instruction: {item['prompt']}\n\n### Response: {item['completion']}"
                    formatted_data.append({"text": text})
                    
                return formatted_data
                
            # Create dataset
            dataset = Dataset.from_list(prepare_training_data(config.training_data))
            
            # Tokenize dataset
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=config.sequence_length,
                )
                
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                desc="Tokenizing dataset",
                remove_columns=["text"]
            )
            
            # Determine target modules
            target_modules = config.target_modules
            if not target_modules:
                # Default target modules based on model architecture
                if "llama" in config.base_model.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                else:
                    target_modules = ["query_key_value"]
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            
            # Apply LoRA config to the model
            model = get_peft_model(model, peft_config)
            
            # Calculate total steps
            num_examples = len(tokenized_dataset)
            steps_per_epoch = num_examples // config.batch_size
            if num_examples % config.batch_size != 0:
                steps_per_epoch += 1
                
            total_steps = steps_per_epoch * config.num_epochs
            
            # Update progress
            self._update_job_progress(job_id, {
                "message": "Starting training...",
                "current_step": 0,
                "total_steps": total_steps
            })
            
            # Check for cancellation
            if job_id in self.canceled_jobs:
                logger.info(f"Job {job_id} was canceled before training start")
                return
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=str(adapter_dir),
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                logging_steps=1,
                save_strategy="epoch",
                report_to="none",  # Disable wandb, tensorboard etc.
                fp16=config.use_fp16,
                bf16=config.use_bf16,
            )
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False  # We're doing causal language modeling, not masked
            )
            
            # Custom callback to track progress
            class ProgressCallback(Trainer.Callback):
                def __init__(self, job_id):
                    self.job_id = job_id
                    
                def on_step_end(self, args, state, control, **kwargs):
                    # Update progress every few steps
                    if state.global_step % 5 == 0 or state.global_step == 1:
                        job_service._update_job_progress(self.job_id, {
                            "current_step": state.global_step,
                            "total_steps": state.max_steps,
                            "loss": state.log_history[-1]["loss"] if state.log_history else None,
                            "learning_rate": state.log_history[-1]["learning_rate"] if state.log_history else None,
                            "epoch": state.epoch,
                        })
                        
                    # Check for cancellation
                    if self.job_id in job_service.canceled_jobs:
                        control.should_training_stop = True
                        
            # Save job_service reference for the callback
            job_service = self
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                callbacks=[ProgressCallback(job_id)]
            )
            
            # Train model
            trainer.train()
            
            # Check if job was canceled during training
            if job_id in self.canceled_jobs:
                logger.info(f"Job {job_id} was canceled during training")
                return
                
            # Save model
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            
            # Update progress
            self._update_job_progress(job_id, {
                "message": "Training completed, saving model...",
                "current_step": total_steps,
                "total_steps": total_steps
            })
            
            # Final progress update
            self._update_job_progress(job_id, {
                "message": "Fine-tuning completed successfully!",
                "current_step": total_steps,
                "total_steps": total_steps
            })
            
        except Exception as e:
            logger.error(f"Error in training job {job_id}: {str(e)}")
            self._update_job_progress(job_id, {
                "message": f"Error: {str(e)}",
            })
            raise

    def _update_job_progress(self, job_id: str, progress_update: Dict[str, Any]) -> None:
        """Update the progress of a job."""
        if job_id not in self.jobs:
            logger.warning(f"Attempted to update progress for non-existent job {job_id}")
            return
            
        job = self.jobs[job_id]
        
        # Update only the fields that are provided
        for key, value in progress_update.items():
            setattr(job.progress, key, value)
            
        job.updated_at = datetime.now()
        self._save_job(job)

    def list_jobs(self) -> List[LoRAJob]:
        """List all jobs."""
        return list(self.jobs.values())

    def get_job(self, job_id: str) -> Optional[LoRAJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> LoRAJob:
        """Cancel a running job."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job = self.jobs[job_id]
        
        if job.status != LoRAJobStatus.RUNNING and job.status != LoRAJobStatus.PENDING:
            raise ValueError(f"Job {job_id} is not running or pending, cannot cancel")
            
        # Mark job as to be canceled
        self.canceled_jobs.add(job_id)
        
        # If job is only pending and not yet running, update status directly
        if job.status == LoRAJobStatus.PENDING:
            job.status = LoRAJobStatus.CANCELED
            job.updated_at = datetime.now()
            self._save_job(job)
        else:
            # For running jobs, wait briefly for the training loop to detect cancellation
            for _ in range(5):  # Wait up to 5 seconds
                if job.status == LoRAJobStatus.CANCELED:
                    break
                await asyncio.sleep(1)
        
        return job

    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        if job_id not in self.jobs:
            return False
            
        job = self.jobs[job_id]
        
        # Cannot delete running jobs
        if job.status == LoRAJobStatus.RUNNING:
            return False
            
        # Delete job file
        job_file = self.jobs_dir / f"{job_id}.json"
        if job_file.exists():
            job_file.unlink()
            
        # Remove from memory
        self.jobs.pop(job_id)
        
        return True

    def list_adapters(self) -> List[str]:
        """List all available adapters."""
        return [d.name for d in self.adapters_dir.iterdir() if d.is_dir()]

    def delete_adapter(self, adapter_name: str) -> bool:
        """Delete an adapter."""
        adapter_dir = self.adapters_dir / adapter_name
        if not adapter_dir.exists():
            return False
            
        # Check if any job is using this adapter
        for job in self.jobs.values():
            if job.status == LoRAJobStatus.RUNNING and job.config.adapter_name == adapter_name:
                return False
                
        # Delete adapter directory
        shutil.rmtree(adapter_dir)
        
        return True

    async def generate_text(self, request: LoRAGenerationRequest) -> LoRAGenerationResponse:
        """Generate text using a fine-tuned model."""
        # Check if adapter exists
        adapter_dir = self.adapters_dir / request.adapter_name
        if not adapter_dir.exists():
            raise ValueError(f"Adapter {request.adapter_name} not found")
            
        # Import libraries here to avoid loading until needed
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from joblib import Memory
        
        # Set up caching to avoid reloading models
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        memory = Memory(str(cache_dir), verbose=0)
        
        # Cached model loading function
        @memory.cache
        def load_model_for_inference(adapter_name):
            # Find base model from job configuration
            base_model = None
            for job in self.jobs.values():
                if job.config.adapter_name == adapter_name:
                    base_model = job.config.base_model
                    break
                    
            if not base_model:
                # Try to load from adapter config
                adapter_config_file = adapter_dir / "adapter_config.json"
                if adapter_config_file.exists():
                    with open(adapter_config_file, "r") as f:
                        config_data = json.load(f)
                        base_model = config_data.get("base_model_name_or_path")
                        
            if not base_model:
                raise ValueError(f"Could not determine base model for adapter {adapter_name}")
                
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                load_in_8bit=True,  # Use 8-bit for inference to save memory
                torch_dtype=torch.float16,
            )
            
            # Load adapter
            model = PeftModel.from_pretrained(model, adapter_dir)
            
            return model, tokenizer, base_model
            
        # Generate text
        start_time = time.time()
        
        try:
            # Load model (will use cache if already loaded)
            model, tokenizer, base_model = load_model_for_inference(request.adapter_name)
            
            # Prepare prompt with instruction format
            formatted_prompt = f"### Instruction: {request.prompt}\n\n### Response:"
            
            # Set up generation parameters
            gen_kwargs = {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty or 1.0,
                "do_sample": request.temperature > 0.01,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            if request.stop_sequences:
                gen_kwargs["stopping_criteria"] = [
                    lambda ids, scores: any(tokenizer.decode(ids[-10:]).endswith(stop) for stop in request.stop_sequences)
                ]
                
            # Tokenize input
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)
                
            # Decode output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the response part
            response_prefix = "### Response:"
            if response_prefix in generated_text:
                generated_text = generated_text.split(response_prefix)[1].strip()
                
            # Calculate generation time
            generation_time = time.time() - start_time
            
            return LoRAGenerationResponse(
                generated_text=generated_text,
                adapter_name=request.adapter_name,
                generation_time=generation_time
            )
            
        except Exception as e:
            logger.error(f"Error generating text with adapter {request.adapter_name}: {str(e)}")
            raise
