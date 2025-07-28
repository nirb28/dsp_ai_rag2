import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Path, Query

from app.model_schemas.lora_models import (
    LoRAJobRequest, LoRAJob, LoRAJobResponse, LoRAJobListResponse,
    LoRAGenerationRequest, LoRAGenerationResponse, LoRAJobStatus
)
from app.services.lora_service import LoRAService

logger = logging.getLogger(__name__)
router = APIRouter()

# Create LoRA service instance
lora_service = LoRAService()


@router.post("/jobs", response_model=LoRAJobResponse)
async def create_job(
    request: LoRAJobRequest,
    background_tasks: BackgroundTasks
):
    """Create a new LoRA fine-tuning job.
    
    The job will be started in the background and can be monitored through the /jobs/{job_id} endpoint.
    """
    try:
        job = await lora_service.create_job(request, background_tasks)
        return LoRAJobResponse(
            job_id=job.id,
            name=job.name,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            updated_at=job.updated_at,
            message="Job created successfully and started in background"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/jobs", response_model=LoRAJobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status")
):
    """List all LoRA fine-tuning jobs."""
    try:
        jobs = lora_service.list_jobs()
        
        # Filter by status if provided
        if status:
            try:
                status_enum = LoRAJobStatus(status)
                jobs = [job for job in jobs if job.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        # Sort by creation date (newest first)
        jobs = sorted(jobs, key=lambda x: x.created_at, reverse=True)
        
        return LoRAJobListResponse(
            jobs=jobs,
            total_count=len(jobs)
        )
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/jobs/{job_id}", response_model=LoRAJob)
async def get_job(
    job_id: str = Path(..., description="ID of the LoRA fine-tuning job")
):
    """Get details of a specific LoRA fine-tuning job."""
    job = lora_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.delete("/jobs/{job_id}", response_model=LoRAJobResponse)
async def delete_job(
    job_id: str = Path(..., description="ID of the LoRA fine-tuning job")
):
    """Delete a LoRA fine-tuning job."""
    job = lora_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    success = lora_service.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete job {job_id}")
    
    return LoRAJobResponse(
        job_id=job_id,
        name=job.name,
        status=LoRAJobStatus.CANCELED,
        progress=job.progress,
        created_at=job.created_at,
        updated_at=job.updated_at,
        message=f"Job {job_id} deleted successfully"
    )


@router.post("/jobs/{job_id}/cancel", response_model=LoRAJobResponse)
async def cancel_job(
    job_id: str = Path(..., description="ID of the LoRA fine-tuning job")
):
    """Cancel a running LoRA fine-tuning job."""
    try:
        job = await lora_service.cancel_job(job_id)
        return LoRAJobResponse(
            job_id=job.id,
            name=job.name,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            updated_at=job.updated_at,
            message=f"Job {job_id} canceled successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error canceling job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/adapters", response_model=List[str])
async def list_adapters():
    """List all available LoRA adapters."""
    return lora_service.list_adapters()


@router.delete("/adapters/{adapter_name}")
async def delete_adapter(
    adapter_name: str = Path(..., description="Name of the LoRA adapter to delete")
):
    """Delete a LoRA adapter."""
    success = lora_service.delete_adapter(adapter_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Adapter {adapter_name} not found")
    return {"message": f"Adapter {adapter_name} deleted successfully"}


@router.post("/generate", response_model=LoRAGenerationResponse)
async def generate_text(request: LoRAGenerationRequest):
    """Generate text using a fine-tuned model."""
    try:
        response = await lora_service.generate_text(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
