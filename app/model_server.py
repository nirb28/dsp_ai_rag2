import os
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import dotenv
import json
import shutil
import tempfile
import time
from collections import defaultdict

import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Query
from app.api.lora_endpoints import router as lora_router
from fastapi.middleware.cors import CORSMiddleware

# Import models from dedicated models file
from app.model_schemas.model_server_models import (
    EmbeddingRequest, EmbeddingResponse,
    RerankerRequest, RerankerResponse,
    ClassificationRequest, ClassificationResult, ClassificationTextResult, ClassificationResponse,
    ClassificationEvalRequest, ClassificationMetrics, ClassificationEvalResponse
)

from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config import settings

# Load environment variables directly
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Model Serving API",
    description="API for serving ML models that aren't compatible with vLLM",
    version="1.0.0"
)

# Define placeholder for routers
router_config = []

# We'll include the routers after they've been fully defined to avoid circular imports
app.include_router(lora_router, prefix="/api/v1/lora", tags=["LoRA Fine-Tuning"])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for loaded models
loaded_models: Dict[str, Union[SentenceTransformer, CrossEncoder, Any]] = {}

# Temp directory for model copies
TEMP_MODEL_DIR = Path(tempfile.gettempdir()) / "rag_models"
if not TEMP_MODEL_DIR.exists():
    os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

# Model base directory - load directly from .env if available, otherwise use default
MODEL_DIR = Path(os.getenv('LOCAL_MODELS_PATH', './models'))



def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load or retrieve embedding model from cache."""
    model_key = f"embedding_{model_name}"
    
    if model_key not in loaded_models:
        logger.info(f"Loading embedding model: {model_name}")
        try:
            # Check if model exists locally using the full model path structure
            # Convert any path separators to be compatible with local filesystem
            model_relative_path = model_name.replace('/', os.sep)
            local_model_path = MODEL_DIR / model_relative_path
            
            # Also try with just the base name as a fallback for backward compatibility
            local_model_base_path = MODEL_DIR / model_name.split('/')[-1]
            
            # Helper function to create a clean model directory and load from it
            def copy_and_load_model(source_path):
                # Create a safe name for the temp directory
                safe_name = f"embed_{int(time.time())}_{hash(str(source_path)) % 1000}"
                temp_model_path = TEMP_MODEL_DIR / safe_name
                
                if temp_model_path.exists():
                    shutil.rmtree(temp_model_path)
                os.makedirs(temp_model_path, exist_ok=True)
                
                # Copy model files to temp directory
                logger.info(f"Copying model files from {source_path} to {temp_model_path}")
                for item in os.listdir(source_path):
                    s = os.path.join(source_path, item)
                    d = os.path.join(temp_model_path, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
                
                # Load model from the clean temp directory
                logger.info(f"Loading model from clean directory: {temp_model_path}")
                return SentenceTransformer(str(temp_model_path), device='cpu', local_files_only=True)
            
            # Check if local paths exist
            if local_model_path.exists():
                logger.info(f"Found model in nested path: {local_model_path}")
                model = copy_and_load_model(local_model_path)
                logger.info(f"Successfully loaded model from local nested path: {local_model_path}")
            elif local_model_base_path.exists():
                logger.info(f"Found model in base path: {local_model_base_path}")
                model = copy_and_load_model(local_model_base_path)
                logger.info(f"Successfully loaded model from local base path: {local_model_base_path}")
            else:
                error_msg = f"Embedding model '{model_name}' not available locally. Please download it first."
                logger.error(error_msg)
                raise HTTPException(status_code=404, detail=error_msg)
            
            loaded_models[model_key] = model
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return loaded_models[model_key]


def get_reranker_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    """Load or retrieve reranker model from cache."""
    model_key = f"reranker_{model_name}"
    
    # Note: classification can use the same models as reranking
    # so we can use the same function for both
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    if model_key not in loaded_models:
        logger.info(f"Loading reranker model: {model_name}")
        try:
            # Check if model exists locally using the full model path structure
            # Convert any path separators to be compatible with local filesystem
            model_relative_path = model_name.replace('/', os.sep)
            local_model_path = MODEL_DIR / model_relative_path
            
            # Also try with just the base name as a fallback for backward compatibility
            local_model_base_path = MODEL_DIR / model_name.split('/')[-1]
            
            # Helper function to create a clean model directory and load from it
            def copy_and_load_model(source_path):
                # Create a safe name for the temp directory
                safe_name = f"reranker_{int(time.time())}_{hash(str(source_path)) % 1000}"
                temp_model_path = TEMP_MODEL_DIR / safe_name
                
                if temp_model_path.exists():
                    shutil.rmtree(temp_model_path)
                os.makedirs(temp_model_path, exist_ok=True)
                
                # Copy model files to temp directory
                logger.info(f"Copying model files from {source_path} to {temp_model_path}")
                for item in os.listdir(source_path):
                    s = os.path.join(source_path, item)
                    d = os.path.join(temp_model_path, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
                
                # Load model from the clean temp directory
                logger.info(f"Loading model from clean directory: {temp_model_path}")
                return CrossEncoder(str(temp_model_path), device='cpu', local_files_only=True)
            
            # Check if local paths exist
            if local_model_path.exists():
                logger.info(f"Found model in nested path: {local_model_path}")
                model = copy_and_load_model(local_model_path)
                logger.info(f"Successfully loaded model from local nested path: {local_model_path}")
            elif local_model_base_path.exists():
                logger.info(f"Found model in base path: {local_model_base_path}")
                model = copy_and_load_model(local_model_base_path)
                logger.info(f"Successfully loaded model from local base path: {local_model_base_path}")
            else:
                error_msg = f"Reranker model '{model_name}' not available locally. Please download it first."
                logger.error(error_msg)
                raise HTTPException(status_code=404, detail=error_msg)
            
            loaded_models[model_key] = model
        except Exception as e:
            logger.error(f"Failed to load reranker model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return loaded_models[model_key]


# Endpoints have been moved to app.api.model_endpoints
# The functions below are utility functions used by the endpoints


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Model Serving API",
        "version": "1.0.0",
        "docs": "/docs",
        "model_services": {
            "health": "/api/v1/models/health",
            "embeddings": "/api/v1/models/embeddings",
            "rerank": "/api/v1/models/rerank",
            "classify": "/api/v1/models/classify",
            "classify_eval": "/api/v1/models/classify/eval",
            "list_models": "/api/v1/models/models"
        },
        "lora_fine_tuning": {
            "jobs": "/api/v1/lora/jobs",
            "adapters": "/api/v1/lora/adapters",
            "generate": "/api/v1/lora/generate"
        }
    }

# Import and include model endpoints router at the end to avoid circular imports
from app.api.model_endpoints import router as model_router
app.include_router(model_router, prefix="/api/v1/models", tags=["Model Services"])

if __name__ == "__main__":
    # Log model directory information
    logger.info(f"Using model directory: {MODEL_DIR.absolute()}")
    if MODEL_DIR.exists():
        models = [d.name for d in MODEL_DIR.iterdir() if d.is_dir()]
        logger.info(f"Found models: {models}")
    else:
        logger.warning(f"Model directory does not exist: {MODEL_DIR.absolute()}")
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created model directory: {MODEL_DIR.absolute()}")
        except Exception as e:
            logger.error(f"Failed to create model directory: {str(e)}")
    
    # Start server
    logger.info("Starting model server on port 9001")
    uvicorn.run(
        "app.model_server:app",
        host="0.0.0.0",
        port=9001,
        reload=False,
        log_level="info"
    )
    logger.info("Model server stopped")
