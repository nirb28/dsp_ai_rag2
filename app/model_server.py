import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import dotenv
import json
import shutil
import tempfile
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for loaded models
loaded_models: Dict[str, Union[SentenceTransformer, CrossEncoder]] = {}

# Temp directory for model copies
TEMP_MODEL_DIR = Path(tempfile.gettempdir()) / "rag_models"
if not TEMP_MODEL_DIR.exists():
    os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

# Model base directory - load directly from .env if available, otherwise use default
MODEL_DIR = Path(os.getenv('LOCAL_MODELS_PATH', './models'))


class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = "all-MiniLM-L6-v2"


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int


class RerankerRequest(BaseModel):
    query: str
    documents: List[str]
    model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerResponse(BaseModel):
    scores: List[float]
    model: str


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


def get_reranker_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load or retrieve reranker model from cache."""
    model_key = f"reranker_{model_name}"
    
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "loaded_models": list(loaded_models.keys())}


@app.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for the input texts."""
    try:
        model = get_embedding_model(request.model_name)
        embeddings = model.encode(request.texts).tolist()
        return {
            "embeddings": embeddings,
            "model": request.model_name,
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.post("/rerank", response_model=RerankerResponse)
async def rerank_documents(request: RerankerRequest):
    """Rerank documents based on relevance to the query."""
    try:
        model = get_reranker_model(request.model_name)
        
        # Create query-document pairs
        pairs = [(request.query, doc) for doc in request.documents]
        
        # Get scores from reranker
        scores = model.predict(pairs).tolist()
        
        return {
            "scores": scores,
            "model": request.model_name
        }
    except Exception as e:
        logger.error(f"Error reranking documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models in the models directory."""
    try:
        available_models = []
        
        # Check local models directory
        if MODEL_DIR.exists():
            # Look for directories that contain model files
            for root, dirs, files in os.walk(MODEL_DIR):
                # Skip the base models directory itself
                if root == str(MODEL_DIR):
                    continue
                
                # Check if this directory has model files
                model_files = [f for f in files if f.endswith(('.bin', '.pt', '.onnx', '.model')) or f == 'config.json']
                if model_files:
                    # Get path relative to MODEL_DIR
                    rel_path = Path(root).relative_to(MODEL_DIR)
                    
                    # Convert Windows path separators to forward slashes for consistency with HF model names
                    model_path = str(rel_path).replace(os.sep, '/')
                    available_models.append(model_path)
        
        # Format the loaded models by removing the prefix
        formatted_loaded_models = []
        for model_key in loaded_models.keys():
            if model_key.startswith('embedding_'):
                formatted_loaded_models.append(model_key[10:])  # Remove 'embedding_' prefix
            elif model_key.startswith('reranker_'):
                formatted_loaded_models.append(model_key[9:])   # Remove 'reranker_' prefix
            else:
                formatted_loaded_models.append(model_key)
        
        # Return list of models and currently loaded models
        return {
            "available_models": available_models,
            "loaded_models": formatted_loaded_models
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


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
