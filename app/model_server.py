import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config import settings

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

# Dictionary to store loaded models
loaded_models: Dict[str, Any] = {}

# Model base directory
MODEL_DIR = Path(settings.LOCAL_MODELS_PATH)


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
            # Check if model exists locally
            local_model_path = MODEL_DIR / model_name.split('/')[-1]
            if local_model_path.exists():
                model = SentenceTransformer(str(local_model_path))
                logger.info(f"Loaded model from local path: {local_model_path}")
            else:
                model = SentenceTransformer(model_name)
                logger.info(f"Downloaded model from Hugging Face: {model_name}")
            
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
            # Check if model exists locally
            local_model_path = MODEL_DIR / model_name.split('/')[-1]
            if local_model_path.exists():
                model = CrossEncoder(str(local_model_path))
                logger.info(f"Loaded model from local path: {local_model_path}")
            else:
                model = CrossEncoder(model_name)
                logger.info(f"Downloaded model from Hugging Face: {model_name}")
            
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
        models = []
        
        # Check local models directory
        if MODEL_DIR.exists():
            local_models = [d.name for d in MODEL_DIR.iterdir() if d.is_dir()]
            models.extend(local_models)
        
        # Return list of models and currently loaded models
        return {
            "available_models": models,
            "loaded_models": list(loaded_models.keys())
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app.model_server:app",
        host="0.0.0.0",
        port=8001,  # Using a different port than the main app
        reload=True,
        log_level="info"
    )
