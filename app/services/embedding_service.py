import logging
from typing import List
import numpy as np
import json
import os
import requests
import openai

from app.config import EmbeddingModel, EmbeddingConfig, settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._initialize_model()

    def _initialize_model(self):
        """Verify API keys and endpoints for embedding services."""
        try:
            if self.config.model.value.startswith("sentence-transformers/"):
                # Sentence transformer models should be accessed through the model server
                model_server_url = self.config.model_server_url or settings.MODEL_SERVER_URL
                if not model_server_url:
                    raise ValueError("Model server URL not provided for sentence transformer models")
                # Check if model server is reachable
                try:
                    response = requests.get(f"{model_server_url}/health")
                    if response.status_code == 200:
                        logger.info(f"Connected to model server for sentence transformers at {model_server_url}")
                    else:
                        logger.warning(f"Model server returned status code {response.status_code}")
                except Exception as e:
                    logger.warning(f"Could not connect to model server: {str(e)}")
            elif self.config.model == EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002:
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OpenAI API key not provided")
                openai.api_key = settings.OPENAI_API_KEY
                logger.info("Verified OpenAI API key is set")
            elif self.config.model == EmbeddingModel.TRITON_EMBEDDING:
                # Verify the server URL is set
                server_url = self.config.server_url
                if not server_url:
                    raise ValueError("Server URL not provided")
                logger.info(f"Using embedding model server at {server_url}")
            elif self.config.model == EmbeddingModel.LOCAL_MODEL_SERVER:
                # Verify the server URL is set
                model_server_url = self.config.model_server_url or settings.MODEL_SERVER_URL
                if not model_server_url:
                    raise ValueError("Model server URL not provided")
                # Check if model server is reachable
                try:
                    response = requests.get(f"{model_server_url}/health")
                    if response.status_code == 200:
                        logger.info(f"Connected to local model server at {model_server_url}")
                    else:
                        logger.warning(f"Local model server returned status code {response.status_code}")
                except Exception as e:
                    logger.warning(f"Could not connect to local model server: {str(e)}")
            else:
                raise ValueError(f"Unsupported embedding model: {self.config.model}")
        except Exception as e:
            logger.error(f"Error initializing embedding service: {str(e)}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            if self.config.model.value.startswith("sentence-transformers/"):
                # Redirect to local model server for sentence transformers
                model_name = self.config.model.value.replace("sentence-transformers/", "")
                return self._embed_with_local_server(texts, model_name)
            elif self.config.model == EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002:
                return self._embed_with_openai(texts)
            elif self.config.model == EmbeddingModel.TRITON_EMBEDDING:
                return self._embed_with_triton(texts)
            elif self.config.model == EmbeddingModel.LOCAL_MODEL_SERVER:
                return self._embed_with_local_server(texts)
            else:
                raise ValueError(f"Unsupported embedding model: {self.config.model}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        embeddings = []
        batch_size = min(self.config.batch_size, 100)  # OpenAI has limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = openai.Embedding.create(
                model=self.config.model.value,
                input=batch
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)
        
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        embeddings = self.embed_texts([query])
        return embeddings[0]

    def _embed_with_triton(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Triton Inference Server."""
        embeddings = []
        batch_size = self.config.batch_size
        server_url = self.config.server_url
        endpoint = f"{server_url}/v2/models/{settings.TRITON_EMBEDDING_MODEL}/infer"
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Format the request according to Triton's API
            payload = {
                "inputs": [{
                    "name": "text_input",
                    "shape": [len(batch)],
                    "datatype": "BYTES",
                    "data": batch
                }]
            }
            
            response = requests.post(
                url=endpoint,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Triton server returned error: {response.status_code}, {response.text}")
                
            result = response.json()
            # Parse the response - exact format may depend on how the model is configured in Triton
            batch_embeddings = result["outputs"][0]["data"]
            embeddings.extend(batch_embeddings)
        
        return embeddings

    def _embed_with_local_server(self, texts: List[str], specific_model=None) -> List[List[float]]:
        """Generate embeddings using the local model server."""
        # Use model_server_url from config if provided, otherwise fall back to settings
        model_server_url = self.config.model_server_url or settings.MODEL_SERVER_URL
        endpoint = f"{model_server_url}/embeddings"
        
        embeddings = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Determine which model to use
            if specific_model:
                model_name = specific_model
            else:
                model_name = self.config.model.value.replace("local-model-server/", "") \
                            if self.config.model.value.startswith("local-model-server/") \
                            else "all-MiniLM-L6-v2"  # default model if none specified
            
            # Format the request according to the model server API
            payload = {
                "texts": batch,
                "model_name": model_name
            }
            
            try:
                response = requests.post(
                    url=endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    raise Exception(f"Model server returned error: {response.status_code}, {response.text}")
                    
                result = response.json()
                # Parse the response according to our model server API
                batch_embeddings = result["embeddings"]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error calling model server: {str(e)}")
                raise
        
        return embeddings
        
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # For all models, we'll determine dimensions by generating a test embedding
        # This ensures we're always getting the current dimensions from the endpoint
        try:
            test_embedding = self.embed_texts(["test"])
            return len(test_embedding[0])
        except Exception as e:
            # Fall back to known dimensions if endpoint call fails
            logger.warning(f"Couldn't determine embedding dimension from endpoint: {str(e)}")
            
            if self.config.model == EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM:
                return 384
            elif self.config.model == EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MPNET:
                return 768
            elif self.config.model == EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002:
                return 1536
            elif self.config.model == EmbeddingModel.TRITON_EMBEDDING:
                return 1024  # Adjust based on your model's dimension
            else:
                # Default fallback
                return 768  # Common dimension for many embedding models
