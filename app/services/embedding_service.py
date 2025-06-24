import logging
from typing import List
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer
import openai

from app.config import EmbeddingModel, EmbeddingConfig, settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model based on configuration."""
        try:
            if self.config.model.value.startswith("sentence-transformers/"):
                model_name = self.config.model.value.replace("sentence-transformers/", "")
                self.model = SentenceTransformer(model_name)
                logger.info(f"Initialized SentenceTransformer model: {model_name}")
            elif self.config.model == EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002:
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OpenAI API key not provided")
                openai.api_key = settings.OPENAI_API_KEY
                logger.info("Initialized OpenAI embedding model")
            elif self.config.model == EmbeddingModel.TRITON_EMBEDDING:
                # Triton doesn't need initialization, just verify the server URL is set
                if not settings.TRITON_SERVER_URL:
                    raise ValueError("Triton server URL not provided")
                logger.info(f"Using Triton embedding model at {settings.TRITON_SERVER_URL}")
            else:
                raise ValueError(f"Unsupported embedding model: {self.config.model}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            if self.config.model.value.startswith("sentence-transformers/"):
                return self._embed_with_sentence_transformers(texts)
            elif self.config.model == EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002:
                return self._embed_with_openai(texts)
            elif self.config.model == EmbeddingModel.TRITON_EMBEDDING:
                return self._embed_with_triton(texts)
            else:
                raise ValueError(f"Unsupported embedding model: {self.config.model}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def _embed_with_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformers."""
        # Process in batches
        embeddings = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=False)
            embeddings.extend(batch_embeddings.tolist())
        
        return embeddings

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
        endpoint = f"{settings.TRITON_SERVER_URL}/v2/models/{settings.TRITON_EMBEDDING_MODEL}/infer"
        
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

    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.config.model == EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM:
            return 384
        elif self.config.model == EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MPNET:
            return 768
        elif self.config.model == EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002:
            return 1536
        elif self.config.model == EmbeddingModel.TRITON_EMBEDDING:
            # This might need to be configured based on your Triton model
            return 1024  # Adjust based on your model's dimension
        else:
            # Default fallback - generate a test embedding to get dimension
            test_embedding = self.embed_texts(["test"])
            return len(test_embedding[0])
