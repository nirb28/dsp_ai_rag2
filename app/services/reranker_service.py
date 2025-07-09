import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import requests
from sentence_transformers import CrossEncoder

from app.config import RerankerConfig, RerankerModel, settings

logger = logging.getLogger(__name__)

class RerankerService:
    """Service for reranking retrieved chunks using different models."""
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = None
        
        if config.enabled:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the reranker model based on configuration."""
        if not self.config.enabled or self.config.model == RerankerModel.NONE:
            return
            
        try:
            if self.config.model == RerankerModel.LOCAL_MODEL_SERVER:
                # Check if model server is reachable
                server_url = self.config.server_url or settings.MODEL_SERVER_URL
                if not server_url:
                    raise ValueError("Server URL not provided for model server reranker")
                try:
                    response = requests.get(f"{server_url}/health")
                    if response.status_code == 200:
                        logger.info(f"Connected to model server for reranker at {server_url}")
                    else:
                        logger.warning(f"Model server returned status code {response.status_code}")
                except Exception as e:
                    logger.warning(f"Could not connect to model server: {str(e)}")
                    
            elif self.config.model == RerankerModel.SENTENCE_TRANSFORMERS_CROSS_ENCODER:
                self.model = CrossEncoder(self.config.model.value)
                logger.info(f"Initialized SentenceTransformers CrossEncoder reranker: {self.config.model.value}")
                
            elif self.config.model == RerankerModel.BGE_RERANKER:
                self.model = CrossEncoder("BAAI/bge-reranker-large")
                logger.info("Initialized BGE large reranker")
                
            elif self.config.model == RerankerModel.COHERE_RERANK:
                # Cohere requires API calls at rerank time, no model to load
                if not settings.COHERE_API_KEY:
                    logger.warning("Cohere API key not set. Reranking will be disabled.")
                    self.config.enabled = False
                else:
                    logger.info("Using Cohere Rerank API for reranking")
            
            else:
                logger.warning(f"Unsupported reranker model: {self.config.model}. Reranking will be disabled.")
                self.config.enabled = False
                
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {str(e)}")
            self.config.enabled = False
    
    async def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to the query.
        
        Args:
            query: User query
            documents: List of documents with 'content' and 'metadata' fields
            
        Returns:
            Reordered list of documents with updated scores
        """
        if not self.config.enabled or not documents:
            return documents
            
        try:
            if self.config.model == RerankerModel.LOCAL_MODEL_SERVER:
                return await self._rerank_with_model_server(query, documents)
            elif self.config.model == RerankerModel.COHERE_RERANK:
                return await self._rerank_with_cohere(query, documents)
            elif self.model:
                return self._rerank_with_local_model(query, documents)
            else:
                return documents
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return documents
    
    def _rerank_with_local_model(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using a local cross-encoder model."""
        document_texts = [doc["content"] for doc in documents]
        
        # Prepare query-document pairs
        pairs = [(query, doc_text) for doc_text in document_texts]
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Create tuples of (document, score)
        scored_docs = []
        for i, score in enumerate(scores):
            doc = documents[i].copy()
            # Replace original score with reranking score
            doc["original_similarity_score"] = doc.get("similarity_score", 0)
            doc["similarity_score"] = float(score)
            scored_docs.append((doc, score))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by score threshold
        filtered_docs = [doc for doc, score in scored_docs if score >= self.config.score_threshold]
        
        # Return just the documents
        return filtered_docs if filtered_docs else [doc for doc, _ in scored_docs][:1]
    
    async def _rerank_with_model_server(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using the model server endpoint."""
        try:
            # Get the server URL from config or settings
            server_url = self.config.server_url or settings.MODEL_SERVER_URL
            endpoint = f"{server_url}/rerank"
            
            # Extract document texts
            document_texts = [doc["content"] for doc in documents]
            
            # Determine which model to use
            model_name = self.config.model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            
            # Format the request according to the model server API
            payload = {
                "query": query,
                "documents": document_texts,
                "model_name": model_name
            }
            
            # Call the model server reranking API
            response = requests.post(
                url=endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"Model server returned error: {response.status_code}, {response.text}")
                return documents
                
            result = response.json()
            scores = result["scores"]
            
            # Create reranked document list
            reranked_docs = []
            for i, score in enumerate(scores):
                doc = documents[i].copy()
                # Replace original score with reranking score
                doc["original_similarity_score"] = doc.get("similarity_score", 0)
                doc["similarity_score"] = float(score)
                reranked_docs.append((doc, score))
            
            # Sort by score in descending order
            reranked_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by score threshold
            filtered_docs = [doc for doc, score in reranked_docs if score >= self.config.score_threshold]
            
            # Return just the documents
            return filtered_docs if filtered_docs else [doc for doc, _ in reranked_docs][:1]
            
        except Exception as e:
            logger.error(f"Error with model server reranking: {str(e)}")
            return documents

    async def _rerank_with_cohere(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using Cohere's Rerank API."""
        try:
            import cohere
            
            co = cohere.Client(settings.COHERE_API_KEY)
            
            # Extract document texts
            document_texts = [doc["content"] for doc in documents]
            
            # Call Cohere's rerank API
            response = co.rerank(
                query=query,
                documents=document_texts,
                top_n=len(documents),
                model="rerank-english-v2.0"
            )
            
            # Create reranked document list
            reranked_docs = []
            for result in response.results:
                index = result.index
                doc = documents[index].copy()
                # Replace original score with reranking score
                doc["original_similarity_score"] = doc.get("similarity_score", 0)
                doc["similarity_score"] = result.relevance_score
                reranked_docs.append(doc)
            
            # Filter by score threshold
            filtered_docs = [doc for doc in reranked_docs if doc["similarity_score"] >= self.config.score_threshold]
            
            return filtered_docs if filtered_docs else reranked_docs[:1]
            
        except ImportError:
            logger.error("Cohere package not installed. Please install it with: pip install cohere")
            return documents
        except Exception as e:
            logger.error(f"Error with Cohere reranking: {str(e)}")
            return documents
