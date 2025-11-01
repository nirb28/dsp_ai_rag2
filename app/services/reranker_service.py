import logging
from typing import List, Dict, Any, Optional, Tuple
import requests

from app.config import RerankerConfig, RerankerModel, settings

logger = logging.getLogger(__name__)

class RerankerService:
    """Service for reranking retrieved chunks using endpoint-based models."""
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        
        if config.enabled:
            self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate the reranker configuration for endpoint-based reranking."""
        if not self.config.enabled:
            return
            
        try:
            # Handle model server case
            if isinstance(self.config.model, RerankerModel) and self.config.model == RerankerModel.LOCAL_MODEL_SERVER:
                # Check if model server is reachable
                server_url = self.config.server_url or settings.MODEL_SERVER_URL
                if not server_url:
                    raise ValueError("Server URL not provided for model server reranker")
                try:
                    response = requests.get(f"{server_url}/health", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Connected to model server for reranker at {server_url}")
                    else:
                        logger.warning(f"Model server returned status code {response.status_code}")
                except Exception as e:
                    logger.warning(f"Could not connect to model server: {str(e)}")
                    
            # Handle Cohere Rerank case
            elif isinstance(self.config.model, RerankerModel) and self.config.model == RerankerModel.COHERE_RERANK:
                # Validate Cohere API key
                if not settings.COHERE_API_KEY:
                    logger.warning("Cohere API key not set. Reranking will be disabled.")
                    self.config.enabled = False
                else:
                    logger.info("Using Cohere Rerank API for reranking")
            
            # For custom models, assume they are endpoint-based
            else:
                logger.info(f"Using endpoint-based reranking with model: {self.config.model}")
                
        except Exception as e:
            logger.error(f"Failed to validate reranker configuration: {str(e)}")
            self.config.enabled = False
    
    async def rerank(self, query: str, documents: List[Dict[str, Any]], filter_after_reranking: bool = True) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to the query.
        
        Args:
            query: User query
            documents: List of documents with 'content' and 'metadata' fields
            filter_after_reranking: Whether to apply score threshold filtering after reranking
            
        Returns:
            Reordered list of documents with updated scores
        """
        if not self.config.enabled or not documents:
            return documents
            
        try:
            # Handle enum models
            if isinstance(self.config.model, RerankerModel):
                if self.config.model == RerankerModel.LOCAL_MODEL_SERVER:
                    return await self._rerank_with_model_server(query, documents, filter_after_reranking)
                elif self.config.model == RerankerModel.COHERE_RERANK:
                    return await self._rerank_with_cohere(query, documents, filter_after_reranking)
                else:
                    # All other models use model server endpoint
                    return await self._rerank_with_model_server(query, documents, filter_after_reranking)
            else:
                # Custom models use model server endpoint
                return await self._rerank_with_model_server(query, documents, filter_after_reranking)
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return documents
    

    async def _rerank_with_model_server(self, query: str, documents: List[Dict[str, Any]], filter_after_reranking: bool = True) -> List[Dict[str, Any]]:
        """Rerank documents using the model server endpoint."""
        try:
            # Get the server URL from config or settings
            server_url = self.config.server_url or settings.MODEL_SERVER_URL
            endpoint = f"{server_url}/rerank"
            
            # Extract document texts
            document_texts = [doc["content"] for doc in documents]
            
            # Format the request according to the model server API
            payload = {
                "query": query,
                "texts": document_texts,
                "model_name": self.config.model
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
            
            # Apply score threshold filtering only if filter_after_reranking is True
            if filter_after_reranking:
                filtered_docs = [doc for doc, score in reranked_docs if score >= self.config.score_threshold]
                # Return just the documents
                return filtered_docs if filtered_docs else [doc for doc, _ in reranked_docs][:1]
            else:
                # Return all documents without filtering
                return [doc for doc, _ in reranked_docs]
            
        except Exception as e:
            logger.error(f"Error with model server reranking: {str(e)}")
            return documents

    async def _rerank_with_cohere(self, query: str, documents: List[Dict[str, Any]], filter_after_reranking: bool = True) -> List[Dict[str, Any]]:
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
            
            # Apply score threshold filtering only if filter_after_reranking is True
            if filter_after_reranking:
                filtered_docs = [doc for doc in reranked_docs if doc["similarity_score"] >= self.config.score_threshold]
                return filtered_docs if filtered_docs else reranked_docs[:1]
            else:
                # Return all documents without filtering
                return reranked_docs
            
        except ImportError:
            logger.error("Cohere package not installed. Please install it with: pip install cohere")
            return documents
        except Exception as e:
            logger.error(f"Error with Cohere reranking: {str(e)}")
            return documents
