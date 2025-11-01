"""
Pydantic models for the model server API
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    texts: List[str]
    model_name: Optional[str] = "all-MiniLM-L6-v2"


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]]
    dimensions: int
    model: str


class RerankerRequest(BaseModel):
    """Request model for reranking."""
    query: str
    texts: List[str]
    model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerResponse(BaseModel):
    """Response model for reranking."""
    scores: List[float]
    model: str


class ClassificationRequest(BaseModel):
    """Request model for text classification."""
    texts: List[str]
    labels: List[str]
    model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    include_results: bool = True


class ClassificationResult(BaseModel):
    """Result model for a single classification."""
    label: str
    score: float


class ClassificationTextResult(BaseModel):
    """Classification result for a single text."""
    results: Optional[List[ClassificationResult]] = None
    top_label: str
    top_score: float


class ClassificationResponse(BaseModel):
    """Response model for text classification."""
    text_results: List[ClassificationTextResult]
    model: str


class ClassificationEvalRequest(BaseModel):
    """Request model for classification evaluation."""
    texts: List[str]
    labels: List[str]
    ground_truths: List[str]
    model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ClassificationMetrics(BaseModel):
    """Metrics for classification evaluation."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    label_count: Dict[str, int]
    confusion_matrix: Dict[str, Dict[str, int]]


class ClassificationEvalResponse(BaseModel):
    """Response model for classification evaluation."""
    metrics: ClassificationMetrics
    predictions: List[Dict[str, Any]]
    model: str
