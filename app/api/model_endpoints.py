"""
FastAPI router for model server endpoints
"""
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict

from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.model_schemas.model_server_models import (
    EmbeddingRequest, EmbeddingResponse,
    RerankerRequest, RerankerResponse,
    ClassificationRequest, ClassificationResult, ClassificationTextResult, ClassificationResponse,
    ClassificationEvalRequest, ClassificationMetrics, ClassificationEvalResponse
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Model Services"])

# Import needed variables - these will be imported dynamically to avoid circular imports
import os
import sys
from pathlib import Path

# We'll import model_server lazily inside each endpoint function to avoid circular imports
# This ensures model_server is fully initialized before we access it


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    from app import model_server
    return {"status": "ok", "loaded_models": list(model_server.loaded_models.keys())}


@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for the input texts."""
    try:
        from app import model_server
        model = model_server.get_embedding_model(request.model_name)
        embeddings = model.encode(request.texts).tolist()
        return {
            "embeddings": embeddings,
            "model": request.model_name,
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@router.post("/rerank", response_model=RerankerResponse)
async def rerank_documents(request: RerankerRequest):
    """Rerank texts based on relevance to the query."""
    try:
        from app import model_server
        model = model_server.get_reranker_model(request.model_name)
        
        # Create query-text pairs
        pairs = [(request.query, text) for text in request.texts]
        
        # Get scores from reranker
        scores = model.predict(pairs).tolist()
        
        return {
            "scores": scores,
            "model": request.model_name
        }
    except Exception as e:
        logger.error(f"Error reranking texts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@router.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """Classify texts into one of the provided labels.
    
    Uses a cross-encoder to score each (label, text) pair and returns sorted results.
    Processes multiple texts and returns results for each text.
    """
    try:
        from app import model_server
        model = model_server.get_reranker_model(request.model_name)
        text_results = []
        
        # Process each text separately
        for text in request.texts:
            # Create label-text pairs for the current text
            pairs = [(label, text) for label in request.labels]
            
            # Get scores from model for current text
            scores = model.predict(pairs).tolist()
            
            # Sort label-score pairs by score (highest first)
            sorted_results = sorted(zip(request.labels, scores), key=lambda x: x[1], reverse=True)
            
            # Create results list if needed
            results = []
            if request.include_results:
                results = [
                    {"label": label, "score": score}
                    for label, score in sorted_results
                ]
            
            # Get the top label and score
            top_label, top_score = sorted_results[0] if sorted_results else ("", 0.0)
            
            # Add results for this text
            result_dict = {
                "top_label": top_label,
                "top_score": top_score
            }
            
            # Only include detailed results if requested
            if request.include_results:
                result_dict["results"] = results
                
            text_results.append(result_dict)
        
        return {
            "text_results": text_results,
            "model": request.model_name
        }
    except Exception as e:
        logger.error(f"Error classifying text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/classify/eval", response_model=ClassificationEvalResponse)
async def evaluate_classification(request: ClassificationEvalRequest):
    """Evaluate classification performance by comparing predictions against ground truth labels.
    
    Uses a cross-encoder to classify texts and computes evaluation metrics including:
    accuracy, precision, recall, F1 score, and confusion matrix.
    """
    try:
        from app import model_server
        if len(request.texts) != len(request.ground_truths):
            raise HTTPException(status_code=400, 
                                detail="Number of texts must match number of ground truth labels")
        
        # Validate ground truths are in provided labels
        invalid_labels = [gt for gt in request.ground_truths if gt not in request.labels]
        if invalid_labels:
            raise HTTPException(status_code=400,
                               detail=f"Ground truth labels {invalid_labels} not found in provided labels")
        
        from app import model_server
        model = model_server.get_reranker_model(request.model_name)
        predictions = []
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        # Process each text and evaluate against ground truth
        for text, ground_truth in zip(request.texts, request.ground_truths):
            # Create label-text pairs for the current text
            pairs = [(label, text) for label in request.labels]
            
            # Get scores from model for current text
            scores = model.predict(pairs).tolist()
            
            # Sort label-score pairs by score (highest first)
            sorted_results = sorted(zip(request.labels, scores), key=lambda x: x[1], reverse=True)
            predicted_label, predicted_score = sorted_results[0] if sorted_results else ("", 0.0)
            
            # Update confusion matrix
            confusion_matrix[ground_truth][predicted_label] += 1
            
            # Save prediction info
            predictions.append({
                "text": text,
                "ground_truth": ground_truth,
                "predicted": predicted_label,
                "score": predicted_score,
                "correct": ground_truth == predicted_label
            })
        
        # Calculate metrics
        label_counts = {label: sum(gt == label for gt in request.ground_truths) for label in request.labels}
        
        # Convert defaultdict to regular dict for JSON serialization
        confusion_dict = {k: dict(v) for k, v in confusion_matrix.items()}
        
        # Ensure all cells in confusion matrix are represented
        for true_label in request.labels:
            if true_label not in confusion_dict:
                confusion_dict[true_label] = {}
            for pred_label in request.labels:
                if pred_label not in confusion_dict[true_label]:
                    confusion_dict[true_label][pred_label] = 0
        
        # Calculate accuracy
        correct = sum(1 for p in predictions if p["correct"])
        accuracy = correct / len(predictions) if predictions else 0
        
        # Calculate per-class metrics
        precision = {}
        recall = {}
        f1_score = {}
        
        for label in request.labels:
            # True positives: cases where we correctly predicted this label
            tp = sum(1 for p in predictions if p["predicted"] == label and p["ground_truth"] == label)
            
            # False positives: cases where we predicted this label but it was wrong
            fp = sum(1 for p in predictions if p["predicted"] == label and p["ground_truth"] != label)
            
            # False negatives: cases where we didn't predict this label but should have
            fn = sum(1 for p in predictions if p["predicted"] != label and p["ground_truth"] == label)
            
            # Calculate precision, recall, and F1 (with handling for division by zero)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            precision[label] = prec
            recall[label] = rec
            f1_score[label] = f1
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "label_count": label_counts,
            "confusion_matrix": confusion_dict
        }
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "model": request.model_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification evaluation failed: {str(e)}")


@router.get("/list")
async def list_models():
    """List available models in the models directory."""
    # Import needed modules and variables
    import os
    from app import model_server
    from app.model_server import MODEL_DIR
    
    try:
        available_models = []
        
        # Check local models directory
        if MODEL_DIR.exists():
            # First level directories (e.g., sentence-transformers)
            for first_level_dir in MODEL_DIR.iterdir():
                if not first_level_dir.is_dir():
                    continue
                    
                # Second level directories (e.g., all-MiniLM-L6-v2)
                for second_level_dir in first_level_dir.iterdir():
                    if not second_level_dir.is_dir():
                        continue
                        
                    # Check if this directory has model files (directly or in subdirectories)
                    has_model_files = False
                    for root, _, files in os.walk(second_level_dir):
                        model_files = [f for f in files if f.endswith(('.bin', '.pt', '.onnx', '.model')) or f == 'config.json']
                        if model_files:
                            has_model_files = True
                            break
                    
                    if has_model_files:
                        # Create path using first two levels only
                        rel_path = second_level_dir.relative_to(MODEL_DIR)
                        # Convert Windows path separators to forward slashes for consistency with HF model names
                        model_path = str(rel_path).replace(os.sep, '/')
                        available_models.append(model_path)
        
        # Format the loaded models by removing the prefix
        formatted_loaded_models = []
        for model_key in model_server.loaded_models.keys():
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
