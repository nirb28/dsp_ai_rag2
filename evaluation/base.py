"""
Base classes and interfaces for RAG evaluation framework.
"""
import abc
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationResult:
    """Container for evaluation results with serialization support."""
    
    def __init__(
        self,
        metric_name: str,
        value: Union[float, int, str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.metric_name = metric_name
        self.value = value
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create result from dictionary."""
        result = cls(
            metric_name=data["metric_name"],
            value=data["value"],
            metadata=data.get("metadata", {})
        )
        result.timestamp = data.get("timestamp", datetime.now().isoformat())
        return result


class BaseEvaluator(abc.ABC):
    """Base class for all evaluators."""
    
    def __init__(self, name: str, save_dir: Optional[str] = None):
        self.name = name
        self.save_dir = save_dir or Path("evaluation/results")
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> List[EvaluationResult]:
        """Run evaluation and return results."""
        pass
    
    def save_results(
        self, 
        results: List[EvaluationResult], 
        filename: Optional[str] = None
    ) -> str:
        """Save evaluation results to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.json"
            
        filepath = self.save_dir / filename
        
        data = {
            "evaluator": self.name,
            "timestamp": datetime.now().isoformat(),
            "results": [result.to_dict() for result in results]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved evaluation results to {filepath}")
        return str(filepath)
    
    @staticmethod
    def load_results(filepath: Union[str, Path]) -> List[EvaluationResult]:
        """Load evaluation results from file."""
        filepath = Path(filepath)
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return [EvaluationResult.from_dict(result) for result in data["results"]]
    
    def to_dataframe(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        records = []
        
        for result in results:
            record = {
                "metric": result.metric_name,
                "value": result.value,
                "timestamp": result.timestamp
            }
            
            # Add metadata as columns
            if result.metadata:
                for key, value in result.metadata.items():
                    if isinstance(value, (dict, list)):
                        record[key] = json.dumps(value)
                    else:
                        record[key] = value
            
            records.append(record)
        
        return pd.DataFrame(records)
