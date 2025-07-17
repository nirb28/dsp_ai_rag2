"""
Dataset classes and utilities for RAG evaluation.
"""
import json
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd

from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


class EvaluationDataset:
    """Base class for evaluation datasets."""
    
    def __init__(self, name: str, data: List[Dict[str, Any]]):
        self.name = name
        self.data = data
    
    def save(self, directory: Union[str, Path]) -> str:
        """Save dataset to JSON file."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        filepath = directory / f"{self.name}.json"
        
        with open(filepath, "w") as f:
            json.dump({
                "name": self.name,
                "count": len(self.data),
                "data": self.data
            }, f, indent=2)
        
        logger.info(f"Saved dataset '{self.name}' with {len(self.data)} items to {filepath}")
        return str(filepath)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        filepath = Path(filepath)
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return cls(name=data["name"], data=data["data"])
    
    @classmethod
    def from_csv(cls, name: str, filepath: Union[str, Path], 
                 query_col: str = "query", 
                 answer_col: Optional[str] = "answer",
                 relevant_docs_col: Optional[str] = "relevant_docs",
                 metadata_cols: Optional[List[str]] = None) -> "EvaluationDataset":
        """Create dataset from CSV file."""
        df = pd.read_csv(filepath)
        
        data = []
        metadata_cols = metadata_cols or []
        
        for _, row in df.iterrows():
            item = {"query": row[query_col]}
            
            if answer_col and answer_col in df.columns:
                item["expected_answer"] = row[answer_col]
            
            if relevant_docs_col and relevant_docs_col in df.columns:
                # Handle case where relevant docs might be a comma-separated string
                relevant_docs = row[relevant_docs_col]
                if isinstance(relevant_docs, str):
                    item["relevant_docs"] = [doc.strip() for doc in relevant_docs.split(",")]
                else:
                    item["relevant_docs"] = [relevant_docs]
            
            if metadata_cols:
                item["metadata"] = {col: row[col] for col in metadata_cols if col in df.columns}
            
            data.append(item)
        
        return cls(name=name, data=data)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""
        return pd.DataFrame(self.data)


class RetrievalEvaluationSet(EvaluationDataset):
    """Dataset specifically for retrieval evaluation with relevant documents."""
    
    @classmethod
    def create_from_queries(cls, 
                           name: str, 
                           queries: List[str],
                           rag_service: RAGService,
                           configuration_name: str,
                           k: int = 5,
                           human_validation: bool = False) -> "RetrievalEvaluationSet":
        """
        Create a retrieval evaluation set from a list of queries.
        
        Args:
            name: Name of the dataset
            queries: List of query strings
            rag_service: RAG service instance to use for retrieving documents
            configuration_name: Configuration name to use
            k: Number of documents to retrieve
            human_validation: If True, requires human validation of relevant documents
                              before creating the dataset
        
        Returns:
            New RetrievalEvaluationSet instance
        """
        data = []
        
        for query in queries:
            response = rag_service.query(
                query=query,
                configuration_name=configuration_name,
                k=k
            )
            
            sources = response.sources if hasattr(response, "sources") else []
            retrieved_docs = [{"id": doc.id, "text": doc.content[:500], "score": doc.score} 
                             for doc in sources if hasattr(doc, "id")]
            
            if human_validation:
                # This is a placeholder - in a real application, you would
                # implement a mechanism for human validation of relevant docs
                logger.info(f"Query: {query}")
                for i, doc in enumerate(retrieved_docs):
                    logger.info(f"Doc {i+1}: {doc['text'][:100]}...")
                
                # For now, we'll just use all retrieved docs as relevant
                relevant_docs = [doc["id"] for doc in retrieved_docs]
            else:
                # For automatic creation, we'll consider all top retrieved docs as relevant
                # This is an approximation - ideally these would be human-validated
                relevant_docs = [doc["id"] for doc in retrieved_docs]
            
            data.append({
                "query": query,
                "relevant_docs": relevant_docs,
                "metadata": {
                    "retrieved_docs": retrieved_docs,
                    "configuration_name": configuration_name
                }
            })
        
        return cls(name=name, data=data)


class QueryEvaluationSet(EvaluationDataset):
    """Dataset specifically for query response evaluation with expected answers."""
    
    @classmethod
    def create_from_qa_pairs(cls,
                            name: str,
                            qa_pairs: List[Dict[str, str]]) -> "QueryEvaluationSet":
        """
        Create a query evaluation set from a list of question-answer pairs.
        
        Args:
            name: Name of the dataset
            qa_pairs: List of dictionaries with 'question' and 'answer' keys
        
        Returns:
            New QueryEvaluationSet instance
        """
        data = []
        
        for pair in qa_pairs:
            data.append({
                "query": pair["question"],
                "expected_answer": pair["answer"],
                "metadata": pair.get("metadata", {})
            })
        
        return cls(name=name, data=data)
    
    @classmethod
    def create_sample_dataset(cls, name: str) -> "QueryEvaluationSet":
        """Create a small sample dataset for testing."""
        qa_pairs = [
            {
                "question": "What is RAG?",
                "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval-based and generation-based approaches for natural language processing tasks. It retrieves relevant documents from a knowledge base and uses them to augment the context for generating answers."
            },
            {
                "question": "How does vector similarity search work?",
                "answer": "Vector similarity search works by converting documents and queries into numerical vector representations (embeddings) using models like transformers. These vectors capture semantic meaning, allowing the system to find documents whose embeddings are most similar to the query embedding using distance metrics like cosine similarity."
            },
            {
                "question": "What is the difference between BM25 and vector search?",
                "answer": "BM25 is a lexical search algorithm that ranks documents based on term frequency and inverse document frequency, focusing on exact keyword matches. Vector search converts text to numerical vectors and finds similar documents based on semantic meaning, capturing contextual relationships beyond keyword matching."
            }
        ]
        
        return cls.create_from_qa_pairs(name, qa_pairs)
