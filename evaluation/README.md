# RAG Evaluation Framework

This directory contains a comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems. The framework provides tools for assessing both retrieval performance and query response quality.

## Directory Structure

```
evaluation/
├── __init__.py           # Package initialization
├── base.py               # Core evaluation classes
├── retrieval.py          # Retrieval evaluation components
├── query.py              # Query response evaluation components
├── visualization.py      # Result visualization utilities
├── datasets/             # Dataset management
│   ├── __init__.py
│   └── dataset.py        # Dataset classes
├── metrics/              # Evaluation metrics
│   ├── __init__.py
│   └── retrieval_metrics.py  # Retrieval-specific metrics
└── results/              # Default directory for evaluation results
```

## How to Use the Framework

### 1. Evaluating Retrieval Performance

```python
from app.services.rag_service import RAGService
from evaluation.retrieval import RetrievalEvaluator, SearchQualityEvaluator
from evaluation.datasets.dataset import RetrievalEvaluationSet

# Initialize services
rag_service = RAGService()

# Create evaluator
retrieval_evaluator = RetrievalEvaluator(rag_service)

# Prepare evaluation data - queries with known relevant documents
eval_queries = [
    {
        "query": "What is RAG?",
        "relevant_docs": ["doc_id_1", "doc_id_2"]  # Document IDs known to be relevant
    },
    # Add more queries...
]

# Run evaluation
results = retrieval_evaluator.evaluate(
    queries=eval_queries,
    configuration_name="your_config",
    k=5  # Number of results to retrieve
)

# Save results
results_path = retrieval_evaluator.save_results(results)

# Alternatively, create an evaluation dataset from test queries
test_queries = ["What is RAG?", "How does vector search work?"]
eval_dataset = RetrievalEvaluationSet.create_from_queries(
    name="test_dataset",
    queries=test_queries,
    rag_service=rag_service,
    configuration_name="your_config"
)

# Save the dataset for reuse
eval_dataset.save("evaluation/datasets")
```

### 2. Evaluating Query Quality

```python
from evaluation.query import QueryEvaluator
from evaluation.datasets.dataset import QueryEvaluationSet

# Create a query evaluator
query_evaluator = QueryEvaluator(rag_service)

# Create or load a dataset with question-answer pairs
qa_pairs = [
    {
        "question": "What is RAG?",
        "answer": "RAG is a technique that combines retrieval with generation..."
    },
    # More pairs...
]
eval_dataset = QueryEvaluationSet.create_from_qa_pairs("test_qa", qa_pairs)

# Run evaluation
results = query_evaluator.evaluate(
    eval_data=eval_dataset.data,
    configuration_name="your_config"
)

# Save results
results_path = query_evaluator.save_results(results)
```

### 3. Visualizing Results

```python
from evaluation.visualization import EvaluationVisualizer

visualizer = EvaluationVisualizer()

# Create various visualizations
plot_path = visualizer.plot_retrieval_metrics(results)

# Create an HTML dashboard from multiple result files
dashboard_path = visualizer.create_dashboard(
    results_files=["path/to/results1.json", "path/to/results2.json"]
)
```

### 4. Comparing Different Configurations

```python
from evaluation.query import QuerySetBenchmark

# Create a benchmark evaluator
benchmark = QuerySetBenchmark(rag_service)

# Run benchmark across multiple configurations
results = benchmark.evaluate(
    query_set=["What is RAG?", "How does embedding work?"],
    configurations=["default", "experimental"]
)
```

## Key Metrics

### Retrieval Metrics

* **Precision@k**: Proportion of retrieved documents that are relevant
* **Recall@k**: Proportion of relevant documents that are retrieved
* **F1 Score**: Harmonic mean of precision and recall
* **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant documents
* **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality with relevance weights
* **Mean Average Precision (MAP)**: Mean of average precision scores across queries

### Query Evaluation Metrics

* **Answer Correctness**: Combined measure of factual accuracy
* **Answer Relevance**: How well the answer addresses the query
* **Answer Completeness**: Whether the answer covers all aspects of the query
* **Semantic Similarity**: Vector similarity between generated and expected answers
* **Token Overlap**: Word-level similarity between generated and expected answers

## Workflow for Effective Evaluation

1. **Create evaluation datasets**
   - Gather suitable test queries with relevant documents
   - Prepare question-answer pairs for query evaluation

2. **Run initial evaluations**
   - Benchmark your current configuration as a baseline
   - Save results for future comparison

3. **Iterate and improve**
   - Use evaluation results to fine-tune your RAG configurations
   - Compare performance across different settings

4. **Visualize and report**
   - Generate visual representations of key metrics
   - Create dashboards for comprehensive analysis

## Best Practices

* Use a diverse set of test queries covering different query types and domains
* Include both easy and challenging queries in your evaluation datasets
* Maintain separate datasets for development and final evaluation
* Regularly update your evaluation datasets as your system evolves
* Compare both retrieval and generation metrics for holistic assessment
