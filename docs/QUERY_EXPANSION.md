# Query Expansion Feature

This document describes the query expansion functionality implemented in the DSP AI RAG2 project.

## Overview

Query expansion improves retrieval performance by generating multiple related queries from a single user query, then merging and deduplicating the results. This helps capture documents that might be missed by the original query due to vocabulary mismatches or different phrasings.

## Features

### Key Features

### LLM Configuration Management

Create and manage LLM configurations for query expansion:

- **Supported Providers**: Groq, OpenAI-compatible APIs, Triton Inference Server
- **Configurable Parameters**: Temperature, max tokens, top-p, top-k, timeout
- **Custom System Prompts**: Tailor the expansion behavior
- **Secure API Key Storage**: API keys are redacted in responses

### Query Expansion Strategies

#### Fusion Strategy
- Generates semantically similar query variations
- Uses different phrasings, synonyms, and perspectives
- Maintains the same core intent
- Best for improving recall on the same topic

#### Multi-Query Strategy
- Generates related queries exploring different aspects
- Covers subtopics and related concepts
- Provides comprehensive topic coverage
- Best for exploratory search

### Result Processing

- **Automatic Merging**: Combines results from multiple queries
- **Deduplication**: Removes duplicate documents based on content
- **Score Preservation**: Maintains similarity scores for ranking
- **Source Tracking**: Tracks which query retrieved each document
- **Fallback Behavior**: Uses original query if expansion fails

## API Usage

### 1. Create LLM Configuration

```bash
POST /llm-configs
```

```json
{
  "name": "groq-llama3",
  "provider": "groq",
  "model": "llama3-8b-8192",
  "endpoint": "https://api.groq.com/openai/v1/chat/completions",
  "api_key": "your-groq-api-key",
  "system_prompt": "You're a query expansion assistant. Generate concise search query variations in JSON format. Follow these rules: \
    1. Maintain original meaning \
    2. Use different phrasing styles",
  "temperature": 0.7,
  "max_tokens": 512,
  "top_p": 0.9,
  "timeout": 30
}
```

### 2. Use Query Expansion in Queries

```bash
POST /query
```

```json
{
  "query": "What is machine learning?",
  "configuration_name": "default",
  "k": 5,
  "query_expansion": {
    "enabled": true,
    "strategy": "fusion",
    "llm_config_name": "groq-llama3",
    "num_queries": 3
  }
}
```

### 3. Use Query Expansion in Retrieval

```bash
POST /retrieve
```

```json
{
  "query": "neural network training",
  "configuration_name": "default",
  "k": 10,
  "query_expansion": {
    "enabled": true,
    "strategy": "multi_query",
    "llm_config_name": "groq-llama3",
    "num_queries": 4
  }
}
```

## Configuration Examples

### Local Model Configuration
```json
{
  "name": "local-llama",
  "provider": "openai_compatible",
  "model": "llama3",
  "endpoint": "http://localhost:8000/v1/chat/completions",
  "api_key": null,
  "system_prompt": "Generate concise query variations.",
  "temperature": 0.3,
  "max_tokens": 128
}
```


## Response Format

When query expansion is used, responses include additional information:

### Query Response
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is...",
  "sources": [
    {
      "content": "Machine learning is a subset of AI...",
      "similarity_score": 0.92,
      "source_query": "What is machine learning?",
      "metadata": {...}
    },
    {
      "content": "ML algorithms learn from data...",
      "similarity_score": 0.88,
      "source_query": "How do ML algorithms work?",
      "metadata": {...}
    }
  ],
  "processing_time": 1.23,
  "configuration_name": "default"
}
```

### Retrieve Response
```json
{
  "query": "neural networks",
  "documents": [
    {
      "content": "Neural networks are computing systems...",
      "similarity_score": 0.95,
      "source_query": "neural networks",
      "metadata": {...}
    },
    {
      "content": "Deep learning uses neural networks...",
      "similarity_score": 0.89,
      "source_query": "deep learning neural networks",
      "metadata": {...}
    }
  ],
  "processing_time": 0.87,
  "total_found": 8
}
```

## Management Endpoints

### List LLM Configurations
```bash
GET /llm-configs
```

### Get Specific Configuration
```bash
GET /llm-configs/{config_name}
```

### Delete Configuration
```bash
DELETE /llm-configs/{config_name}
```

## Best Practices

### Strategy Selection
- Use **fusion** for:
  - Improving recall on specific topics
  - Handling vocabulary variations
  - Semantic similarity searches

- Use **multi_query** for:
  - Exploratory research
  - Comprehensive topic coverage
  - Finding related concepts

### Performance Optimization
- **Limit num_queries**: Start with 3-4 queries to balance performance and coverage
- **Use appropriate k values**: Consider that results will be merged and deduplicated
- **Configure timeouts**: Set reasonable timeouts for LLM calls (15-30 seconds)
- **Monitor costs**: Be aware of API costs when using external LLM providers

### Error Handling
- The system automatically falls back to the original query if expansion fails
- Monitor logs for expansion failures and adjust configurations as needed
- Test LLM configurations before using them in production

## Environment Variables

Set these environment variables for LLM providers:

```bash
# Groq
GROQ_API_KEY=your_groq_api_key

# OpenAI (if using OpenAI-compatible endpoints)
OPENAI_API_KEY=your_openai_api_key

# Custom endpoints
CUSTOM_LLM_ENDPOINT=http://localhost:8000
```

## Examples

See the following files for complete examples:
- `examples/query_expansion_example.py` - Comprehensive usage examples
- `test_query_expansion.py` - Basic functionality tests

## Troubleshooting

### Common Issues

1. **LLM Configuration Not Found**
   - Ensure the LLM configuration exists: `GET /llm-configs`
   - Check the `llm_config_name` in your request

2. **API Key Issues**
   - Verify API keys are correctly set in environment variables
   - Check that the API key has sufficient permissions

3. **Timeout Errors**
   - Increase the timeout value in the LLM configuration
   - Check network connectivity to the LLM endpoint

4. **Poor Expansion Quality**
   - Adjust the system prompt for better instructions
   - Try different temperature values (0.3-0.8)
   - Experiment with different models

### Debugging

Enable debug logging to see expansion details:

```python
import logging
logging.getLogger("app.services.query_expansion_service").setLevel(logging.DEBUG)
```

## Integration with Existing Features

Query expansion works seamlessly with:
- **Reranking**: Expanded results are reranked if enabled
- **Multi-configuration retrieval**: Each configuration can use expansion
- **Context injection**: Expansion works with additional context items
- **Metadata filtering**: Metadata handling is preserved
- **Vector inclusion**: Embeddings can still be included in responses

The feature is designed to be backward-compatible - existing queries work unchanged, and expansion is only applied when explicitly requested.
