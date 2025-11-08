# OpenAI-Compatible Chat Completions API

The RAG system now provides OpenAI-compatible chat completions endpoints for each configuration. This allows you to use your RAG system as a drop-in replacement for OpenAI's API, with the added benefit of retrieval-augmented generation from your document collections.

## Overview

The RAG system provides a unified OpenAI-compatible endpoint where the `model` parameter specifies which RAG configuration to use:

```
/v1/chat/completions
```

**Example:**
```bash
POST http://localhost:9000/v1/chat/completions
{
  "model": "malts_faq",
  "messages": [...]
}
```

This approach is similar to LiteLLM and the standard OpenAI API, allowing you to use the same endpoint for all configurations and switch between them by changing the `model` parameter.

## Features

### RAG-Specific Extensions
- Automatic document retrieval based on user query
- Configurable number of documents to retrieve (`k` parameter)

## API Endpoints

### Chat Completions

**POST** `/{configuration_name}/v1/chat/completions`

Create a chat completion with RAG.

#### Request Body

```json
{
  "model": "malts_faq",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is machine learning?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false,
  "k": 5,
  "similarity_threshold": 0.7,
  "filter": {
    "category": "tutorial"
  },
  "filter_after_reranking": true,
  "include_sources": true
}
```

#### Standard OpenAI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model name (maps to configuration name) |
| `messages` | array | required | List of messages in the conversation |
| `temperature` | float | 0.7 | Sampling temperature (0.0 to 2.0) |
| `top_p` | float | 1.0 | Nucleus sampling parameter |
| `n` | integer | 1 | Number of completions to generate |
| `stream` | boolean | false | Whether to stream the response |
| `stop` | string/array | null | Stop sequences |
| `max_tokens` | integer | null | Maximum tokens to generate |
| `presence_penalty` | float | 0.0 | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | float | 0.0 | Frequency penalty (-2.0 to 2.0) |
| `user` | string | null | Unique identifier for the end-user |

#### RAG-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | integer | 5 | Number of documents to retrieve (1-20) |
| `similarity_threshold` | float | 0.7 | Minimum similarity score for retrieval |
| `filter` | object | null | LangChain-style metadata filter |
| `filter_after_reranking` | boolean | true | Apply score threshold after reranking |
| `include_sources` | boolean | true | Include source documents in response |
| `rag_config` | object | null | Override RAG configuration settings |

#### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "malts_faq",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 125,
    "total_tokens": 181
  },
  "sources": [
    {
      "content": "Machine learning is...",
      "metadata": {
        "filename": "ml_basics.pdf",
        "page": 1
      },
      "score": 0.92
    }
  ],
  "processing_time": 1.23
}
```

### List All Models (Unified)

**GET** `/v1/models`

List all available models (RAG configurations) across the entire system.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "malts_faq",
      "object": "model",
      "created": 1677610602,
      "owned_by": "rag-system"
    }
  ]
}
```

### Retrieve Model

**GET** `/v1/models/{model_id}`

Get information about a specific model (RAG configuration).

## Usage Examples

### Example 1: Simple Query (Unified Endpoint)

```python
import requests

# Use the unified endpoint - specify model in payload
url = "http://localhost:9000/v1/chat/completions"

payload = {
    "model": "malts_faq",  # Specify which configuration to use
    "messages": [
        {
            "role": "user",
            "content": "What are the benefits of malts?"
        }
    ],
    "temperature": 0.7,
    "k": 5,
    "include_sources": True
}

response = requests.post(url, json=payload)
result = response.json()

print(result["choices"][0]["message"]["content"])
```

### Example 2: Conversation History

```python
payload = {
    "model": "malts_faq",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant specializing in malts."
        },
        {
            "role": "user",
            "content": "What are malts?"
        },
        {
            "role": "assistant",
            "content": "Malts are grains that have been processed..."
        },
        {
            "role": "user",
            "content": "How are they used in brewing?"
        }
    ],
    "k": 5
}

response = requests.post(url, json=payload)
```

### Example 3: Streaming Response

```python
payload = {
    "model": "malts_faq",
    "messages": [
        {
            "role": "user",
            "content": "Explain the malting process."
        }
    ],
    "stream": True
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data_str = line_str[6:]
            if data_str != '[DONE]':
                chunk = json.loads(data_str)
                content = chunk['choices'][0]['delta'].get('content', '')
                print(content, end='', flush=True)
```

### Example 4: Using OpenAI Python Library

```python
from openai import OpenAI

# Point the OpenAI client to the unified endpoint
client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="dummy-key"  # Not used if security is disabled
)

# List available models
models = client.models.list()
print(f"Available models: {[m.id for m in models.data]}")

# Create completion with specific model
completion = client.chat.completions.create(
    model="malts_faq",  # Specify which RAG configuration to use
    messages=[
        {
            "role": "user",
            "content": "What is the difference between pale malt and crystal malt?"
        }
    ],
    temperature=0.7
)

print(completion.choices[0].message.content)
```

### Example 5: With Metadata Filtering

```python
payload = {
    "model": "malts_faq",
    "messages": [
        {
            "role": "user",
            "content": "What are the brewing techniques?"
        }
    ],
    "filter": {
        "category": "brewing",
        "difficulty": {"$in": ["beginner", "intermediate"]}
    },
    "k": 5
}

response = requests.post(url, json=payload)
```

### Example 6: With Authentication

```python
headers = {
    "Authorization": "Bearer your-jwt-token-here",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
```
### 6. With RAG-Specific Parameters

```python
response = requests.post(
    "http://localhost:9000/malts_faq/v1/chat/completions",
    json={
        "model": "malts_faq",
        "messages": [
            {"role": "user", "content": "What are the types of malts?"}
        ],
        "k": 10,  # Retrieve 10 documents
        "similarity_threshold": 0.8,  # Higher threshold
        "filter": {  # Metadata filter
            "category": "brewing",
            "difficulty": {"$in": ["beginner", "intermediate"]}
        },
        "include_sources": True  # Include source documents
    }
)
```

### 7. With Authentication

```python
headers = {
    "Authorization": "Bearer your-jwt-token-here",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:9000/malts_faq/v1/chat/completions",
    json={
        "model": "malts_faq",
        "messages": [
            {"role": "user", "content": "What are the benefits of malts?"}
        ]
    },
    headers=headers
)
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `messages` | Conversation messages | Required |
| `temperature` | Sampling temperature (0-2) | 0.7 |
| `max_tokens` | Max tokens to generate | None |
| `stream` | Enable streaming | false |
| `k` | Documents to retrieve | 5 |
| `filter` | Metadata filter | None |
| `include_sources` | Include sources | true |

## Authentication

If security is enabled for a configuration, include a JWT token in the Authorization header:

```python
headers = {
    "Authorization": "Bearer <your-jwt-token>",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
```

The JWT token can include metadata filters in the `metadata_filter` claim:

```json
{
  "sub": "user123",
  "metadata_filter": {
    "department": "engineering",
    "access_level": "standard"
  }
}
```

These filters will be automatically applied to document retrieval.

## Differences from OpenAI API

### Additions
- `k` parameter for controlling document retrieval
- `similarity_threshold` for filtering retrieved documents
- `filter` parameter for metadata filtering
- `filter_after_reranking` for reranking control
- `include_sources` to include source documents
- `sources` field in response with retrieved documents
- `processing_time` field in response

### Limitations
- `n` parameter (multiple completions) always returns 1
- `logit_bias` is accepted but not implemented
- `logprobs` is not implemented
- Function calling is not yet supported

## Best Practices

1. **Use conversation history**: Include previous messages for context-aware responses
2. **Set appropriate k**: Start with k=5 and adjust based on your needs
3. **Use metadata filters**: Filter documents by category, date, or other metadata
4. **Enable sources**: Include sources to verify the information
5. **Adjust temperature**: Lower for factual responses, higher for creative ones
6. **Use streaming**: For better user experience with long responses
7. **Configure per-use-case**: Create separate configurations for different use cases

