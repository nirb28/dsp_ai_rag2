# Endpoint-Only Configuration Guide

This document provides guidance on configuring the DSP AI RAG2 system to use only external endpoints without downloading any models locally.

## Configuration Overview

To ensure no models are downloaded locally, follow these configuration principles:

### Vector Store Options

1. **BM25 (Recommended for endpoint-only setups)**
   - Uses pure statistical algorithms with no model downloads
   - Works without embeddings
   - Example configuration:
   ```json
   "vector_store": {
     "type": "bm25",
     "index_path": "./storage/bm25_index"
   }
   ```

2. **FAISS or Redis**
   - Can be used if paired with endpoint-based embeddings only

### Embedding Service Options

Avoid local model downloads by using these options:

1. **Disable embeddings entirely with BM25**
   ```json
   "embedding": {
     "enabled": false,
     "model": "none"
   }
   ```

2. **OpenAI API**
   ```json
   "embedding": {
     "enabled": true,
     "model": "text-embedding-ada-002",
     "provider": "openai"
   }
   ```

3. **Triton Endpoint**
   ```json
   "embedding": {
     "enabled": true,
     "model": "triton-embedding",
     "provider": "triton",
     "url": "http://your-triton-server:8000"
   }
   ```

4. **Local Model Server Endpoint**
   ```json
   "embedding": {
     "enabled": true,
     "model": "local-model-server",
     "url": "http://your-model-server:8080/embeddings"
   }
   ```

❌ **AVOID**: Sentence Transformers configuration as it downloads models locally

### Generation Service Options

Use these endpoint-based options:

1. **Groq API**
   ```json
   "generation": {
     "enabled": true,
     "provider": "groq",
     "model": "llama3-70b-8192"
   }
   ```

2. **OpenAI Compatible API**
   ```json
   "generation": {
     "enabled": true,
     "provider": "openai_compatible",
     "url": "http://your-llm-server/v1",
     "model": "llama3"
   }
   ```

3. **Triton API**
   ```json
   "generation": {
     "enabled": true,
     "provider": "triton",
     "url": "http://your-triton-server:8000",
     "model": "llama3-vllm"
   }
   ```

## Sample Fully Endpoint-Based Configuration

```json
{
  "configuration_name": "endpoint_only_config",
  "config": {
    "embedding": {
      "enabled": false,
      "model": "none"
    },
    "vector_store": {
      "type": "bm25",
      "index_path": "./storage/bm25_index"
    },
    "generation": {
      "enabled": true,
      "provider": "groq",
      "model": "llama3-70b-8192",
      "temperature": 0.1,
      "system_prompt": "You are a helpful assistant that provides accurate information based on the given context."
    },
    "retrieval": {
      "k": 5
    }
  }
}
```
