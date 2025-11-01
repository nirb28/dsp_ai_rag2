# Model Server for DSP AI RAG

This component provides a standalone server for hosting embedding and reranking models that aren't compatible with vLLM. It allows you to serve these models via a REST API on a different port than the main application.

## Configuration

### Environment Variables

Add these variables to your `.env` file:

```
# Model Server Configuration
MODEL_SERVER_URL=http://localhost:8001
LOCAL_MODELS_PATH=./models
```

### Supported Models

The model server currently supports:

- **Embedding Models**: SentenceTransformers models like `all-MiniLM-L6-v2`
- **Reranker Models**: CrossEncoder models like `cross-encoder/ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-large`

## Starting the Server

1. Make sure your virtual environment is activated:
   ```bash
   # On Windows
   .\.venv\Scripts\activate
   
   # On Linux/Mac
   source .venv/bin/activate
   ```

2. Start the model server:
   ```bash
   python -m app.model_server
   ```

3. The server will run on port 8001 by default (http://localhost:8001)

## API Endpoints

### Health Check

Verify the server is running and see which models are loaded.

**Request:**
```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "loaded_models": []
}
```

### List Available Models

List models available in your models directory.

**Request:**
```
GET /models
```

**Response:**
```json
{
  "available_models": ["all-MiniLM-L6-v2"],
  "loaded_models": []
}
```

### Generate Embeddings

Create vector embeddings for text inputs.

**Request:**
```
POST /embeddings
```

**Payload:**
```json
{
  "texts": [
    "This is a sample sentence.",
    "Another example text for embedding."
  ],
  "model_name": "all-MiniLM-L6-v2"
}
```

**Response:**
```json
{
  "embeddings": [
    [0.045, 0.024, ..., 0.065],
    [0.032, 0.018, ..., 0.087]
  ],
  "model": "all-MiniLM-L6-v2",
  "dimensions": 384
}
```

### Rerank Documents

Rerank a list of documents based on relevance to a query.

**Request:**
```
POST /rerank
```

**Payload:**
```json
{
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a branch of artificial intelligence that focuses on developing systems that learn from data.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers.",
    "Python is a popular programming language used for data science and machine learning."
  ],
  "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

**Response:**
```json
{
  "scores": [0.9245, 0.7652, 0.4378],
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

## Using with RAG Pipeline

The model server integrates with the RAG pipeline through the `EmbeddingService` and `RerankerService` classes. To use it:

### Method 1: Configure in Code

Specify the model server URL directly in your embedding configuration:

```python
embedding=EmbeddingConfig(
  model=EmbeddingModel.LOCAL_MODEL_SERVER,
  batch_size=32,
  server_url="http://localhost:8001"  # Explicitly set the URL
)
```

### Method 2: Configure via Environment

1. Set your embedding model in code without specifying the URL:
   ```python
   embedding=EmbeddingConfig(
     model=EmbeddingModel.LOCAL_MODEL_SERVER,
     batch_size=32
   )
   ```

2. Make sure your `.env` file has the `MODEL_SERVER_URL` set correctly:
   ```
   MODEL_SERVER_URL=http://localhost:8001
   ```

The system will first check if `server_url` is provided in the config, and if not, it will fall back to the `MODEL_SERVER_URL` from environment settings.

## Example Postman Collection

You can import the following Postman collection to test the API:

```json
{
  "info": {
    "_postman_id": "your-postman-id",
    "name": "RAG Model Server API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "http://localhost:8001/health"
      }
    },
    {
      "name": "List Models",
      "request": {
        "method": "GET",
        "url": "http://localhost:8001/models"
      }
    },
    {
      "name": "Generate Embeddings",
      "request": {
        "method": "POST",
        "url": "http://localhost:8001/embeddings",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"texts\": [\"This is a sample document.\", \"Another test document.\"],\n  \"model_name\": \"all-MiniLM-L6-v2\"\n}"
        }
      }
    },
    {
      "name": "Rerank Documents",
      "request": {
        "method": "POST",
        "url": "http://localhost:8001/rerank",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"query\": \"artificial intelligence applications\",\n  \"documents\": [\n    \"AI is transforming healthcare with advanced diagnostics.\",\n    \"Machine learning algorithms are used in recommendation systems.\",\n    \"Natural language processing enables chatbots and virtual assistants.\"\n  ],\n  \"model_name\": \"cross-encoder/ms-marco-MiniLM-L-6-v2\"\n}"
        }
      }
    }
  ]
}
```

## Troubleshooting

- **Server won't start**: Make sure your virtual environment has all the required packages installed.
- **Model not found**: Check that the model exists in your `LOCAL_MODELS_PATH` directory.
- **Connection error**: Ensure the port 8001 is not already in use by another service.

## Running on a Different Port

To run the model server on a different port, modify the `port` parameter in the `uvicorn.run()` call in `app/model_server.py`.
