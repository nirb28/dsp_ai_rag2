import logging
import json
import datetime
import sys
import os

# Add project root to sys.path for direct debugging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.encoders import jsonable_encoder

from app.api.endpoints import router
from app.model_schemas import ErrorResponse

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# Configure root logger
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set all existing loggers to this level
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(log_level)

logger = logging.getLogger(__name__)
print(f"Logger effective level: {logging.getLevelName(log_level)}")

# Create FastAPI app
app = FastAPI(
    title="RAG as a Service",
    description="A comprehensive RAG (Retrieval-Augmented Generation) platform with configurable chunking, embedding, and generation strategies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG Service"])

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return super().default(obj)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    error_content = ErrorResponse(
        error=exc.detail,
        detail=str(exc.detail) if hasattr(exc, 'detail') else None
    ).dict()
    
    # Use jsonable_encoder to properly handle datetime objects
    json_compatible_content = jsonable_encoder(error_content)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=json_compatible_content
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    error_content = ErrorResponse(
        error="Internal server error",
        detail=str(exc)
    ).dict()
    
    # Use jsonable_encoder to properly handle datetime objects
    json_compatible_content = jsonable_encoder(error_content)
    
    return JSONResponse(
        status_code=500,
        content=json_compatible_content
    )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG as a Service API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "documentation": {
            "chunking_strategies": "/api/v1/documentation/chunking-strategies",
            "embedding_models": "/api/v1/documentation/embedding-models",
            "vector_stores": "/api/v1/documentation/vector-stores",
            "reranker_models": "/api/v1/documentation/reranker-models",
            "mcp_integration": "/api/v1/documentation/mcp-integration"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
        log_level="info"
    )
