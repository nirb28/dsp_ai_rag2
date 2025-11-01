#!/usr/bin/env python3
"""
Standalone MCP Server Application

This is a separate FastAPI server dedicated to MCP (Model Context Protocol) functionality.
It connects to the RAG service for document retrieval and query processing.
"""

import logging
import json
import datetime
import sys
import os
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.encoders import jsonable_encoder

from app.services.rag_service import RAGService
from app.services.mcp.mcp_server_impl import MCPServerImpl
from app.model_schemas import ErrorResponse

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level_str = os.getenv("MCP_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set all existing loggers to this level
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(log_level)

logger = logging.getLogger(__name__)

# Global services
rag_service = None
mcp_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global rag_service, mcp_manager
    
    logger.info("🚀 Starting MCP Server...")
    
    # Initialize RAG service
    rag_service = RAGService()
    logger.info("✅ RAG Service initialized")
    
    # Initialize MCP server manager
    mcp_manager = MCPServerImpl(rag_service)
    logger.info("✅ MCP Server Manager initialized")
    
    # Auto-start MCP servers with startup_enabled=true
    await auto_start_mcp_servers()
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down MCP Server...")
    if mcp_manager:
        await mcp_manager.shutdown_all()
    logger.info("✅ MCP Server shutdown complete")


async def auto_start_mcp_servers():
    """Auto-start MCP servers for configurations with startup_enabled=true."""
    auto_start_configs = []
    
    for config_name, config in rag_service.configurations.items():
        if (config.mcp_server and 
            config.mcp_server.enabled and 
            config.mcp_server.startup_enabled):
            auto_start_configs.append(config_name)
    
    if auto_start_configs:
        logger.info(f"Auto-starting MCP servers for configurations: {auto_start_configs}")
        
        for config_name in auto_start_configs:
            try:
                result = await mcp_manager.start_server(config_name)
                if result["success"]:
                    logger.info(f"✅ Auto-started MCP server for '{config_name}': {result['message']}")
                else:
                    logger.error(f"❌ Failed to auto-start MCP server for '{config_name}': {result['message']}")
            except Exception as e:
                logger.error(f"❌ Error auto-starting MCP server for '{config_name}': {str(e)}")
    else:
        logger.info("No MCP servers configured for auto-startup")


# Create FastAPI app with lifespan
app = FastAPI(
    title="MCP Server Hub",
    description="Standalone Model Context Protocol server for RAG document retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for MCP clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


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
    
    json_compatible_content = jsonable_encoder(error_content)
    
    return JSONResponse(
        status_code=500,
        content=json_compatible_content
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with MCP server information."""
    if not mcp_manager:
        return {"message": "MCP Server initializing..."}
    
    return {
        "message": "MCP Server Hub",
        "version": "1.0.0",
        "active_configurations": list(mcp_manager.configurations.keys()) if mcp_manager else [],
        "total_configurations": len(mcp_manager.configurations) if mcp_manager else 0,
        "endpoints": {
            "health": "/health",
            "mcp_servers": "/mcp-servers",
            "docs": "/docs"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "MCP Server Hub",
        "version": "1.0.0",
        "running": mcp_manager._is_running if mcp_manager else False,
        "configurations": len(mcp_manager.configurations) if mcp_manager else 0
    }


# Import and include MCP endpoints
from app.api.mcp_endpoints import router as mcp_router
app.include_router(mcp_router, tags=["MCP Servers"])


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("MCP_SERVER_PORT", "7080"))
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    
    logger.info(f"Starting MCP Server on {host}:{port}")
    
    uvicorn.run(
        "app.mcp_main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload for production
        log_level=log_level_str.lower()
    )
