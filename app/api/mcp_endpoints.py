"""
MCP Server API Endpoints

This module contains all the REST API endpoints for MCP server management.
These endpoints were moved from the main RAG server to the dedicated MCP server.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import StreamingResponse

from app.model_schemas.mcp_models import (
    MCPServerStartRequest,
    MCPServerStopRequest, 
    MCPServerStatusResponse,
    MCPServerListResponse,
    MCPToolExecuteRequest,
    MCPToolExecuteResponse
)

logger = logging.getLogger(__name__)

# Router for MCP endpoints
router = APIRouter()


# Global reference to MCP manager (will be set by main app)
mcp_manager = None


def get_mcp_manager():
    """Get the global MCP manager instance."""
    from app.mcp_main import mcp_manager as global_manager
    if not global_manager:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")
    return global_manager


# MCP Protocol Endpoints (path-based routing)
@router.post("/{configuration_name}/mcp")
async def handle_mcp_request(configuration_name: str, request: Dict[str, Any]):
    """Handle MCP JSON-RPC request for a specific configuration."""
    manager = get_mcp_manager()
    response = await manager._handle_mcp_request(configuration_name, request)
    
    # Handle notifications (no response expected)
    if response is None:
        # Return 204 No Content for notifications (MCP protocol requirement)
        return Response(status_code=204)
    
    return response


@router.get("/{configuration_name}/sse")
async def handle_sse(configuration_name: str):
    """Handle Server-Sent Events for a specific configuration."""
    manager = get_mcp_manager()
    return StreamingResponse(
        manager._handle_sse_stream(configuration_name),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.websocket("/{configuration_name}/ws")
async def handle_websocket(configuration_name: str, websocket: WebSocket):
    """Handle WebSocket connection for a specific configuration."""
    manager = get_mcp_manager()
    await manager._handle_websocket(configuration_name, websocket)


@router.get("/{configuration_name}/info")
async def get_server_info(configuration_name: str):
    """Get server information for a specific configuration."""
    manager = get_mcp_manager()
    return manager.get_server_status(configuration_name)


# MCP Management Endpoints
@router.post("/mcp-servers/start")
async def start_mcp_server(request: MCPServerStartRequest):
    """Start MCP server for a configuration."""
    try:
        manager = get_mcp_manager()
        result = await manager.start_server(request.configuration_name, request.force_restart)
        
        return {
            "success": result["success"],
            "message": result["message"],
            "configuration_name": request.configuration_name,
            "endpoints": result.get("endpoints", {})
        }
        
    except Exception as e:
        logger.error(f"Error starting MCP server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start MCP server: {str(e)}")


@router.post("/mcp-servers/stop")
async def stop_mcp_server(request: MCPServerStopRequest):
    """Stop MCP server for a configuration."""
    try:
        manager = get_mcp_manager()
        result = await manager.stop_server(request.configuration_name)
        
        return {
            "success": result["success"],
            "message": result["message"],
            "configuration_name": request.configuration_name
        }
        
    except Exception as e:
        logger.error(f"Error stopping MCP server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop MCP server: {str(e)}")


@router.get("/mcp-servers/{configuration_name}", response_model=MCPServerStatusResponse)
async def get_mcp_server_status(configuration_name: str):
    """Get status of MCP server for a specific configuration."""
    try:
        manager = get_mcp_manager()
        status = manager.get_server_status(configuration_name)
        
        if not status["running"] and configuration_name not in manager.configurations:
            raise HTTPException(
                status_code=404, 
                detail=f"MCP server for configuration '{configuration_name}' not found"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting MCP server status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get server status: {str(e)}")


@router.get("/mcp-servers", response_model=MCPServerListResponse)
async def list_mcp_servers():
    """List all MCP servers and their status."""
    try:
        manager = get_mcp_manager()
        servers = manager.list_servers()
        
        return {
            "servers": servers,
            "total_count": len(servers),
            "running": manager._is_running,
            "port": manager._base_port,
            "host": manager._base_host,
            "active_configurations": list(manager.configurations.keys()),
            "hub_url": f"http://{manager._base_host}:{manager._base_port}"
        }
        
    except Exception as e:
        logger.error(f"Error listing MCP servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list MCP servers: {str(e)}")


@router.post("/mcp-servers/{configuration_name}/tools/execute", response_model=MCPToolExecuteResponse)
async def execute_mcp_tool(configuration_name: str, request: MCPToolExecuteRequest):
    """Execute an MCP tool for a specific configuration."""
    try:
        manager = get_mcp_manager()
        
        # Check if configuration exists
        if configuration_name not in manager.configurations:
            raise HTTPException(
                status_code=404,
                detail=f"MCP server for configuration '{configuration_name}' not found"
            )
        
        # Execute the tool
        result = await manager._execute_tool(
            configuration_name, 
            request.tool_name, 
            request.parameters
        )
        
        return {
            "success": not result.get("isError", False),
            "result": result,
            "tool_name": request.tool_name,
            "configuration_name": configuration_name,
            "execution_time": result.get("executionTime")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing MCP tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute tool: {str(e)}")


@router.get("/mcp-servers/{configuration_name}/tools")
async def list_mcp_tools(configuration_name: str):
    """List available tools for a specific MCP configuration."""
    try:
        manager = get_mcp_manager()
        
        if configuration_name not in manager.configurations:
            raise HTTPException(
                status_code=404,
                detail=f"MCP server for configuration '{configuration_name}' not found"
            )
        
        tools_schema = manager._get_tools_schema(configuration_name)
        
        return {
            "configuration_name": configuration_name,
            "tools": tools_schema,
            "total_tools": len(tools_schema)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing MCP tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@router.post("/mcp-servers/shutdown-all")
async def shutdown_all_mcp_servers():
    """Shutdown all MCP servers."""
    try:
        manager = get_mcp_manager()
        await manager.shutdown_all()
        
        return {
            "success": True,
            "message": "All MCP servers shut down successfully"
        }
        
    except Exception as e:
        logger.error(f"Error shutting down MCP servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to shutdown servers: {str(e)}")


@router.get("/mcp-servers/hub/status")
async def get_hub_status():
    """Get overall MCP hub status."""
    try:
        manager = get_mcp_manager()
        
        return {
            "running": manager._is_running,
            "port": manager._base_port,
            "host": manager._base_host,
            "active_configurations": list(manager.configurations.keys()),
            "total_configurations": len(manager.configurations),
            "hub_url": f"http://{manager._base_host}:{manager._base_port}",
            "start_time": manager._start_time.isoformat() if manager._start_time else None
        }
        
    except Exception as e:
        logger.error(f"Error getting hub status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get hub status: {str(e)}")
