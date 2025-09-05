from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime


class MCPServerStatusResponse(BaseModel):
    """Response model for MCP server status."""
    configuration_name: str
    enabled: bool
    running: bool
    protocols: List[str]
    endpoints: Dict[str, str]  # protocol -> endpoint mapping
    tools: List[Dict[str, Any]]
    client_count: int = 0
    uptime_seconds: Optional[float] = None
    message: str


class MCPServerListResponse(BaseModel):
    """Response model for listing MCP servers."""
    servers: List[MCPServerStatusResponse]
    total_count: int


class MCPToolExecutionRequest(BaseModel):
    """Request model for MCP tool execution."""
    tool_name: str
    parameters: Dict[str, Any]


class MCPToolExecutionResponse(BaseModel):
    """Response model for MCP tool execution."""
    tool_name: str
    result: Any
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class MCPServerStartRequest(BaseModel):
    """Request model for starting MCP server."""
    configuration_name: str
    force_restart: bool = False


class MCPServerStopRequest(BaseModel):
    """Request model for stopping MCP server."""
    configuration_name: str


class MCPServerStartStopResponse(BaseModel):
    """Response model for MCP server start/stop operations."""
    configuration_name: str
    action: str  # "started" or "stopped"
    success: bool
    message: str
    endpoints: Optional[Dict[str, str]] = None
