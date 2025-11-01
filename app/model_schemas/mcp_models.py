"""
MCP Server Model Schemas

Pydantic models for MCP server API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


class MCPServerStartRequest(BaseModel):
    """Request model for starting an MCP server."""
    configuration_name: str = Field(..., description="Name of the RAG configuration")
    force_restart: bool = Field(default=False, description="Force restart if already running")


class MCPServerStopRequest(BaseModel):
    """Request model for stopping an MCP server."""
    configuration_name: str = Field(..., description="Name of the RAG configuration")


class MCPToolExecuteRequest(BaseModel):
    """Request model for executing an MCP tool."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class MCPToolExecuteResponse(BaseModel):
    """Response model for MCP tool execution."""
    success: bool = Field(..., description="Whether the tool execution was successful")
    result: Dict[str, Any] = Field(..., description="Tool execution result")
    tool_name: str = Field(..., description="Name of the executed tool")
    configuration_name: str = Field(..., description="Configuration name used")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")


class MCPServerStatusResponse(BaseModel):
    """Response model for MCP server status."""
    configuration_name: str = Field(..., description="Configuration name")
    enabled: bool = Field(..., description="Whether MCP server is enabled for this configuration")
    running: bool = Field(..., description="Whether MCP server is currently running")
    protocols: List[str] = Field(..., description="Enabled MCP protocols")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    tools: List[Dict[str, Any]] = Field(..., description="Available tools")
    client_count: int = Field(..., description="Number of connected clients")
    uptime_seconds: Optional[float] = Field(None, description="Server uptime in seconds")
    message: str = Field(..., description="Status message")


class MCPServerListResponse(BaseModel):
    """Response model for listing MCP servers."""
    servers: List[MCPServerStatusResponse] = Field(..., description="List of MCP server statuses")
    total_count: int = Field(..., description="Total number of servers")
    running: bool = Field(..., description="Whether the MCP hub is running")
    port: int = Field(..., description="MCP hub port")
    host: str = Field(..., description="MCP hub host")
    active_configurations: List[str] = Field(..., description="List of active configuration names")
    hub_url: str = Field(..., description="MCP hub base URL")


class MCPHubStatusResponse(BaseModel):
    """Response model for MCP hub status."""
    running: bool = Field(..., description="Whether the MCP hub is running")
    port: int = Field(..., description="MCP hub port")
    host: str = Field(..., description="MCP hub host")
    active_configurations: List[str] = Field(..., description="List of active configuration names")
    total_configurations: int = Field(..., description="Total number of configurations")
    hub_url: str = Field(..., description="MCP hub base URL")
    start_time: Optional[str] = Field(None, description="Hub start time (ISO format)")


class MCPToolSchema(BaseModel):
    """Schema for an MCP tool."""
    name: str = Field(..., description="Tool name")
    type: str = Field(..., description="Tool type")
    enabled: bool = Field(..., description="Whether tool is enabled")
    description: str = Field(..., description="Tool description")
    parameters_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for parameters")
    max_results: int = Field(..., description="Maximum number of results")


class MCPToolListResponse(BaseModel):
    """Response model for listing MCP tools."""
    configuration_name: str = Field(..., description="Configuration name")
    tools: List[MCPToolSchema] = Field(..., description="Available tools")
    total_tools: int = Field(..., description="Total number of tools")


class MCPServerResponse(BaseModel):
    """Generic MCP server response."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    configuration_name: str = Field(..., description="Configuration name")
    endpoints: Optional[Dict[str, str]] = Field(None, description="Available endpoints")
    error_details: Optional[str] = Field(None, description="Detailed error information")
