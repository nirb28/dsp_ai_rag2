"""
Complete MCP Server Implementation

This module contains the actual MCP server logic that was moved from the main RAG service.
It implements the full MCP protocol and server management functionality.
"""

import logging
import asyncio
import json
import datetime
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import aiohttp

logger = logging.getLogger(__name__)


class MCPServerImpl:
    """Full MCP server implementation with protocol handling."""
    
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.configurations: Dict[str, Any] = {}
        self.client_connections: Dict[str, List[WebSocket]] = {}
        self._is_running = False
        self._base_port = 8080
        self._base_host = "localhost"
        self._start_time = None
        
    async def start_server(self, configuration_name: str, force_restart: bool = False) -> Dict[str, Any]:
        """Start MCP server for a configuration."""
        try:
            rag_config = self.rag_service.get_configuration(configuration_name)
            logger.info(f"Loaded config for '{configuration_name}', MCP server enabled: {rag_config.mcp_server and rag_config.mcp_server.enabled}")
            
            if not rag_config.mcp_server or not rag_config.mcp_server.enabled:
                raise ValueError(f"MCP server not enabled for configuration '{configuration_name}'")
            
            logger.info(f"MCP server tools for '{configuration_name}': {[tool.name for tool in rag_config.mcp_server.tools]}")
            
            if force_restart and configuration_name in self.configurations:
                await self.stop_server(configuration_name)
            
            if configuration_name in self.configurations:
                return {
                    "success": False, 
                    "message": f"MCP server for '{configuration_name}' is already running"
                }
            
            # Add configuration
            self.configurations[configuration_name] = rag_config.mcp_server
            self.client_connections[configuration_name] = []
            
            # Start shared server if not running
            if not self._is_running:
                await self._start_shared_server()
            
            endpoints = self._get_endpoints(configuration_name)
            
            return {
                "success": True,
                "message": f"MCP server for '{configuration_name}' started successfully",
                "endpoints": endpoints
            }
            
        except Exception as e:
            logger.error(f"Error starting MCP server for '{configuration_name}': {str(e)}")
            return {
                "success": False,
                "message": f"Failed to start MCP server: {str(e)}"
            }
    
    async def stop_server(self, configuration_name: str) -> Dict[str, Any]:
        """Stop MCP server for a configuration."""
        try:
            if configuration_name not in self.configurations:
                return {
                    "success": False,
                    "message": f"No MCP server found for configuration '{configuration_name}'"
                }
            
            # Close all client connections
            for connection in list(self.client_connections[configuration_name]):
                try:
                    await connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {str(e)}")
            
            # Remove configuration
            del self.client_connections[configuration_name]
            del self.configurations[configuration_name]
            
            # Stop shared server if no configurations remain
            if not self.configurations and self._is_running:
                await self._stop_shared_server()
            
            return {
                "success": True,
                "message": f"MCP server for '{configuration_name}' stopped successfully"
            }
            
        except Exception as e:
            logger.error(f"Error stopping MCP server for '{configuration_name}': {str(e)}")
            return {
                "success": False,
                "message": f"Failed to stop MCP server: {str(e)}"
            }
    
    def get_server_status(self, configuration_name: str) -> Dict[str, Any]:
        """Get server status for a configuration."""
        config = self.configurations.get(configuration_name)
        if not config:
            return {
                "configuration_name": configuration_name,
                "enabled": False,
                "running": False,
                "protocols": [],
                "endpoints": {},
                "tools": [],
                "client_count": 0,
                "message": "Server not found or not running"
            }
        
        tools = self._get_tools_schema(configuration_name)
        endpoints = self._get_endpoints(configuration_name)
        
        return {
            "configuration_name": configuration_name,
            "enabled": True,
            "running": True,
            "protocols": ["http", "websocket", "sse"],
            "endpoints": endpoints,
            "tools": tools,
            "client_count": len(self.client_connections.get(configuration_name, [])),
            "message": "Server running"
        }
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all active MCP servers."""
        servers = []
        for config_name in self.configurations:
            servers.append(self.get_server_status(config_name))
        return servers
    
    async def shutdown_all(self):
        """Shutdown all MCP servers."""
        config_names = list(self.configurations.keys())
        for config_name in config_names:
            await self.stop_server(config_name)
        
        if self._is_running:
            await self._stop_shared_server()
    
    async def _start_shared_server(self):
        """Start the shared MCP server."""
        self._is_running = True
        self._start_time = datetime.datetime.now()
        logger.info(f"MCP shared server started on {self._base_host}:{self._base_port}")
    
    async def _stop_shared_server(self):
        """Stop the shared MCP server."""
        self._is_running = False
        self._start_time = None
        logger.info("MCP shared server stopped")
    
    def _get_endpoints(self, configuration_name: str) -> Dict[str, str]:
        """Get endpoints for a configuration."""
        base_url = f"http://{self._base_host}:{self._base_port}"
        return {
            "mcp": f"{base_url}/{configuration_name}/mcp",
            "sse": f"{base_url}/{configuration_name}/sse",
            "websocket": f"ws://{self._base_host}:{self._base_port}/{configuration_name}/ws",
            "info": f"{base_url}/{configuration_name}/info",
            "tools": f"{base_url}/mcp-servers/{configuration_name}/tools"
        }
    
    def _get_tools_schema(self, configuration_name: str) -> List[Dict[str, Any]]:
        """Get tools schema for a configuration."""
        config = self.configurations.get(configuration_name)
        if not config or not config.tools:
            return []
        
        tools = []
        for tool in config.tools:
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for document retrieval"
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": f"Similarity threshold (default: {getattr(tool, 'similarity_threshold', 0.5)})",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": getattr(tool, 'similarity_threshold', 0.5)
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of documents to retrieve",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
            tools.append(tool_schema)
        
        return tools
    
    async def _execute_tool(self, configuration_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool."""
        try:
            config = self.configurations.get(configuration_name)
            if not config:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Configuration '{configuration_name}' not found"}]
                }
            
            # Find the tool
            tool_config = None
            for tool in config.tools:
                if tool.name == tool_name:
                    tool_config = tool
                    break
            
            if not tool_config:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Tool '{tool_name}' not found"}]
                }
            
            # Extract parameters
            query = parameters.get("query", "")
            similarity_threshold = parameters.get("similarity_threshold", getattr(tool_config, 'similarity_threshold', 0.5))
            k = parameters.get("k", getattr(tool_config, 'max_results', 10))
            
            if not query:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": "Query parameter is required"}]
                }
            
            # Execute the query based on tool type, not tool name
            start_time = datetime.datetime.now()
            
            if tool_config.type == "retrieve":
                results = await self.rag_service.retrieve(
                    configuration_name=configuration_name,
                    query=query,
                    similarity_threshold=similarity_threshold,
                    k=k
                )
            else:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Unknown tool type: {tool_config.type}"}]
                }
            
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Format results - handle all possible response formats
            content = []
            
            # RAGService.retrieve() returns (documents, metadata) tuple
            if isinstance(results, tuple) and len(results) >= 1:
                documents = results[0]
            # Handle direct list of documents
            elif isinstance(results, list):
                documents = results
            # Handle dictionary format (for query method)
            elif isinstance(results, dict) and "results" in results:
                documents = results["results"]
            else:
                # Handle None, empty, or unexpected formats
                documents = []
            
            # Process documents if any exist
            if documents and len(documents) > 0:
                for i, result in enumerate(documents[:k]):
                    content.append({
                        "type": "text",
                        "text": f"Document {i+1}:\n"
                               f"Content: {result.get('content', 'N/A')}\n"
                               f"Score: {result.get('score', 'N/A')}\n"
                               f"Metadata: {json.dumps(result.get('metadata', {}), indent=2)}\n"
                    })
            else:
                # No documents found
                content.append({
                    "type": "text", 
                    "text": "No results found"
                })
            
            return {
                "isError": False,
                "content": content,
                "executionTime": execution_time
            }
                
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error executing tool: {str(e)}"}]
            }
    
    async def _handle_mcp_request(self, configuration_name: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle MCP JSON-RPC request."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            # Handle notifications (no response expected)
            if request_id is None:
                if method == "notifications/initialized":
                    # Client acknowledges initialization - no response needed
                    logger.info(f"Client initialized for configuration '{configuration_name}'")
                    return None
                else:
                    # Unknown notification - log and ignore
                    logger.warning(f"Unknown notification method: {method}")
                    return None
            
            # Handle requests (response expected)
            if method == "initialize":
                # MCP initialization handshake
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {"listChanged": True},
                            "resources": {}
                        },
                        "serverInfo": {
                            "name": f"RAG MCP Server - {configuration_name}",
                            "version": "1.0.0"
                        }
                    }
                }
            
            elif method == "ping":
                # Ping/pong for connection health
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {}
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                result = await self._execute_tool(configuration_name, tool_name, arguments)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            
            elif method == "tools/list":
                tools = self._get_tools_schema(configuration_name)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": tools}
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error handling MCP request: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def _handle_websocket(self, configuration_name: str, websocket: WebSocket):
        """Handle WebSocket connection."""
        await websocket.accept()
        
        if configuration_name not in self.configurations:
            await websocket.close(code=1008, reason="Configuration not found")
            return
        
        self.client_connections[configuration_name].append(websocket)
        
        try:
            while True:
                data = await websocket.receive_text()
                request = json.loads(data)
                response = await self._handle_mcp_request(configuration_name, request)
                
                # Only send response if one is expected (not for notifications)
                if response is not None:
                    await websocket.send_text(json.dumps(response))
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            if websocket in self.client_connections[configuration_name]:
                self.client_connections[configuration_name].remove(websocket)
    
    def _handle_sse_stream(self, configuration_name: str):
        """Handle Server-Sent Events stream.
        Return an async generator instance suitable for StreamingResponse.
        """
        async def event_generator():
            if configuration_name not in self.configurations:
                yield f"data: {json.dumps({'error': 'Configuration not found'})}\n\n"
                return
            
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'configuration': configuration_name})}\n\n"
            
            # Keep connection alive
            while True:
                await asyncio.sleep(30)  # Send keepalive every 30 seconds
                yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': datetime.datetime.now().isoformat()})}\n\n"
        
        # Return the async generator instance (not a coroutine nor the function itself)
        return event_generator()
