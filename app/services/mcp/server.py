import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.routing import APIRouter
import uvicorn
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor

from app.config import MCPProtocol, MCPToolType, MCPServerConfig, MCPToolConfig
from app.services.rag_service import RAGService
from app.model_schemas.base_models import RetrieveRequest

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manager for MCP servers across different configurations."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.servers: Dict[str, 'MCPServer'] = {}
        self.server_threads: Dict[str, threading.Thread] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    async def start_server(self, configuration_name: str, force_restart: bool = False) -> Dict[str, Any]:
        """Start MCP server for a configuration."""
        try:
            # Get RAG configuration
            rag_config = self.rag_service.get_configuration(configuration_name)
            if not rag_config.mcp_server or not rag_config.mcp_server.enabled:
                raise ValueError(f"MCP server not enabled for configuration '{configuration_name}'")
            
            # Stop existing server if force restart
            if force_restart and configuration_name in self.servers:
                await self.stop_server(configuration_name)
            
            # Check if already running
            if configuration_name in self.servers and self.servers[configuration_name].is_running:
                return {
                    "success": False,
                    "message": f"MCP server for '{configuration_name}' is already running"
                }
            
            # Create and start server
            mcp_server = MCPServer(configuration_name, rag_config.mcp_server, self.rag_service)
            await mcp_server.start()
            
            self.servers[configuration_name] = mcp_server
            
            return {
                "success": True,
                "message": f"MCP server for '{configuration_name}' started successfully",
                "endpoints": mcp_server.get_endpoints()
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
            if configuration_name not in self.servers:
                return {
                    "success": False,
                    "message": f"No MCP server found for configuration '{configuration_name}'"
                }
            
            server = self.servers[configuration_name]
            await server.stop()
            del self.servers[configuration_name]
            
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
        """Get status of MCP server for a configuration."""
        if configuration_name not in self.servers:
            return {
                "configuration_name": configuration_name,
                "enabled": False,
                "running": False,
                "protocols": [],
                "endpoints": {},
                "tools": [],
                "client_count": 0,
                "message": "Server not found"
            }
        
        server = self.servers[configuration_name]
        return server.get_status()
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all MCP servers and their status."""
        servers = []
        
        # Get all configurations with MCP enabled
        for config_name in self.rag_service.configurations.keys():
            try:
                config = self.rag_service.get_configuration(config_name)
                if config.mcp_server and config.mcp_server.enabled:
                    servers.append(self.get_server_status(config_name))
            except Exception as e:
                logger.warning(f"Error checking configuration '{config_name}': {str(e)}")
        
        return servers
    
    async def shutdown_all(self):
        """Shutdown all MCP servers."""
        for config_name in list(self.servers.keys()):
            await self.stop_server(config_name)


class MCPServer:
    """Individual MCP server instance for a configuration."""
    
    def __init__(self, configuration_name: str, mcp_config: MCPServerConfig, rag_service: RAGService):
        self.configuration_name = configuration_name
        self.mcp_config = mcp_config
        self.rag_service = rag_service
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.client_connections: Set[Any] = set()
        self.app: Optional[FastAPI] = None
        self.server_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the MCP server with configured protocols."""
        if self.is_running:
            return
        
        self.start_time = datetime.now(timezone.utc)
        self.is_running = True
        
        # Create FastAPI app for HTTP and SSE protocols
        if MCPProtocol.HTTP in self.mcp_config.protocols or MCPProtocol.SSE in self.mcp_config.protocols:
            await self._setup_http_server()
        
        # TODO: Implement stdio protocol support
        if MCPProtocol.STDIO in self.mcp_config.protocols:
            logger.info(f"STDIO protocol support not yet implemented for '{self.configuration_name}'")
    
    async def _setup_http_server(self):
        """Setup HTTP/SSE server."""
        self.app = FastAPI(
            title=f"MCP Server - {self.mcp_config.name}",
            description=self.mcp_config.description,
            version=self.mcp_config.version
        )
        
        # Add MCP endpoints
        router = APIRouter(prefix=f"/{self.configuration_name}")
        
        # HTTP JSON-RPC endpoint
        if MCPProtocol.HTTP in self.mcp_config.protocols:
            @router.post("/mcp")
            async def handle_mcp_request(request: Dict[str, Any]):
                return await self._handle_mcp_request(request)
        
        # SSE endpoint
        if MCPProtocol.SSE in self.mcp_config.protocols:
            @router.get("/sse")
            async def handle_sse():
                return StreamingResponse(
                    self._handle_sse_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
        
        # WebSocket endpoint for real-time communication
        @router.websocket("/ws")
        async def handle_websocket(websocket: WebSocket):
            await self._handle_websocket(websocket)
        
        # Server info endpoint
        @router.get("/info")
        async def get_server_info():
            return self.get_status()
        
        self.app.include_router(router)
        
        # Start server in background
        config = uvicorn.Config(
            self.app,
            host=self.mcp_config.http_host,
            port=self.mcp_config.http_port + hash(self.configuration_name) % 1000,  # Unique port per config
            log_level="info"
        )
        server = uvicorn.Server(config)
        self.server_task = asyncio.create_task(server.serve())
    
    async def _handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP JSON-RPC request."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": self._get_tools_schema()}
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = await self._execute_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            
            elif method == "initialize":
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
                            "name": self.mcp_config.name,
                            "version": self.mcp_config.version
                        }
                    }
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
    
    async def _handle_sse_stream(self):
        """Handle SSE stream for real-time updates."""
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'server': self.mcp_config.name})}\n\n"
            
            # Keep connection alive
            while self.is_running:
                # Send heartbeat every 30 seconds
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
                await asyncio.sleep(30)
                
        except Exception as e:
            logger.error(f"Error in SSE stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection."""
        await websocket.accept()
        self.client_connections.add(websocket)
        
        try:
            while True:
                data = await websocket.receive_text()
                request = json.loads(data)
                response = await self._handle_mcp_request(request)
                await websocket.send_text(json.dumps(response))
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            self.client_connections.discard(websocket)
    
    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get MCP tools schema."""
        tools = []
        
        for tool_config in self.mcp_config.tools:
            if not tool_config.enabled:
                continue
                
            tool_schema = {
                "name": tool_config.name,
                "description": tool_config.description,
                "inputSchema": tool_config.parameters_schema or self._get_default_schema(tool_config.type)
            }
            tools.append(tool_schema)
        
        return tools
    
    def _get_default_schema(self, tool_type: MCPToolType) -> Dict[str, Any]:
        """Get default parameter schema for tool type."""
        if tool_type == MCPToolType.RETRIEVE:
            return {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "description": "Number of results", "default": 5},
                    "similarity_threshold": {"type": "number", "description": "Similarity threshold", "default": 0.7},
                    "filter": {"type": "object", "description": "Metadata filter", "default": {}}
                },
                "required": ["query"]
            }
        else:
            return {"type": "object", "properties": {}}
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool."""
        start_time = time.time()
        
        try:
            # Find tool config
            tool_config = None
            for config in self.mcp_config.tools:
                if config.name == tool_name and config.enabled:
                    tool_config = config
                    break
            
            if not tool_config:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Tool '{tool_name}' not found or disabled"}]
                }
            
            # Execute based on tool type
            if tool_config.type == MCPToolType.RETRIEVE:
                result = await self._execute_retrieve_tool(arguments, tool_config)
            else:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Unknown tool type: {tool_config.type}"}]
                }
            
            execution_time = time.time() - start_time
            
            return {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                "executionTime": execution_time
            }
            
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool execution error: {str(e)}"}]
            }
    
    async def _execute_retrieve_tool(self, arguments: Dict[str, Any], tool_config: MCPToolConfig) -> Dict[str, Any]:
        """Execute retrieve tool."""
        query = arguments.get("query", "")
        k = min(arguments.get("k", 5), tool_config.max_results)
        similarity_threshold = arguments.get("similarity_threshold", 0.7)
        filter_dict = arguments.get("filter", {})
        
        # Create retrieve request
        request = RetrieveRequest(
            query=query,
            configuration_name=self.configuration_name,
            k=k,
            similarity_threshold=similarity_threshold,
            include_metadata=tool_config.include_metadata,
            filter=filter_dict if filter_dict else None
        )
        
        # Execute retrieval
        documents, _ = await self.rag_service.retrieve(
            query=request.query,
            configuration_name=request.configuration_name,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
            filter_metadata=request.filter,
            include_metadata=request.include_metadata
        )
        
        return {
            "query": query,
            "documents": documents,
            "total_found": len(documents),
            "configuration_name": self.configuration_name
        }
    
    
    def get_endpoints(self) -> Dict[str, str]:
        """Get server endpoints."""
        endpoints = {}
        base_url = f"http://{self.mcp_config.http_host}:{self.mcp_config.http_port + hash(self.configuration_name) % 1000}/{self.configuration_name}"
        
        if MCPProtocol.HTTP in self.mcp_config.protocols:
            endpoints["http"] = f"{base_url}/mcp"
        if MCPProtocol.SSE in self.mcp_config.protocols:
            endpoints["sse"] = f"{base_url}/sse"
        if "ws" not in endpoints:  # Always include WebSocket
            endpoints["websocket"] = f"{base_url.replace('http://', 'ws://')}/ws"
        
        return endpoints
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        uptime = None
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        tools = []
        for tool_config in self.mcp_config.tools:
            tools.append({
                "name": tool_config.name,
                "type": tool_config.type.value,
                "enabled": tool_config.enabled,
                "description": tool_config.description
            })
        
        return {
            "configuration_name": self.configuration_name,
            "enabled": self.mcp_config.enabled,
            "running": self.is_running,
            "protocols": [p.value for p in self.mcp_config.protocols],
            "endpoints": self.get_endpoints() if self.is_running else {},
            "tools": tools,
            "client_count": len(self.client_connections),
            "uptime_seconds": uptime,
            "message": "Running" if self.is_running else "Stopped"
        }
    
    async def stop(self):
        """Stop the MCP server."""
        self.is_running = False
        
        # Close client connections
        for connection in list(self.client_connections):
            try:
                await connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {str(e)}")
        
        self.client_connections.clear()
        
        # Stop server task
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        
        self.start_time = None
