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
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor

from app.config import MCPProtocol, MCPToolType, MCPServerConfig, MCPToolConfig
from app.services.rag_service import RAGService
from app.model_schemas.base_models import RetrieveRequest, QueryRequest

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manager for MCP servers across different configurations using single port with path-based routing."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.configurations: Dict[str, MCPServerConfig] = {}
        self.client_connections: Dict[str, Set[WebSocket]] = {}
        self._app: Optional[FastAPI] = None
        self._server_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._base_port = 8080
        self._base_host = "localhost"
        self._start_time: Optional[datetime] = None
        
    async def start_server(self, configuration_name: str, force_restart: bool = False) -> Dict[str, Any]:
        """Start MCP server for a configuration (adds to shared server)."""
        try:
            # Get RAG configuration
            rag_config = self.rag_service.get_configuration(configuration_name)
            if not rag_config.mcp_server or not rag_config.mcp_server.enabled:
                raise ValueError(f"MCP server not enabled for configuration '{configuration_name}'")
            
            # Stop existing configuration if force restart
            if force_restart and configuration_name in self.configurations:
                await self.stop_server(configuration_name)
            
            # Check if already running
            if configuration_name in self.configurations:
                return {
                    "success": False,
                    "message": f"MCP server for '{configuration_name}' is already running"
                }
            
            # Add configuration to shared server
            self.configurations[configuration_name] = rag_config.mcp_server
            self.client_connections[configuration_name] = set()
            
            # Start the shared server if not already running
            if not self._is_running:
                await self._start_shared_server()
            
            return {
                "success": True,
                "message": f"MCP server for '{configuration_name}' started successfully",
                "endpoints": self._get_endpoints(configuration_name)
            }
            
        except Exception as e:
            logger.error(f"Error starting MCP server for '{configuration_name}': {str(e)}")
            return {
                "success": False,
                "message": f"Failed to start MCP server: {str(e)}"
            }
    
    async def stop_server(self, configuration_name: str) -> Dict[str, Any]:
        """Stop MCP server for a configuration (removes from shared server)."""
        try:
            if configuration_name not in self.configurations:
                return {
                    "success": False,
                    "message": f"No MCP server found for configuration '{configuration_name}'"
                }
            
            # Close client connections for this configuration
            if configuration_name in self.client_connections:
                for connection in list(self.client_connections[configuration_name]):
                    try:
                        await connection.close()
                    except Exception as e:
                        logger.warning(f"Error closing connection: {str(e)}")
                del self.client_connections[configuration_name]
            
            # Remove configuration
            del self.configurations[configuration_name]
            
            # If no configurations left, stop the shared server
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
        """Get status of MCP server for a configuration."""
        if configuration_name not in self.configurations:
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
        
        mcp_config = self.configurations[configuration_name]
        
        tools = []
        for tool_config in mcp_config.tools:
            tools.append({
                "name": tool_config.name,
                "type": tool_config.type.value,
                "enabled": tool_config.enabled,
                "description": tool_config.description
            })
        
        uptime = None
        if self._is_running and self._start_time:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        
        return {
            "configuration_name": configuration_name,
            "enabled": mcp_config.enabled,
            "running": self._is_running,
            "protocols": [p.value for p in mcp_config.protocols],
            "endpoints": self._get_endpoints(configuration_name) if self._is_running else {},
            "tools": tools,
            "client_count": len(self.client_connections.get(configuration_name, set())),
            "uptime_seconds": uptime,
            "message": "Running" if self._is_running else "Stopped"
        }
    
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
        for config_name in list(self.configurations.keys()):
            await self.stop_server(config_name)
    
    async def _start_shared_server(self):
        """Start the shared FastAPI server."""
        if self._is_running:
            return
        
        self._app = FastAPI(
            title="MCP Server Hub",
            description="Multi-configuration MCP server with path-based routing",
            version="1.0.0"
        )
        
        # Add CORS middleware to allow MCP client connections
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins for MCP clients
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Add dynamic routing for all MCP endpoints
        @self._app.post("/{configuration_name}/mcp")
        async def handle_mcp_request(configuration_name: str, request: Dict[str, Any]):
            return await self._handle_mcp_request(configuration_name, request)
        
        @self._app.get("/{configuration_name}/sse")
        async def handle_sse(configuration_name: str):
            return StreamingResponse(
                self._handle_sse_stream(configuration_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        @self._app.websocket("/{configuration_name}/ws")
        async def handle_websocket(configuration_name: str, websocket: WebSocket):
            await self._handle_websocket(configuration_name, websocket)
        
        @self._app.get("/{configuration_name}/info")
        async def get_server_info(configuration_name: str):
            return self.get_server_status(configuration_name)
        
        # Global endpoints
        @self._app.get("/")
        async def root():
            return {
                "message": "MCP Server Hub",
                "active_configurations": list(self.configurations.keys()),
                "total_configurations": len(self.configurations)
            }
        
        @self._app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "configurations": len(self.configurations),
                "running": self._is_running
            }
        
        # Start server
        config = uvicorn.Config(
            self._app,
            host=self._base_host,
            port=self._base_port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        self._is_running = True
        self._start_time = datetime.now(timezone.utc)
        
        logger.info(f"MCP Server Hub started on {self._base_host}:{self._base_port}")
    
    async def _stop_shared_server(self):
        """Stop the shared FastAPI server."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Close all client connections
        for config_connections in self.client_connections.values():
            for connection in list(config_connections):
                try:
                    await connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {str(e)}")
        
        self.client_connections.clear()
        
        # Stop server task
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MCP Server Hub stopped")
    
    def _get_endpoints(self, configuration_name: str) -> Dict[str, str]:
        """Get server endpoints for a configuration."""
        if not self._is_running or configuration_name not in self.configurations:
            return {}
        
        mcp_config = self.configurations[configuration_name]
        base_url = f"http://{self._base_host}:{self._base_port}/{configuration_name}"
        
        endpoints = {}
        if MCPProtocol.HTTP in mcp_config.protocols:
            endpoints["http"] = f"{base_url}/mcp"
        if MCPProtocol.SSE in mcp_config.protocols:
            endpoints["sse"] = f"{base_url}/sse"
        endpoints["websocket"] = f"{base_url.replace('http://', 'ws://')}/ws"
        endpoints["info"] = f"{base_url}/info"
        
        return endpoints


    async def _handle_mcp_request(self, configuration_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP JSON-RPC request for a specific configuration."""
        if configuration_name not in self.configurations:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32001,
                    "message": f"Configuration '{configuration_name}' not found"
                }
            }
        
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": self._get_tools_schema(configuration_name)}
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
            
            elif method == "initialize":
                mcp_config = self.configurations[configuration_name]
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
                            "name": mcp_config.name,
                            "version": mcp_config.version
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
    
    async def _handle_sse_stream(self, configuration_name: str):
        """Handle SSE stream for real-time updates."""
        if configuration_name not in self.configurations:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Configuration {configuration_name} not found'})}\n\n"
            return
        
        try:
            mcp_config = self.configurations[configuration_name]
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'server': mcp_config.name, 'configuration': configuration_name})}\n\n"
            
            # Keep connection alive
            while self._is_running and configuration_name in self.configurations:
                # Send heartbeat every 30 seconds
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat(), 'configuration': configuration_name})}\n\n"
                await asyncio.sleep(30)
                
        except Exception as e:
            logger.error(f"Error in SSE stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    async def _handle_websocket(self, configuration_name: str, websocket: WebSocket):
        """Handle WebSocket connection for a specific configuration."""
        if configuration_name not in self.configurations:
            await websocket.close(code=4004, reason=f"Configuration '{configuration_name}' not found")
            return
        
        await websocket.accept()
        self.client_connections[configuration_name].add(websocket)
        
        try:
            while True:
                data = await websocket.receive_text()
                request = json.loads(data)
                response = await self._handle_mcp_request(configuration_name, request)
                await websocket.send_text(json.dumps(response))
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            self.client_connections[configuration_name].discard(websocket)
    
    def _get_tools_schema(self, configuration_name: str) -> List[Dict[str, Any]]:
        """Get MCP tools schema for a configuration."""
        if configuration_name not in self.configurations:
            return []
        
        mcp_config = self.configurations[configuration_name]
        tools = []
        
        for tool_config in mcp_config.tools:
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
        elif tool_type == MCPToolType.SEARCH:
            return {
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum results", "default": 10}
                },
                "required": ["query"]
            }
        elif tool_type == MCPToolType.LIST_DOCUMENTS:
            return {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum documents to list", "default": 20}
                }
            }
        else:
            return {"type": "object", "properties": {}}
    
    async def _execute_tool(self, configuration_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool for a specific configuration."""
        start_time = time.time()
        
        try:
            if configuration_name not in self.configurations:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Configuration '{configuration_name}' not found"}]
                }
            
            mcp_config = self.configurations[configuration_name]
            
            # Find tool config
            tool_config = None
            for config in mcp_config.tools:
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
                result = await self._execute_retrieve_tool(configuration_name, arguments, tool_config)
            elif tool_config.type == MCPToolType.SEARCH:
                result = await self._execute_search_tool(configuration_name, arguments, tool_config)
            elif tool_config.type == MCPToolType.LIST_DOCUMENTS:
                result = await self._execute_list_documents_tool(configuration_name, arguments, tool_config)
            elif tool_config.type == MCPToolType.QUERY:
                result = await self._execute_query_tool(configuration_name, arguments, tool_config)
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
    
    async def _execute_retrieve_tool(self, configuration_name: str, arguments: Dict[str, Any], tool_config: MCPToolConfig) -> Dict[str, Any]:
        """Execute retrieve tool."""
        query = arguments.get("query", "")
        k = min(arguments.get("k", 5), tool_config.max_results)
        similarity_threshold = arguments.get("similarity_threshold", 0.7)
        filter_dict = arguments.get("filter", {})
        
        # Create retrieve request
        request = RetrieveRequest(
            query=query,
            configuration_name=configuration_name,
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
            filter=request.filter
        )
        
        
        return {
            "query": query,
            "documents": documents,
            "total_found": len(documents),
            "configuration_name": configuration_name
        }
    
    async def _execute_search_tool(self, configuration_name: str, arguments: Dict[str, Any], tool_config: MCPToolConfig) -> Dict[str, Any]:
        """Execute search tool (same as retrieve but different interface)."""
        return await self._execute_retrieve_tool(configuration_name, arguments, tool_config)
    
    async def _execute_list_documents_tool(self, configuration_name: str, arguments: Dict[str, Any], tool_config: MCPToolConfig) -> Dict[str, Any]:
        """Execute list documents tool."""
        limit = min(arguments.get("limit", 20), tool_config.max_results)
        
        try:
            vector_store = self.rag_service._get_vector_store(configuration_name)
            documents = vector_store.get_all_documents(limit=limit)
            
            # Extract unique document IDs and metadata
            unique_docs = {}
            for doc in documents:
                doc_id = doc.metadata.get('document_id', 'unknown')
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        'document_id': doc_id,
                        'metadata': doc.metadata,
                        'chunk_count': 0
                    }
                unique_docs[doc_id]['chunk_count'] += 1
            
            return {
                "documents": list(unique_docs.values()),
                "total_unique_documents": len(unique_docs),
                "configuration_name": configuration_name
            }
            
        except Exception as e:
            raise Exception(f"Failed to list documents: {str(e)}")
    
    async def _execute_query_tool(self, configuration_name: str, arguments: Dict[str, Any], tool_config: MCPToolConfig) -> Dict[str, Any]:
        """Execute query tool with LLM generation."""
        query = arguments.get("query", "")
        k = min(arguments.get("k", 5), tool_config.max_results)
        
        # Create query request
        request = QueryRequest(
            query=query,
            configuration_name=configuration_name,
            k=k,
            include_metadata=tool_config.include_metadata
        )
        
        # Execute query
        response = await self.rag_service.query(
            query=request.query,
            configuration_name=request.configuration_name,
            k=request.k
        )
        
        return {
            "query": query,
            "answer": response["answer"],
            "sources": response["sources"],
            "configuration_name": configuration_name
        }


# Keep MCPServer class for backward compatibility (now just a simple wrapper)
class MCPServer:
    """Legacy compatibility class - functionality moved to MCPServerManager."""
    
    def __init__(self, configuration_name: str, mcp_config: MCPServerConfig, rag_service: RAGService):
        logger.warning("MCPServer class is deprecated. Use MCPServerManager directly.")
        self.configuration_name = configuration_name
        self.mcp_config = mcp_config
        self.rag_service = rag_service
        self.is_running = False
        self.start_time = None
        self.client_connections = set()
        self.app = None
        self.server_task = None
        
    async def start(self):
        """Start method - deprecated, use MCPServerManager instead."""
        raise DeprecationWarning("MCPServer.start() is deprecated. Use MCPServerManager.start_server() instead.")
        
    async def stop(self):
        """Stop method - deprecated, use MCPServerManager instead."""
        raise DeprecationWarning("MCPServer.stop() is deprecated. Use MCPServerManager.stop_server() instead.")
        
    def get_status(self):
        """Get status method - deprecated, use MCPServerManager instead."""
        raise DeprecationWarning("MCPServer.get_status() is deprecated. Use MCPServerManager.get_server_status() instead.")
        
    def get_endpoints(self):
        """Get endpoints method - deprecated, use MCPServerManager instead."""
        raise DeprecationWarning("MCPServer.get_endpoints() is deprecated. Use MCPServerManager._get_endpoints() instead.")
