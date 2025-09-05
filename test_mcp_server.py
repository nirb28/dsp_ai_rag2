#!/usr/bin/env python3
"""
Comprehensive test script for MCP server functionality.

This script tests the MCP server integration including:
- Configuration creation with MCP server settings
- Server lifecycle management (start/stop)
- Tool execution via REST and JSON-RPC
- Error handling and edge cases
- Server monitoring and status endpoints

Usage:
    python test_mcp_server.py [--config CONFIG_NAME] [--create-new]
"""

import asyncio
import json
import logging
import time
import requests
import websockets
from typing import Dict, Any, Optional
import argparse
from pathlib import Path

from app.services.rag_service import RAGService
from app.services.mcp.server import MCPServerManager
from app.config import (
    RAGConfig, VectorStoreConfig, EmbeddingConfig, GenerationConfig,
    MCPServerConfig, MCPToolConfig, MCPProtocol, MCPToolType, VectorStore
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCPServerTest:
    """Comprehensive MCP server test suite."""
    
    def __init__(self, config_name: str = "test_mcp_config", base_url: str = "http://localhost:8080"):
        self.config_name = config_name
        self.base_url = base_url
        self.rag_service = RAGService()
        
    def create_test_configuration(self) -> bool:
        """Create a test RAG configuration with MCP server settings."""
        try:
            logger.info(f"Creating test configuration: {self.config_name}")
            
            # Define MCP tools
            mcp_tools = [
                MCPToolConfig(
                    name="retrieve_documents",
                    type=MCPToolType.RETRIEVE,
                    enabled=True,
                    description="Retrieve relevant documents based on query",
                    max_results=10,
                    include_metadata=True
                )
            ]
            
            # Define MCP server configuration
            mcp_server_config = MCPServerConfig(
                enabled=True,
                name=f"MCP Server - {self.config_name}",
                description="Test MCP server for validation",
                version="1.0.0",
                protocols=[MCPProtocol.HTTP, MCPProtocol.SSE],
                http_host="localhost",
                http_port=8080,
                tools=mcp_tools
            )
            
            # Create RAG configuration
            rag_config = RAGConfig(
                vector_store=VectorStoreConfig(
                    type=VectorStore.FAISS,
                    faiss_index_path=f"./storage/{self.config_name}_index.faiss",
                    faiss_store_path=f"./storage/{self.config_name}_store.pkl"
                ),
                embedding=EmbeddingConfig(),
                generation=GenerationConfig(),
                mcp_server=mcp_server_config
            )
            
            # Add configuration to service
            success = self.rag_service.add_configuration(self.config_name, rag_config.dict())
            
            if success:
                logger.info(f"✅ Test configuration '{self.config_name}' created successfully")
                return True
            else:
                logger.error(f"❌ Failed to create test configuration '{self.config_name}'")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating test configuration: {str(e)}")
            return False
    
    def add_test_documents(self) -> bool:
        """Add some test documents to the configuration."""
        try:
            logger.info("Adding test documents...")
            
            test_documents = [
                {
                    "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                    "filename": "ml_intro.txt",
                    "metadata": {"topic": "machine_learning", "type": "educational"}
                },
                {
                    "content": "Python is a high-level programming language known for its simplicity and versatility.",
                    "filename": "python_intro.txt", 
                    "metadata": {"topic": "programming", "type": "educational"}
                },
                {
                    "content": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
                    "filename": "rag_overview.txt",
                    "metadata": {"topic": "rag", "type": "technical"}
                }
            ]
            
            for doc in test_documents:
                result = asyncio.run(self.rag_service.upload_text_content(
                    content=doc["content"],
                    filename=doc["filename"],
                    configuration_name=self.config_name,
                    metadata=doc["metadata"],
                    process_immediately=True
                ))
                logger.info(f"Added document: {result.filename}")
            
            logger.info("✅ Test documents added successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding test documents: {str(e)}")
            return False
    
    async def test_server_lifecycle(self) -> bool:
        """Test MCP server start/stop lifecycle."""
        try:
            logger.info("Testing MCP server lifecycle...")
            
            # Test server start
            start_response = requests.post(
                f"{self.base_url}/mcp-servers/start",
                json={"configuration_name": self.config_name, "force_restart": False}
            )
            
            if start_response.status_code != 200:
                logger.error(f"❌ Failed to start MCP server: {start_response.text}")
                return False
            
            start_data = start_response.json()
            logger.info(f"✅ MCP server started: {start_data['message']}")
            
            # Wait a bit for server to fully start
            await asyncio.sleep(2)
            
            # Test server status
            status_response = requests.get(f"{self.base_url}/mcp-servers/{self.config_name}")
            
            if status_response.status_code != 200:
                logger.error(f"❌ Failed to get server status: {status_response.text}")
                return False
            
            status_data = status_response.json()
            logger.info(f"✅ Server status retrieved: Running={status_data['running']}")
            
            # Test server list
            list_response = requests.get(f"{self.base_url}/mcp-servers")
            
            if list_response.status_code != 200:
                logger.error(f"❌ Failed to list MCP servers: {list_response.text}")
                return False
            
            list_data = list_response.json()
            logger.info(f"✅ Server list retrieved: {list_data['total_count']} servers")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in server lifecycle test: {str(e)}")
            return False
    
    async def test_tool_execution(self) -> bool:
        """Test MCP tool execution via REST API."""
        try:
            logger.info("Testing MCP tool execution...")
            
            # Test retrieve tool
            retrieve_request = {
                "tool_name": "retrieve_documents",
                "parameters": {
                    "query": "machine learning",
                    "k": 3,
                    "similarity_threshold": 0.5
                }
            }
            
            retrieve_response = requests.post(
                f"{self.base_url}/mcp-servers/{self.config_name}/tools/execute",
                json=retrieve_request
            )
            
            if retrieve_response.status_code != 200:
                logger.error(f"❌ Failed to execute retrieve tool: {retrieve_response.text}")
                return False
            
            retrieve_data = retrieve_response.json()
            logger.info(f"✅ Retrieve tool executed: {retrieve_data['success']}")
            
            # Test list documents tool
            list_request = {
                "tool_name": "list_documents",
                "parameters": {"limit": 10}
            }
            
            list_response = requests.post(
                f"{self.base_url}/mcp-servers/{self.config_name}/tools/execute",
                json=list_request
            )
            
            if list_response.status_code != 200:
                logger.error(f"❌ Failed to execute list tool: {list_response.text}")
                return False
            
            list_data = list_response.json()
            logger.info(f"✅ List tool executed: {list_data['success']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in tool execution test: {str(e)}")
            return False
    
    async def test_mcp_protocol(self) -> bool:
        """Test direct MCP protocol communication."""
        try:
            logger.info("Testing MCP protocol communication...")
            
            # Get server endpoints
            status_response = requests.get(f"{self.base_url}/mcp-servers/{self.config_name}")
            status_data = status_response.json()
            endpoints = status_data.get("endpoints", {})
            
            if "http" not in endpoints:
                logger.error("❌ HTTP endpoint not available")
                return False
            
            mcp_endpoint = endpoints["http"]
            logger.info(f"Using MCP endpoint: {mcp_endpoint}")
            
            # Test initialize method
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {}
                }
            }
            
            init_response = requests.post(mcp_endpoint, json=init_request)
            
            if init_response.status_code != 200:
                logger.error(f"❌ MCP initialize failed: {init_response.text}")
                return False
            
            init_data = init_response.json()
            logger.info(f"✅ MCP initialize successful: {init_data.get('result', {}).get('serverInfo', {}).get('name')}")
            
            # Test tools/list method
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            
            tools_response = requests.post(mcp_endpoint, json=tools_request)
            
            if tools_response.status_code != 200:
                logger.error(f"❌ MCP tools/list failed: {tools_response.text}")
                return False
            
            tools_data = tools_response.json()
            tools_list = tools_data.get("result", {}).get("tools", [])
            logger.info(f"✅ MCP tools/list successful: {len(tools_list)} tools available")
            
            # Test tools/call method
            if tools_list:
                call_request = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "retrieve_documents",
                        "arguments": {
                            "query": "python programming",
                            "k": 2
                        }
                    }
                }
                
                call_response = requests.post(mcp_endpoint, json=call_request)
                
                if call_response.status_code != 200:
                    logger.error(f"❌ MCP tools/call failed: {call_response.text}")
                    return False
                
                call_data = call_response.json()
                logger.info(f"✅ MCP tools/call successful: {len(call_data.get('result', {}).get('content', []))} content items")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in MCP protocol test: {str(e)}")
            return False
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket MCP connection."""
        try:
            logger.info("Testing WebSocket MCP connection...")
            
            # Get WebSocket endpoint
            status_response = requests.get(f"{self.base_url}/mcp-servers/{self.config_name}")
            status_data = status_response.json()
            endpoints = status_data.get("endpoints", {})
            
            if "websocket" not in endpoints:
                logger.error("❌ WebSocket endpoint not available")
                return False
            
            ws_endpoint = endpoints["websocket"]
            logger.info(f"Using WebSocket endpoint: {ws_endpoint}")
            
            # Test WebSocket connection and communication
            async with websockets.connect(ws_endpoint) as websocket:
                # Send initialize request
                init_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {}
                    }
                }
                
                await websocket.send(json.dumps(init_request))
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if "result" not in response_data:
                    logger.error(f"❌ WebSocket initialize failed: {response_data}")
                    return False
                
                logger.info("✅ WebSocket initialize successful")
                
                # Send tools/list request
                tools_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {}
                }
                
                await websocket.send(json.dumps(tools_request))
                response = await websocket.recv()
                response_data = json.loads(response)
                
                tools = response_data.get("result", {}).get("tools", [])
                logger.info(f"✅ WebSocket tools/list successful: {len(tools)} tools")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in WebSocket test: {str(e)}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling scenarios."""
        try:
            logger.info("Testing error handling...")
            
            # Test invalid tool execution
            invalid_tool_request = {
                "tool_name": "nonexistent_tool",
                "parameters": {"query": "test"}
            }
            
            response = requests.post(
                f"{self.base_url}/mcp-servers/{self.config_name}/tools/execute",
                json=invalid_tool_request
            )
            
            if response.status_code == 200:
                data = response.json()
                if not data.get("success", True):
                    logger.info("✅ Invalid tool properly rejected")
                else:
                    logger.warning("⚠️ Invalid tool was not rejected")
            
            # Test invalid configuration
            invalid_config_response = requests.get(f"{self.base_url}/mcp-servers/nonexistent_config")
            
            if invalid_config_response.status_code == 200:
                data = invalid_config_response.json()
                if not data.get("running", True):
                    logger.info("✅ Invalid configuration properly handled")
                else:
                    logger.warning("⚠️ Invalid configuration not handled properly")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in error handling test: {str(e)}")
            return False
    
    async def test_server_stop(self) -> bool:
        """Test MCP server stop."""
        try:
            logger.info("Testing MCP server stop...")
            
            # Stop server
            stop_response = requests.post(
                f"{self.base_url}/mcp-servers/stop",
                json={"configuration_name": self.config_name}
            )
            
            if stop_response.status_code != 200:
                logger.error(f"❌ Failed to stop MCP server: {stop_response.text}")
                return False
            
            stop_data = stop_response.json()
            logger.info(f"✅ MCP server stopped: {stop_data['message']}")
            
            # Verify server is stopped
            status_response = requests.get(f"{self.base_url}/mcp-servers/{self.config_name}")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                if not status_data.get("running", True):
                    logger.info("✅ Server status confirms it's stopped")
                else:
                    logger.warning("⚠️ Server still appears to be running")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in server stop test: {str(e)}")
            return False
    
    def cleanup_test_configuration(self) -> bool:
        """Clean up test configuration."""
        try:
            logger.info(f"Cleaning up test configuration: {self.config_name}")
            
            success = self.rag_service.delete_configuration(self.config_name)
            
            if success:
                logger.info(f"✅ Test configuration '{self.config_name}' cleaned up successfully")
                return True
            else:
                logger.warning(f"⚠️ Failed to clean up test configuration '{self.config_name}'")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up test configuration: {str(e)}")
            return False
    
    async def run_full_test_suite(self, create_config: bool = False) -> Dict[str, bool]:
        """Run the complete MCP server test suite."""
        logger.info("🚀 Starting comprehensive MCP server test suite")
        
        results = {}
        
        try:
            # Create test configuration if requested
            if create_config:
                results["create_config"] = self.create_test_configuration()
                if results["create_config"]:
                    results["add_documents"] = self.add_test_documents()
                else:
                    logger.error("❌ Cannot proceed without test configuration")
                    return results
            
            # Test server lifecycle
            results["server_lifecycle"] = await self.test_server_lifecycle()
            
            # Test tool execution
            if results["server_lifecycle"]:
                results["tool_execution"] = await self.test_tool_execution()
                results["mcp_protocol"] = await self.test_mcp_protocol()
                results["websocket_connection"] = await self.test_websocket_connection()
                results["error_handling"] = await self.test_error_handling()
                results["server_stop"] = await self.test_server_stop()
            
            # Clean up if we created the config
            if create_config:
                results["cleanup"] = self.cleanup_test_configuration()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed with error: {str(e)}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("📊 TEST SUITE SUMMARY")
        logger.info("="*50)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name:20} | {status}")
        
        logger.info("="*50)
        logger.info(f"📈 Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! MCP server is working correctly.")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} test(s) failed. Please check the logs.")
        
        return results


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test MCP server functionality")
    parser.add_argument("--config", default="batch_ml_ai_basics_test", help="Configuration name to test")
    # parser.add_argument("--create-new", action="store_true", help="Create new test configuration")
    parser.add_argument("--base-url", default="http://localhost:9000", help="Base URL for API server")
    
    args = parser.parse_args()
    
    # Create test instance
    test = MCPServerTest(config_name=args.config, base_url=args.base_url)
    
    # Run test suite
    results = await test.run_full_test_suite(create_config=args.create_new)
    
    # Exit with appropriate code
    if all(results.values()):
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
