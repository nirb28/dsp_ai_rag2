# MCP Server Integration Documentation

## Overview

The DSP AI RAG2 project includes comprehensive Model Context Protocol (MCP) server integration that allows LLM clients (Claude, Cursor, etc.) to directly access RAG functionality through standard MCP protocols. This implementation enables each RAG configuration to optionally expose an MCP server with configurable tools and protocols.

### Single-Port Design Benefits

- **Simplified Port Management**: Only one port (8080) to configure and manage
- **Better Resource Utilization**: Single FastAPI server process handles all configurations
- **Easier Client Integration**: Consistent base URL across all configurations
- **Production-Ready**: Simplified deployment, reverse proxy setup, and load balancing
- **Cleaner Architecture**: Centralized routing and request handling

## Configuration

### Basic MCP Server Configuration

Add the `mcp_server` section to your RAG configuration:

```json
{
  "chunking": { ... },
  "vector_store": { ... },
  "embedding": { ... },
  "generation": { ... },
  "mcp_server": {
    "enabled": true,
    "name": "My RAG MCP Server",
    "description": "MCP server for document retrieval and querying",
    "version": "1.0.0",
    "protocols": ["http", "sse"],
    "tools": [
      {
        "name": "retrieve_documents",
        "type": "retrieve",
        "enabled": true,
        "description": "Retrieve relevant documents based on query",
        "max_results": 10,
        "include_metadata": true
      }
    ]
  }
}
```

### Connection Endpoints
### MCP Server (localhost:8080)
- `GET /info` - Server information
- `GET /mcp-servers` - List all MCP servers
- `GET /mcp-servers/{config}` - Get specific server status
- `POST /mcp-servers/start` - Start MCP server
- `POST /mcp-servers/stop` - Stop MCP server
- `POST /mcp-servers/{config}/tools/execute` - Execute MCP tool
- `POST /mcp-servers/shutdown-all` - Shutdown all servers

## Configuration

MCP servers are configured in the RAG configuration with the `mcp_server` section:

```json
    "mcp_server": {
      "startup_enabled": true,
      "enabled": true,
      "name": "mcp_batch_ml_ai_basics_test",
      "protocols": ["http", "sse"],
      "tools": [
        {
          "type": "retrieve",
          "name": "query_machine_learning_docs",
          "description": "Query machine learning documents. Be precise and include key terms for better results.",
          "max_results": 1,
          "similarity_threshold": 0.1
        }
      ]
    }    

```


### Connection Endpoints

Each configuration in the MCP server hub exposes endpoints at:
- **HTTP JSON-RPC**: `http://localhost:8080/{config_name}/mcp`
- **Server-Sent Events**: `http://localhost:8080/{config_name}/sse`
- **WebSocket**: `ws://localhost:8080/{config_name}/ws`
- **Configuration Info**: `http://localhost:8080/{config_name}/info`

### Direct Hub Access

The MCP server hub also provides direct access endpoints:
- **Hub Root**: `http://localhost:8080/` - Shows active configurations
- **Hub Health**: `http://localhost:8080/health` - Health status

### Claude Desktop Integration

Add to your Claude configuration:

```json
{
  "mcpServers": {
    "rag-default": {
      "command": "curl",
      "args": [
        "-X", "POST",
        "http://localhost:8080/default/mcp",
        "-H", "Content-Type: application/json",
        "--data-raw"
      ],
      "transport": "http"
    },
    "rag-research": {
      "command": "curl", 
      "args": [
        "-X", "POST",
        "http://localhost:8080/research/mcp",
        "-H", "Content-Type: application/json",
        "--data-raw"
      ],
      "transport": "http"
    }
  }
}
```

### Cursor Integration

Configure multiple MCP servers in Cursor settings:

```json
{
  "mcp.servers": {
    "rag-default": {
      "url": "http://localhost:8080/default/mcp",
      "protocol": "http"
    },
    "rag-research": {
      "url": "http://localhost:8080/research/mcp", 
      "protocol": "http"
    }
  }
}
```

## Advanced Features

### Single Hub, Multiple Configurations

The new architecture allows running multiple RAG configurations through a single MCP server hub:

```bash
# Start multiple configurations
curl -X POST http://localhost:8080/mcp-servers/start -d '{"configuration_name": "research"}'
curl -X POST http://localhost:8080/mcp-servers/start -d '{"configuration_name": "support"}'
curl -X POST http://localhost:8080/mcp-servers/start -d '{"configuration_name": "internal"}'

# All available at different paths on same port:
# http://localhost:8080/research/mcp
# http://localhost:8080/support/mcp  
# http://localhost:8080/internal/mcp
```

### Server-Sent Events (SSE)

Connect to configuration-specific SSE endpoints:

```javascript
// Connect to specific configuration's SSE stream
const eventSource = new EventSource('http://localhost:8080/default/sse');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('MCP Server Event:', data);
  // Event includes configuration name for identification
};
```

### WebSocket Integration

```javascript
// Connect to specific configuration's WebSocket
const ws = new WebSocket('ws://localhost:8080/default/ws');

ws.onopen = function() {
  // Send MCP initialize request
  ws.send(JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: { protocolVersion: "2024-11-05" }
  }));
};

ws.onmessage = function(event) {
  const response = JSON.parse(event.data);
  console.log('MCP Response:', response);
};
```

## Performance Considerations

### Optimization Tips

1. **Single Server Process**: Reduced resource overhead compared to multiple server instances
2. **Shared Connection Pool**: All configurations share the same HTTP server and connection pool
3. **Efficient Routing**: FastAPI's path-based routing is highly optimized
4. **Tool Result Limits**: Configure appropriate `max_results` for tools
5. **Resource Management**: Monitor memory usage with large document sets

### Scaling

- **Horizontal Scaling**: Deploy multiple MCP server hubs behind a load balancer
- **Configuration Distribution**: Different hubs can serve different sets of configurations
- **Resource Isolation**: Each hub runs in its own process for isolation
- **Client Load Balancing**: Clients can connect to different hub instances

### Port Management

- **Single Port**: Only port 8080 needs to be opened/configured
- **Firewall Simplicity**: Only one port to manage in firewall rules
- **Reverse Proxy**: Single upstream target for nginx/Apache configurations
- **Service Discovery**: Single endpoint for service discovery systems

## Troubleshooting

### Common Issues

1. **Hub Won't Start**
   - Check if port 8080 is already in use
   - Verify no other MCP hub is running
   - Check logs for specific error messages

2. **Configuration Not Accessible**
   - Ensure configuration was successfully added to hub (`GET /mcp-servers`)
   - Check if configuration has MCP server enabled
   - Verify correct path format: `/{configuration_name}/mcp`

3. **Tools Not Working**
   - Check if hub is running (`GET /mcp-hub/status`)
   - Verify configuration is active in hub
   - Ensure documents are loaded in the configuration

### Debug Mode

Check hub status and active configurations:

```bash
# Check hub status
curl http://localhost:8080/mcp-hub/status

# List all active configurations
curl http://localhost:8080/mcp-servers

# Check specific configuration
curl http://localhost:8080/default/info
```

### Health Checks

```bash
# Hub health check
curl http://localhost:8080/health

# Configuration-specific health
curl http://localhost:8080/{config_name}/info

# REST API server status  
curl http://localhost:8080/mcp-hub/status
```

## Migration from Multi-Port Architecture

If you're migrating from the previous multi-port architecture:

### URL Changes
- **Before**: `http://localhost:8081/config1/mcp`, `http://localhost:8082/config2/mcp`
- **After**: `http://localhost:8080/config1/mcp`, `http://localhost:8080/config2/mcp`

### Configuration Changes
- Remove `http_host` and `http_port` from MCP server configurations
- Update client configurations to use single port with path-based routing
- Update firewall rules to only allow port 8080

### Benefits of Migration
- Simplified port management and configuration
- Better resource utilization and performance  
- Easier deployment and reverse proxy setup
- Consistent client connection patterns
