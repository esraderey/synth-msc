#!/usr/bin/env python3
"""
Generate API documentation for MSC Framework v4.0
Creates OpenAPI/Swagger specification and Markdown documentation
"""

import json
import yaml
from datetime import datetime
from pathlib import Path

# OpenAPI specification
openapi_spec = {
    "openapi": "3.0.0",
    "info": {
        "title": "MSC Framework API",
        "version": "4.0.0",
        "description": "Meta-cognitive Collective Synthesis Framework with Claude AI",
        "contact": {
            "name": "MSC Framework Team",
            "url": "https://github.com/yourusername/msc-framework"
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    },
    "servers": [
        {
            "url": "http://localhost:5000",
            "description": "Local development server"
        },
        {
            "url": "https://api.mscframework.io",
            "description": "Production server"
        }
    ],
    "tags": [
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "graph", "description": "Knowledge graph operations"},
        {"name": "agents", "description": "Agent management"},
        {"name": "simulation", "description": "Simulation control"},
        {"name": "analytics", "description": "Analytics and metrics"}
    ],
    "components": {
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        },
        "schemas": {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                    "code": {"type": "integer"}
                }
            },
            "Node": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "content": {"type": "string"},
                    "state": {"type": "number", "minimum": 0, "maximum": 1},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "connections": {
                        "type": "object",
                        "properties": {
                            "in": {"type": "integer"},
                            "out": {"type": "integer"}
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "importance": {"type": "number"},
                            "cluster_id": {"type": "integer"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "created_at": {"type": "number"},
                            "created_by": {"type": "string"}
                        }
                    }
                }
            },
            "Agent": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "omega": {"type": "number"},
                    "reputation": {"type": "number"},
                    "performance_score": {"type": "number"},
                    "specialization": {"type": "array", "items": {"type": "string"}}
                }
            },
            "SimulationStatus": {
                "type": "object",
                "properties": {
                    "running": {"type": "boolean"},
                    "paused": {"type": "boolean"},
                    "phase": {"type": "string"},
                    "step_count": {"type": "integer"},
                    "runtime": {"type": "number"},
                    "statistics": {"type": "object"}
                }
            }
        }
    },
    "paths": {
        "/api/auth/login": {
            "post": {
                "tags": ["auth"],
                "summary": "Login with credentials",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "username": {"type": "string"},
                                    "password": {"type": "string"}
                                },
                                "required": ["username", "password"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Successful login",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "access_token": {"type": "string"},
                                        "user": {"type": "object"}
                                    }
                                }
                            }
                        }
                    },
                    "401": {"description": "Invalid credentials"}
                }
            }
        },
        "/api/graph/status": {
            "get": {
                "tags": ["graph"],
                "summary": "Get graph status",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "Graph status",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "nodes": {"type": "integer"},
                                        "edges": {"type": "integer"},
                                        "health": {"type": "object"},
                                        "clusters": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/graph/nodes": {
            "get": {
                "tags": ["graph"],
                "summary": "List nodes with pagination",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                    {"name": "per_page", "in": "query", "schema": {"type": "integer", "default": 50}},
                    {"name": "sort_by", "in": "query", "schema": {"type": "string", "enum": ["id", "state", "importance", "created"]}},
                    {"name": "order", "in": "query", "schema": {"type": "string", "enum": ["asc", "desc"]}},
                    {"name": "min_state", "in": "query", "schema": {"type": "number"}},
                    {"name": "max_state", "in": "query", "schema": {"type": "number"}},
                    {"name": "keywords", "in": "query", "schema": {"type": "string"}},
                    {"name": "tags", "in": "query", "schema": {"type": "string"}},
                    {"name": "cluster_id", "in": "query", "schema": {"type": "integer"}}
                ],
                "responses": {
                    "200": {
                        "description": "List of nodes",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "nodes": {"type": "array", "items": {"$ref": "#/components/schemas/Node"}},
                                        "pagination": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["graph"],
                "summary": "Create a new node",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "initial_state": {"type": "number", "default": 0.5},
                                    "keywords": {"type": "array", "items": {"type": "string"}},
                                    "properties": {"type": "object"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Node created successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "node_id": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/graph/nodes/{node_id}": {
            "get": {
                "tags": ["graph"],
                "summary": "Get node details",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {"name": "node_id", "in": "path", "required": True, "schema": {"type": "integer"}}
                ],
                "responses": {
                    "200": {"description": "Node details"},
                    "404": {"description": "Node not found"}
                }
            }
        },
        "/api/graph/search": {
            "get": {
                "tags": ["graph"],
                "summary": "Search nodes",
                "security": [{"bearerAuth": []}],
                "parameters": [
                    {"name": "q", "in": "query", "required": True, "schema": {"type": "string"}},
                    {"name": "type", "in": "query", "schema": {"type": "string", "enum": ["keyword", "content", "semantic"]}},
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}
                ],
                "responses": {
                    "200": {"description": "Search results"}
                }
            }
        },
        "/api/agents": {
            "get": {
                "tags": ["agents"],
                "summary": "List all agents",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "List of agents",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "agents": {"type": "array", "items": {"$ref": "#/components/schemas/Agent"}},
                                        "total": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/simulation/status": {
            "get": {
                "tags": ["simulation"],
                "summary": "Get simulation status",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "Simulation status",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SimulationStatus"}
                            }
                        }
                    }
                }
            }
        },
        "/api/simulation/control": {
            "post": {
                "tags": ["simulation"],
                "summary": "Control simulation",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": ["start", "stop", "pause", "resume"]}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Action executed successfully"}
                }
            }
        }
    }
}

def generate_markdown_docs():
    """Generate Markdown API documentation"""
    
    md_content = f"""# MSC Framework API Documentation

Version: 4.0.0  
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

The MSC Framework provides a comprehensive REST API for interacting with the meta-cognitive collective synthesis system. All API endpoints return JSON responses and require authentication (except public endpoints).

## Base URL

```
http://localhost:5000
```

## Authentication

Most endpoints require JWT authentication. To authenticate:

1. Login using `/api/auth/login` endpoint
2. Include the JWT token in the Authorization header:
   ```
   Authorization: Bearer YOUR_JWT_TOKEN
   ```

## Rate Limiting

API endpoints are rate-limited to prevent abuse:
- Default: 1000 requests per hour
- Strict endpoints: 100 requests per hour
- Auth endpoints: 5 requests per minute

## Common Response Formats

### Success Response
```json
{{
  "success": true,
  "data": {{...}},
  "message": "Operation successful"
}}
```

### Error Response
```json
{{
  "error": "Error type",
  "message": "Detailed error message",
  "code": 400
}}
```

## Endpoints

"""

    # Generate endpoint documentation
    for path, methods in openapi_spec["paths"].items():
        md_content += f"\n### {path}\n\n"
        
        for method, details in methods.items():
            md_content += f"#### {method.upper()} {path}\n\n"
            md_content += f"**Summary:** {details.get('summary', 'No summary')}\n\n"
            
            # Tags
            if 'tags' in details:
                md_content += f"**Tags:** {', '.join(details['tags'])}\n\n"
            
            # Security
            if 'security' in details:
                md_content += "**Authentication:** Required (JWT Bearer)\n\n"
            else:
                md_content += "**Authentication:** Not required\n\n"
            
            # Parameters
            if 'parameters' in details:
                md_content += "**Parameters:**\n\n"
                md_content += "| Name | In | Type | Required | Description |\n"
                md_content += "|------|-----|------|----------|-------------|\n"
                
                for param in details['parameters']:
                    required = "Yes" if param.get('required', False) else "No"
                    param_type = param['schema'].get('type', 'string')
                    desc = param.get('description', '-')
                    md_content += f"| {param['name']} | {param['in']} | {param_type} | {required} | {desc} |\n"
                
                md_content += "\n"
            
            # Request body
            if 'requestBody' in details:
                md_content += "**Request Body:**\n\n```json\n"
                if 'content' in details['requestBody']:
                    schema = details['requestBody']['content']['application/json']['schema']
                    md_content += json.dumps(schema, indent=2)
                md_content += "\n```\n\n"
            
            # Responses
            if 'responses' in details:
                md_content += "**Responses:**\n\n"
                for code, response in details['responses'].items():
                    md_content += f"- `{code}`: {response.get('description', 'No description')}\n"
                md_content += "\n"
            
            md_content += "---\n"

    # Add WebSocket documentation
    md_content += """
## WebSocket API

The framework provides real-time updates via WebSocket connections.

### Connection

```javascript
const socket = io('http://localhost:5000');
```

### Events

#### Client to Server

- **subscribe**: Subscribe to a room for updates
  ```json
  {"room": "updates"}
  ```

- **unsubscribe**: Unsubscribe from a room
  ```json
  {"room": "updates"}
  ```

- **request_update**: Request specific update
  ```json
  {"type": "status"}
  ```

#### Server to Client

- **connected**: Connection established
  ```json
  {"message": "Connected to MSC Framework", "sid": "socket_id"}
  ```

- **event**: System event
  ```json
  {
    "id": "event_id",
    "type": "NODE_CREATED",
    "data": {...},
    "timestamp": 1234567890
  }
  ```

- **status_update**: Status update
- **graph_update**: Graph update
- **agents_update**: Agents update

### Rooms

- `updates`: General system updates
- `metrics`: Performance metrics
- `agents`: Agent-specific updates

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Missing or invalid token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

## Examples

### Authentication

```bash
# Login
curl -X POST http://localhost:5000/api/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"username": "admin", "password": "password"}'

# Response
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "user": {
    "username": "admin",
    "role": "admin"
  }
}
```

### Create Node

```bash
curl -X POST http://localhost:5000/api/graph/nodes \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "content": "Quantum computing concept",
    "keywords": ["quantum", "computing", "technology"],
    "initial_state": 0.7
  }'
```

### Search Nodes

```bash
# Semantic search
curl -H "Authorization: Bearer YOUR_TOKEN" \\
  "http://localhost:5000/api/graph/search?q=artificial%20intelligence&type=semantic&limit=10"
```

### WebSocket Example

```javascript
// Connect and subscribe
const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('Connected');
  socket.emit('subscribe', {room: 'updates'});
});

// Handle events
socket.on('event', (data) => {
  if (data.type === 'NODE_CREATED') {
    console.log('New node created:', data.data.node_id);
  }
});
```

## SDK Libraries

Official SDK libraries are available for:

- Python: `pip install msc-framework-sdk`
- JavaScript/TypeScript: `npm install @msc-framework/sdk`
- Go: `go get github.com/msc-framework/go-sdk`

## Support

- Documentation: https://docs.mscframework.io
- GitHub: https://github.com/yourusername/msc-framework
- Discord: https://discord.gg/msc-framework
"""

    return md_content

def main():
    """Generate API documentation files"""
    
    # Create docs directory
    docs_dir = Path("docs/api")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save OpenAPI spec as JSON
    with open(docs_dir / "openapi.json", "w") as f:
        json.dump(openapi_spec, f, indent=2)
    print("✓ Generated openapi.json")
    
    # Save OpenAPI spec as YAML
    with open(docs_dir / "openapi.yaml", "w") as f:
        yaml.dump(openapi_spec, f, default_flow_style=False)
    print("✓ Generated openapi.yaml")
    
    # Generate Markdown documentation
    md_docs = generate_markdown_docs()
    with open(docs_dir / "API.md", "w") as f:
        f.write(md_docs)
    print("✓ Generated API.md")
    
    # Generate Postman collection
    postman_collection = {
        "info": {
            "name": "MSC Framework API",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": []
    }
    
    # Convert OpenAPI to Postman format (simplified)
    for tag in openapi_spec["tags"]:
        folder = {
            "name": tag["name"].capitalize(),
            "item": []
        }
        
        for path, methods in openapi_spec["paths"].items():
            for method, details in methods.items():
                if tag["name"] in details.get("tags", []):
                    request = {
                        "name": details.get("summary", path),
                        "request": {
                            "method": method.upper(),
                            "url": {
                                "raw": f"{{{{base_url}}}}{path}",
                                "host": ["{{base_url}}"],
                                "path": path.strip("/").split("/")
                            },
                            "header": []
                        }
                    }
                    
                    # Add auth header if required
                    if "security" in details:
                        request["request"]["header"].append({
                            "key": "Authorization",
                            "value": "Bearer {{access_token}}"
                        })
                    
                    folder["item"].append(request)
        
        if folder["item"]:
            postman_collection["item"].append(folder)
    
    with open(docs_dir / "MSC_Framework.postman_collection.json", "w") as f:
        json.dump(postman_collection, f, indent=2)
    print("✓ Generated Postman collection")
    
    print(f"\nAPI documentation generated in {docs_dir}")
    print("\nTo view OpenAPI documentation:")
    print("1. Install swagger-ui: npm install -g @apidevtools/swagger-cli")
    print("2. Run: swagger-cli serve docs/api/openapi.yaml")

if __name__ == "__main__":
    main()