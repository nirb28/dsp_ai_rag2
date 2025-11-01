# Security Configuration Documentation

## Overview

The DSP AI RAG2 system supports optional security authentication to control access to query and retrieve endpoints. The security system is designed to be flexible, configurable, and backward-compatible.

## Table of Contents

1. [Security Types](#security-types)
2. [JWT Bearer Token Authentication](#jwt-bearer-token-authentication)
3. [Configuration](#configuration)
4. [Metadata Filtering](#metadata-filtering)
5. [API Usage](#api-usage)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

## Security Types

Currently supported authentication types:

- **JWT Bearer Token** (`jwt_bearer`) - JSON Web Token based authentication
- **API Key** (`api_key`) - Simple API key authentication (future implementation)
- **OAuth2** (`oauth2`) - OAuth2 flow authentication (future implementation)

## JWT Bearer Token Authentication

### Features

- **Configurable JWT validation** with support for:
  - Custom secret keys
  - Multiple algorithms (HS256, HS512, RS256, etc.)
  - Issuer (`iss`) validation
  - Audience (`aud`) validation
  - Expiration (`exp`) and issued-at (`iat`) validation
  - Configurable leeway for clock skew
- **Metadata filtering** via JWT claims
- **Secure token validation** with proper error handling

### JWT Claims

#### Standard Claims

| Claim | Description | Required |
|-------|-------------|----------|
| `sub` | Subject (user identifier) | No |
| `iat` | Issued at timestamp | Configurable |
| `exp` | Expiration timestamp | Configurable |
| `iss` | Issuer | Configurable |
| `aud` | Audience | Configurable |

#### Custom Claims

| Claim | Description | Type | Example |
|-------|-------------|------|---------|
| `metadata_filter` | Document filtering criteria | Object | `{"department": "engineering", "level": "public"}` |

## Configuration

### Security Configuration Schema

```json
{
  "security": {
    "enabled": false,
    "type": "jwt_bearer",
    "jwt_secret_key": "your-secret-key",
    "jwt_algorithm": "HS256",
    "jwt_issuer": "your-issuer",
    "jwt_audience": "your-audience",
    "jwt_require_exp": true,
    "jwt_require_iat": true,
    "jwt_leeway": 0,
    "api_key_header": "X-API-Key",
    "api_keys": ["key1", "key2"]
  }
}
```

### Configuration Parameters

#### General Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `false` | Whether to enable security authentication |
| `type` | string | `"jwt_bearer"` | Type of authentication (`jwt_bearer`, `api_key`, `oauth2`) |

#### JWT Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jwt_secret_key` | string | `null` | Secret key for JWT validation (required when JWT enabled) |
| `jwt_algorithm` | string | `"HS256"` | Algorithm for JWT validation |
| `jwt_issuer` | string | `null` | Expected issuer of JWT tokens |
| `jwt_audience` | string | `null` | Expected audience of JWT tokens |
| `jwt_require_exp` | boolean | `true` | Whether to require expiration claim |
| `jwt_require_iat` | boolean | `true` | Whether to require issued-at claim |
| `jwt_leeway` | integer | `0` | Leeway in seconds for token expiration |

#### API Key Settings (Future)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key_header` | string | `"X-API-Key"` | Header name for API key |
| `api_keys` | array | `null` | List of valid API keys |

## Metadata Filtering

### Overview

JWT tokens can include a `metadata_filter` claim that automatically applies document filtering to query and retrieve operations. This enables fine-grained access control based on document metadata.

### Filter Syntax

The metadata filter uses MongoDB-style query operators:

#### Basic Equality
```json
{
  "metadata_filter": {
    "department": "engineering",
    "level": "public"
  }
}
```

#### Comparison Operators
```json
{
  "metadata_filter": {
    "score": {"$gte": 0.8},
    "created_date": {"$lt": "2024-01-01"}
  }
}
```

#### Array Operators
```json
{
  "metadata_filter": {
    "tags": {"$in": ["ai", "ml", "nlp"]},
    "categories": {"$nin": ["restricted", "confidential"]}
  }
}
```

#### Logical Operators
```json
{
  "metadata_filter": {
    "$and": [
      {"department": "engineering"},
      {"$or": [
        {"level": "public"},
        {"level": "internal"}
      ]}
    ]
  }
}
```

### Filter Merging

When both request filters and JWT metadata filters are present, they are combined using the `$and` operator:

```json
{
  "$and": [
    {"source": "user_request_filter"},
    {"department": "jwt_metadata_filter"}
  ]
}
```

## API Usage

### Authentication Header

Include the JWT token in the Authorization header:

```
Authorization: Bearer <jwt_token>
```

### Query Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "configuration_name": "secure_config",
    "k": 5
  }'
```

### Retrieve Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep learning",
    "configuration_name": "secure_config",
    "k": 10,
    "include_metadata": true
  }'
```

## Examples

### Example 1: Basic JWT Configuration

```json
{
  "configuration_name": "secure_basic",
  "config": {
    "chunking": {
      "strategy": "recursive_text",
      "chunk_size": 1000,
      "chunk_overlap": 200
    },
    "vector_store": {
      "type": "faiss",
      "index_path": "./storage/secure_basic_index"
    },
    "embedding": {
      "model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "generation": {
      "model": "llama3-8b-8192",
      "provider": "groq"
    },
    "security": {
      "enabled": true,
      "type": "jwt_bearer",
      "jwt_secret_key": "my-secret-key-123",
      "jwt_algorithm": "HS256"
    }
  }
}
```

### Example 2: JWT with Issuer/Audience Validation

```json
{
  "security": {
    "enabled": true,
    "type": "jwt_bearer",
    "jwt_secret_key": "my-secret-key-123",
    "jwt_algorithm": "HS256",
    "jwt_issuer": "my-auth-service",
    "jwt_audience": "rag-api",
    "jwt_require_exp": true,
    "jwt_require_iat": true,
    "jwt_leeway": 10
  }
}
```

### Example 3: Creating JWT Tokens

#### Python Example

```python
import jwt
from datetime import datetime, timedelta, timezone

def create_jwt_token(secret_key, user_id, metadata_filter=None):
    payload = {
        "sub": user_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iss": "my-auth-service",
        "aud": "rag-api"
    }
    
    if metadata_filter:
        payload["metadata_filter"] = metadata_filter
    
    return jwt.encode(payload, secret_key, algorithm="HS256")

# Create token with metadata filter
token = create_jwt_token(
    secret_key="my-secret-key-123",
    user_id="user123",
    metadata_filter={
        "department": "engineering",
        "level": {"$in": ["public", "internal"]}
    }
)
```

#### Node.js Example

```javascript
const jwt = require('jsonwebtoken');

function createJwtToken(secretKey, userId, metadataFilter = null) {
    const payload = {
        sub: userId,
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + (60 * 60), // 1 hour
        iss: 'my-auth-service',
        aud: 'rag-api'
    };
    
    if (metadataFilter) {
        payload.metadata_filter = metadataFilter;
    }
    
    return jwt.sign(payload, secretKey, { algorithm: 'HS256' });
}

// Create token with metadata filter
const token = createJwtToken(
    'my-secret-key-123',
    'user123',
    {
        department: 'engineering',
        level: { $in: ['public', 'internal'] }
    }
);
```

### Example 4: Multi-Configuration Security

For multi-configuration requests (retrieve endpoint), security validation uses the first configuration's security settings:

```json
{
  "query": "machine learning algorithms",
  "configuration_names": ["secure_config1", "secure_config2"],
  "k": 10
}
```

The security settings from `secure_config1` will be used for authentication.

## Testing

### Running Security Tests

Use the provided test script to validate security functionality:

```bash
cd /path/to/dsp_ai_rag2
python test_security_feature.py
```

### Test Coverage

The test script validates:

- ✅ Configuration creation with security enabled
- ✅ Document upload with authentication
- ✅ Query without authentication (should fail)
- ✅ Query with valid JWT token
- ✅ Retrieve with JWT metadata filtering
- ✅ Invalid JWT token handling
- ✅ Expired JWT token handling

### Manual Testing

#### 1. Create Secure Configuration

```bash
curl -X POST "http://localhost:8000/api/v1/configurations" \
  -H "Content-Type: application/json" \
  -d '{
    "configuration_name": "test_secure",
    "config": {
      "security": {
        "enabled": true,
        "type": "jwt_bearer",
        "jwt_secret_key": "test-secret-123"
      }
    }
  }'
```

#### 2. Test Without Authentication (Should Fail)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "configuration_name": "test_secure"
  }'
```

Expected response: `401 Unauthorized`

#### 3. Test With Valid Token

```bash
# First create a valid JWT token using your preferred method
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."

curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "configuration_name": "test_secure"
  }'
```

Expected response: `200 OK` with query results

## Troubleshooting

### Common Issues

#### 1. "Authorization header is required"

**Cause**: No Authorization header provided when security is enabled.

**Solution**: Include the Authorization header with Bearer token:
```
Authorization: Bearer <your_jwt_token>
```

#### 2. "Invalid token: Signature verification failed"

**Cause**: JWT token signed with different secret key.

**Solution**: Ensure the token is signed with the same secret key configured in the security settings.

#### 3. "Token has expired"

**Cause**: JWT token has passed its expiration time.

**Solution**: Generate a new token with a valid expiration time.

#### 4. "Invalid authentication scheme. Expected 'Bearer'"

**Cause**: Authorization header doesn't use Bearer scheme.

**Solution**: Use the correct format:
```
Authorization: Bearer <token>
```

#### 5. "jwt_secret_key is required when JWT Bearer authentication is enabled"

**Cause**: Security is enabled but no JWT secret key is configured.

**Solution**: Add `jwt_secret_key` to your security configuration.

### Debug Tips

1. **Check Configuration**: Verify security configuration is properly set:
   ```bash
   curl "http://localhost:8000/api/v1/configurations/your_config_name"
   ```

2. **Validate JWT Token**: Use online JWT decoders to verify token structure and claims.

3. **Check Server Logs**: Review application logs for detailed error messages.

4. **Test with Simple Token**: Create a minimal JWT token for testing:
   ```python
   import jwt
   from datetime import datetime, timedelta, timezone
   
   token = jwt.encode({
       "sub": "test_user",
       "exp": datetime.now(timezone.utc) + timedelta(hours=1)
   }, "your-secret-key", algorithm="HS256")
   ```

### Error Codes

| HTTP Code | Description | Common Causes |
|-----------|-------------|---------------|
| 401 | Unauthorized | Missing/invalid token, expired token |
| 400 | Bad Request | Invalid configuration, malformed request |
| 500 | Internal Server Error | Server configuration issues |

## Security Best Practices

1. **Use Strong Secret Keys**: Generate cryptographically secure secret keys
2. **Set Appropriate Expiration**: Use reasonable token expiration times
3. **Validate Issuer/Audience**: Configure issuer and audience validation for production
4. **Use HTTPS**: Always use HTTPS in production to protect tokens in transit
5. **Rotate Keys**: Implement key rotation for long-term security
6. **Monitor Access**: Log and monitor authentication attempts
7. **Principle of Least Privilege**: Use metadata filters to limit document access

## Migration Guide

### Enabling Security on Existing Configurations

1. **Update Configuration**: Add security block to existing configuration
2. **Test Endpoints**: Verify endpoints work with authentication
3. **Update Clients**: Modify client applications to include JWT tokens
4. **Monitor**: Watch for authentication errors during rollout

### Disabling Security

To disable security on a configuration:

```json
{
  "security": {
    "enabled": false
  }
}
```

This maintains backward compatibility with existing clients.
