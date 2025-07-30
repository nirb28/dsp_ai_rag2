# Security Quick Reference Guide

## Quick Setup

### 1. Enable Security in Configuration

```json
{
  "configuration_name": "secure_config",
  "config": {
    "security": {
      "enabled": true,
      "type": "jwt_bearer",
      "jwt_secret_key": "your-secret-key-here",
      "jwt_algorithm": "HS256"
    }
  }
}
```

### 2. Create JWT Token

```python
import jwt
from datetime import datetime, timedelta, timezone

def create_token(secret_key, user_id, metadata_filter=None):
    payload = {
        "sub": user_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
    }
    if metadata_filter:
        payload["metadata_filter"] = metadata_filter
    return jwt.encode(payload, secret_key, algorithm="HS256")

# Example usage
token = create_token(
    "your-secret-key-here", 
    "user123",
    {"department": "engineering", "level": "public"}
)
```

### 3. Use Token in API Calls

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "configuration_name": "secure_config"}'
```

## Common Metadata Filters

```json
// Simple equality
{"department": "engineering"}

// Multiple conditions (AND)
{"department": "engineering", "level": "public"}

// Comparison operators
{"score": {"$gte": 0.8}, "priority": {"$lt": 5}}

// Array membership
{"tags": {"$in": ["ai", "ml", "nlp"]}}

// Complex logical operations
{
  "$and": [
    {"department": "engineering"},
    {"$or": [{"level": "public"}, {"level": "internal"}]}
  ]
}
```

## Error Responses

| Code | Meaning | Solution |
|------|---------|----------|
| 401 | Missing/invalid token | Include valid JWT token |
| 401 | Token expired | Generate new token |
| 401 | Invalid signature | Check secret key matches |
| 400 | Bad configuration | Verify security config |

## Testing Commands

```bash
# Test without auth (should fail)
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "configuration_name": "secure_config"}'

# Test with valid token
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "configuration_name": "secure_config"}'

# Run comprehensive tests
python test_security_feature.py
```

## Configuration Options Reference

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `enabled` | No | `false` | Enable/disable security |
| `type` | No | `"jwt_bearer"` | Authentication type |
| `jwt_secret_key` | Yes* | `null` | JWT signing key |
| `jwt_algorithm` | No | `"HS256"` | JWT algorithm |
| `jwt_issuer` | No | `null` | Expected issuer |
| `jwt_audience` | No | `null` | Expected audience |
| `jwt_require_exp` | No | `true` | Require expiration |
| `jwt_require_iat` | No | `true` | Require issued-at |
| `jwt_leeway` | No | `0` | Clock skew tolerance |

*Required when JWT authentication is enabled

## Security Best Practices

✅ **DO:**
- Use strong, random secret keys (32+ characters)
- Set reasonable token expiration times (1-24 hours)
- Use HTTPS in production
- Configure issuer/audience validation
- Use metadata filters for access control
- Monitor authentication logs

❌ **DON'T:**
- Hardcode secret keys in source code
- Use weak or predictable secret keys
- Set very long token expiration times
- Ignore token validation errors
- Use HTTP in production
- Share tokens between users
