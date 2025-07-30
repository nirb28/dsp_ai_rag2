"""
Security service for handling authentication and authorization.
"""

import logging
import jwt
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from fastapi import HTTPException, status
from app.config import SecurityConfig, SecurityType

logger = logging.getLogger(__name__)


class SecurityService:
    """Service for handling security authentication and authorization."""
    
    def __init__(self, security_config: SecurityConfig):
        """Initialize the security service with configuration.
        
        Args:
            security_config: Security configuration settings
        """
        self.config = security_config
        
    def validate_request(self, authorization_header: Optional[str]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate an incoming request based on security configuration.
        
        Args:
            authorization_header: The Authorization header value from the request
            
        Returns:
            Tuple of (is_valid, claims_dict)
            - is_valid: Whether the request is authorized
            - claims_dict: JWT claims if JWT authentication, None otherwise
            
        Raises:
            HTTPException: If authentication fails
        """
        if not self.config.enabled:
            # Security is disabled, allow all requests
            return True, None
            
        if not authorization_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header is required",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        if self.config.type == SecurityType.JWT_BEARER:
            return self._validate_jwt_bearer(authorization_header)
        elif self.config.type == SecurityType.API_KEY:
            return self._validate_api_key(authorization_header)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unsupported security type: {self.config.type}"
            )
    
    def _validate_jwt_bearer(self, authorization_header: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate JWT Bearer token.
        
        Args:
            authorization_header: The Authorization header value
            
        Returns:
            Tuple of (is_valid, jwt_claims)
            
        Raises:
            HTTPException: If JWT validation fails
        """
        # Extract token from "Bearer <token>" format
        try:
            scheme, token = authorization_header.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme. Expected 'Bearer'",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format. Expected 'Bearer <token>'",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate JWT token
        try:
            # Prepare JWT decode options
            options = {
                "verify_signature": True,
                "verify_exp": self.config.jwt_require_exp,
                "verify_iat": self.config.jwt_require_iat,
                "verify_aud": self.config.jwt_audience is not None,
                "verify_iss": self.config.jwt_issuer is not None,
            }
            
            # Decode and validate JWT
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
                audience=self.config.jwt_audience,
                issuer=self.config.jwt_issuer,
                leeway=self.config.jwt_leeway,
                options=options
            )
            
            logger.debug(f"JWT token validated successfully for subject: {payload.get('sub', 'unknown')}")
            return True, payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            logger.error(f"Unexpected error during JWT validation: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during authentication"
            )
    
    def _validate_api_key(self, authorization_header: str) -> Tuple[bool, None]:
        """Validate API Key authentication.
        
        Args:
            authorization_header: The Authorization header value
            
        Returns:
            Tuple of (is_valid, None)
            
        Raises:
            HTTPException: If API key validation fails
        """
        # For API key, we expect the key directly in the authorization header
        # or we could extract it from a custom header
        api_key = authorization_header.strip()
        
        if not self.config.api_keys or api_key not in self.config.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        logger.debug("API key validated successfully")
        return True, None
    
    def extract_metadata_filters(self, jwt_claims: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract metadata filters from JWT claims.
        
        Args:
            jwt_claims: JWT payload claims
            
        Returns:
            Metadata filters if present in JWT claims, None otherwise
        """
        if not jwt_claims:
            return None
            
        # Look for metadata_filter claim in JWT
        metadata_filter = jwt_claims.get("metadata_filter")
        if metadata_filter and isinstance(metadata_filter, dict):
            logger.debug(f"Extracted metadata filters from JWT: {metadata_filter}")
            return metadata_filter
            
        return None
    
    def merge_filters(self, 
                     request_filter: Optional[Dict[str, Any]], 
                     jwt_filter: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Merge request filters with JWT-based filters.
        
        Args:
            request_filter: Filters from the request
            jwt_filter: Filters extracted from JWT claims
            
        Returns:
            Merged filters using $and operator if both exist
        """
        if not request_filter and not jwt_filter:
            return None
        elif not request_filter:
            return jwt_filter
        elif not jwt_filter:
            return request_filter
        else:
            # Both filters exist, merge them with $and
            return {
                "$and": [request_filter, jwt_filter]
            }
