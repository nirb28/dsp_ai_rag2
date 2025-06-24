"""
Authentication models for the RAG service API.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ProviderPermission(str, Enum):
    """Enumeration of available provider permissions."""
    OPENAI = "openai"
    GROQ = "groq"
    ALL = "all"


class Scope(str, Enum):
    """Enumeration of available API scopes/permissions."""
    UPLOAD = "upload"
    QUERY = "query"
    CONFIGURE = "configure"
    COLLECTIONS = "collections"
    ALL = "all"


class AuthClient(BaseModel):
    """Authentication client model."""
    client_id: str
    client_secret: str
    name: Optional[str] = None
    description: Optional[str] = None
    scopes: List[Scope] = [Scope.ALL]
    providers: List[ProviderPermission] = [ProviderPermission.ALL]
    is_active: bool = True

    def has_scope(self, scope: Scope) -> bool:
        """Check if client has the specified scope."""
        return Scope.ALL in self.scopes or scope in self.scopes

    def has_provider_permission(self, provider: ProviderPermission) -> bool:
        """Check if client has permission to use the specified provider."""
        return ProviderPermission.ALL in self.providers or provider in self.providers


class TokenPayload(BaseModel):
    """Token payload model."""
    client_id: str
    scopes: List[Scope]
    providers: List[ProviderPermission]
    exp: Optional[int] = None


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    scope: str = Field(..., description="Space-separated list of scopes")


class TokenRequest(BaseModel):
    """Token request model for client credentials flow."""
    grant_type: str = Field(..., description="Must be 'client_credentials'")
    client_id: str
    client_secret: str
    scope: Optional[str] = Field(None, description="Space-separated list of requested scopes")
