"""
Admin Authentication: JWT-based auth for admin dashboard endpoints.
"""
import jwt
import time
import logging
from functools import wraps

from fastapi import Request, HTTPException
from pydantic import BaseModel

from src.config import ADMIN_USERNAME, ADMIN_PASSWORD, JWT_SECRET

logger = logging.getLogger(__name__)

TOKEN_EXPIRY_HOURS = 24


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    expires_in: int


def create_token(username: str) -> str:
    """Create a JWT token for an authenticated admin."""
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + (TOKEN_EXPIRY_HOURS * 3600),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token. Raises on invalid/expired tokens."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def authenticate(username: str, password: str) -> str | None:
    """Check credentials and return a JWT token, or None if invalid."""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return create_token(username)
    return None


async def require_admin(request: Request):
    """FastAPI dependency: extract and verify admin JWT from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token")
    token = auth_header[7:]
    payload = verify_token(token)
    return payload
