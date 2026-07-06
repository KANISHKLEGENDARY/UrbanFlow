"""
UrbanFlow -- Authentication API Routes

Endpoints:
    POST /api/v1/auth/register  -- Create a new user account
    POST /api/v1/auth/login     -- Authenticate and receive JWT tokens
    POST /api/v1/auth/refresh   -- Refresh an expired access token
    GET  /api/v1/auth/me        -- Get current user info
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

import config
from api.auth.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from api.schemas import (
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthRefreshRequest,
    AuthTokenResponse,
    AuthUserResponse,
)
from db.database import database_manager
from db.models import User

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/register", response_model=AuthTokenResponse, status_code=201)
async def register(req: AuthRegisterRequest):
    """Register a new user account.

    Returns access + refresh tokens on successful registration.
    """
    if not database_manager.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable — registration requires PostgreSQL",
        )

    async with database_manager.session() as session:
        # Check for existing username
        existing = (
            await session.execute(
                select(User).where(
                    (User.username == req.username) | (User.email == req.email)
                )
            )
        ).scalar_one_or_none()

        if existing:
            field = "username" if existing.username == req.username else "email"
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A user with that {field} already exists",
            )

        user = User(
            username=req.username,
            email=req.email,
            hashed_password=hash_password(req.password),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

    token_data = {"sub": user.username, "user_id": user.id, "email": user.email, "is_admin": user.is_admin}
    return AuthTokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
        token_type="bearer",
        username=user.username,
        email=user.email,
    )


@router.post("/login", response_model=AuthTokenResponse)
async def login(req: AuthLoginRequest):
    """Authenticate a user and return JWT tokens.

    Accepts username and password. Returns access + refresh tokens.
    """
    if not database_manager.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable — login requires PostgreSQL",
        )

    async with database_manager.session() as session:
        user = (
            await session.execute(
                select(User).where(User.username == req.username)
            )
        ).scalar_one_or_none()

        if user is None or not verify_password(req.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated",
            )

    token_data = {"sub": user.username, "user_id": user.id, "email": user.email, "is_admin": user.is_admin}
    return AuthTokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
        token_type="bearer",
        username=user.username,
        email=user.email,
    )


@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh_token(req: AuthRefreshRequest):
    """Exchange a valid refresh token for a new access token."""
    payload = decode_token(req.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — provide a refresh token",
        )

    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )

    # Build new access token from the refresh token claims
    token_data = {
        "sub": username,
        "user_id": payload.get("user_id", 0),
        "email": payload.get("email", ""),
        "is_admin": payload.get("is_admin", False),
    }

    return AuthTokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=req.refresh_token,  # reuse same refresh token
        token_type="bearer",
        username=username,
        email=payload.get("email", ""),
    )


@router.get("/me", response_model=AuthUserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return information about the currently authenticated user."""
    return AuthUserResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        is_admin=current_user.get("is_admin", False),
    )
