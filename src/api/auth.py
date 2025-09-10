"""Authentication API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import AuthenticationError, ConflictError
from ..db.base import get_db
from ..db.models import User
from ..schemas.auth import (
    GarminCredentials,
    PasswordChange,
    TokenRefresh,
    TokenResponse,
    UserLogin,
    UserRegister,
)
from ..schemas.common import SuccessResponse
from ..schemas.sync import GarminConnectionTest
from ..schemas.background_task import TaskCreationResponse
from ..schemas.user import UserDataDeletionResponse, UserResponse
from ..services.auth import AuthService
from ..services.sync import SyncService
from ..services.user_cleanup import UserCleanupService
from .dependencies import get_current_active_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Register a new user."""
    try:
        auth_service = AuthService(db)
        user_id, tokens = await auth_service.register_user(user_data)
        return tokens
    except ConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate user and return access token."""
    try:
        # Convert OAuth2PasswordRequestForm to UserLogin format
        login_data = UserLogin(email=form_data.username, password=form_data.password)
        auth_service = AuthService(db)
        user_id, tokens = await auth_service.authenticate_user(login_data)
        return tokens
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: TokenRefresh,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Refresh access token using refresh token."""
    try:
        auth_service = AuthService(db)
        tokens = await auth_service.refresh_token(refresh_data.refresh_token)
        return tokens
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """Get current user information."""
    return UserResponse.from_db_model(current_user)


@router.post("/garmin-credentials", response_model=SuccessResponse)
async def update_garmin_credentials(
    credentials: GarminCredentials,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> SuccessResponse:
    """Update user's Garmin Connect credentials."""
    try:
        auth_service = AuthService(db)
        success = await auth_service.update_garmin_credentials(current_user.id, credentials)
        
        if success:
            return SuccessResponse(message="Garmin credentials updated successfully")
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update Garmin credentials"
            )
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/garmin-test", response_model=GarminConnectionTest)
async def test_garmin_connection(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GarminConnectionTest:
    """Test Garmin Connect connection."""
    try:
        sync_service = SyncService(db)
        success, message, user_info = await sync_service.test_garmin_connection(current_user.id)
        
        return GarminConnectionTest(
            success=success,
            message=message,
            user_info=user_info,
            test_timestamp=datetime.utcnow(),
        )
    except Exception as e:
        return GarminConnectionTest(
            success=False,
            message=f"Connection test failed: {str(e)}",
            user_info=None,
            test_timestamp=datetime.utcnow(),
        )


@router.post("/change-password", response_model=SuccessResponse)
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> SuccessResponse:
    """Change user password."""
    try:
        auth_service = AuthService(db)
        success = await auth_service.change_password(
            current_user.id,
            password_data.current_password,
            password_data.new_password,
        )
        
        if success:
            return SuccessResponse(message="Password changed successfully")
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to change password"
            )
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/me/data", response_model=TaskCreationResponse)
async def delete_user_data(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete all user data including activities, sync history, and vectors in background.
    
    This is a destructive operation that cannot be undone.
    It will remove:
    - All user activities from the database
    - All sync history records
    - All user vectors from Pinecone
    """
    try:
        cleanup_service = UserCleanupService()
        
        # Start background cleanup
        task_id = await cleanup_service.start_background_cleanup(
            user_id=current_user.id,
            session=db
        )
        
        return TaskCreationResponse(
            task_id=task_id,
            message="Data deletion started successfully",
            status_url=f"/api/v1/tasks/{task_id}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user data: {str(e)}"
        )