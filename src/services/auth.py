"""Authentication service for user management and JWT handling."""

from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import AuthenticationError, ConflictError, NotFoundError
from ..core.logging import get_logger
from ..core.security import encryption_handler, password_handler, token_handler
from ..db.repositories import UserRepository
from ..schemas.auth import GarminCredentials, TokenResponse, UserLogin, UserRegister

logger = get_logger(__name__)


class AuthService:
    """Service for handling authentication and user management."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.user_repo = UserRepository(session)

    async def register_user(self, user_data: UserRegister) -> Tuple[str, TokenResponse]:
        """Register a new user."""
        try:
            # Check if user already exists
            existing_user = await self.user_repo.get_by_email(user_data.email)
            if existing_user:
                raise ConflictError("User with this email already exists")

            # Hash password
            hashed_password = password_handler.get_password_hash(user_data.password)

            # Create user
            user_create_data = {
                "email": user_data.email,
                "hashed_password": hashed_password,
                "full_name": user_data.full_name,
                "is_active": True,
            }

            user = await self.user_repo.create(user_create_data)
            logger.info(f"Created new user: {user.email}")

            # Generate tokens
            tokens = self._generate_tokens(user.id, user.email)

            # Update last login
            await self.user_repo.update(user.id, {"last_login": datetime.utcnow()})

            return user.id, tokens

        except ConflictError:
            raise
        except Exception as e:
            logger.error(f"Failed to register user: {str(e)}")
            raise AuthenticationError(f"Registration failed: {str(e)}")

    async def authenticate_user(self, login_data: UserLogin) -> Tuple[str, TokenResponse]:
        """Authenticate a user with email and password."""
        try:
            # Get user by email
            user = await self.user_repo.get_by_email(login_data.email)
            if not user:
                raise AuthenticationError("Invalid email or password")

            # Check if user is active
            if not user.is_active:
                raise AuthenticationError("User account is disabled")

            # Verify password
            if not password_handler.verify_password(login_data.password, user.hashed_password):
                raise AuthenticationError("Invalid email or password")

            logger.info(f"User authenticated: {user.email}")

            # Generate tokens
            tokens = self._generate_tokens(user.id, user.email)

            # Update last login
            await self.user_repo.update(user.id, {"last_login": datetime.utcnow()})

            return user.id, tokens

        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Failed to authenticate user: {str(e)}")
            raise AuthenticationError(f"Authentication failed: {str(e)}")

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token."""
        try:
            # Verify refresh token
            payload = token_handler.verify_token(refresh_token, "refresh")
            user_id = payload.get("sub")
            email = payload.get("email")

            if not user_id or not email:
                raise AuthenticationError("Invalid refresh token")

            # Check if user still exists and is active
            user = await self.user_repo.get_by_id(user_id)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")

            # Generate new tokens
            return self._generate_tokens(user_id, email)

        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Failed to refresh token: {str(e)}")
            raise AuthenticationError(f"Token refresh failed: {str(e)}")

    async def get_current_user(self, token: str):
        """Get current user from access token."""
        try:
            # Verify token
            payload = token_handler.verify_token(token, "access")
            user_id = payload.get("sub")

            if not user_id:
                raise AuthenticationError("Invalid access token")

            # Get user
            user = await self.user_repo.get_by_id(user_id)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")

            return user

        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Failed to get current user: {str(e)}")
            raise AuthenticationError(f"Failed to get user: {str(e)}")

    async def update_garmin_credentials(
        self, user_id: str, credentials: GarminCredentials
    ) -> bool:
        """Update user's Garmin Connect credentials."""
        try:
            # Encrypt credentials
            encrypted_username = encryption_handler.encrypt(credentials.username)
            encrypted_password = encryption_handler.encrypt(credentials.password)

            # Update user
            await self.user_repo.update_garmin_credentials(
                user_id, encrypted_username, encrypted_password
            )

            logger.info(f"Updated Garmin credentials for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update Garmin credentials: {str(e)}")
            raise AuthenticationError(f"Failed to update credentials: {str(e)}")

    async def get_user_garmin_credentials(self, user_id: str) -> Optional[Tuple[str, str, Optional[dict]]]:
        """Get user's encrypted Garmin credentials."""
        try:
            user = await self.user_repo.get_by_id(user_id)
            if not user or not user.garmin_username or not user.garmin_password:
                return None

            return (
                user.garmin_username,
                user.garmin_password,
                user.garmin_session_data,
            )

        except Exception as e:
            logger.error(f"Failed to get Garmin credentials: {str(e)}")
            return None

    async def update_garmin_session_data(
        self, user_id: str, session_data: Optional[dict]
    ) -> bool:
        """Update user's Garmin session data."""
        try:
            await self.user_repo.update(user_id, {"garmin_session_data": session_data})
            return True
        except Exception as e:
            logger.error(f"Failed to update Garmin session data: {str(e)}")
            return False

    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user's password."""
        try:
            # Get user
            user = await self.user_repo.get_by_id(user_id)
            if not user:
                raise NotFoundError("User not found")

            # Verify current password
            if not password_handler.verify_password(current_password, user.hashed_password):
                raise AuthenticationError("Current password is incorrect")

            # Hash new password
            new_hashed_password = password_handler.get_password_hash(new_password)

            # Update password
            await self.user_repo.update(user_id, {"hashed_password": new_hashed_password})

            logger.info(f"Password changed for user: {user.email}")
            return True

        except (AuthenticationError, NotFoundError):
            raise
        except Exception as e:
            logger.error(f"Failed to change password: {str(e)}")
            raise AuthenticationError(f"Failed to change password: {str(e)}")

    def _generate_tokens(self, user_id: str, email: str) -> TokenResponse:
        """Generate access and refresh tokens for a user."""
        token_data = {"sub": user_id, "email": email}

        access_token = token_handler.create_access_token(token_data)
        refresh_token = token_handler.create_refresh_token(token_data)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=1800,  # 30 minutes in seconds
        )