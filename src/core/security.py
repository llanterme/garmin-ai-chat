"""Security utilities for authentication and encryption."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from cryptography.fernet import Fernet
from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings
from .exceptions import AuthenticationError


class PasswordHandler:
    """Handle password hashing and verification."""

    def __init__(self) -> None:
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return self.pwd_context.hash(password)


class TokenHandler:
    """Handle JWT token creation and validation."""

    def __init__(self) -> None:
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str, expected_type: str = "access") -> Dict[str, Any]:
        """Verify and decode token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            token_type = payload.get("type")
            
            if token_type != expected_type:
                raise AuthenticationError(f"Invalid token type. Expected {expected_type}, got {token_type}")
            
            return payload
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}") from e


class EncryptionHandler:
    """Handle encryption and decryption of sensitive data."""

    def __init__(self) -> None:
        # Use the configured encryption key
        key = settings.garmin_encryption_key.encode()
        # Pad or truncate to 32 bytes, then base64 encode for Fernet
        import base64
        self.cipher = Fernet(base64.urlsafe_b64encode(key))

    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data."""
        if isinstance(data, str):
            data = data.encode()
        encrypted = self.cipher.encrypt(data)
        return encrypted.decode()

    def decrypt(self, encrypted_data: Union[str, bytes]) -> str:
        """Decrypt data."""
        if isinstance(encrypted_data, str):
            encrypted_data = encrypted_data.encode()
        decrypted = self.cipher.decrypt(encrypted_data)
        return decrypted.decode()


# Global instances
password_handler = PasswordHandler()
token_handler = TokenHandler()
encryption_handler = EncryptionHandler()