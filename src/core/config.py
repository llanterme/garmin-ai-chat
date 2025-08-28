"""Application configuration settings."""

from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Garmin AI Chat API"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = Field(default="development", pattern="^(development|staging|production)$")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # Database
    database_url: str = Field(
        default="mysql+aiomysql://root:root@localhost:3306/garmin_ai_chat",
        description="Database connection URL"
    )
    database_echo: bool = False
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Authentication
    secret_key: str = Field(
        default="your-super-secret-key-change-this-in-production",
        min_length=32,
        description="JWT secret key"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30

    # Garmin Connect
    garmin_encryption_key: str = Field(
        default="your-default-32-char-key-for-dev",
        min_length=32,
        max_length=32,
        description="32-character key for encrypting Garmin credentials"
    )

    # AI/ML Configuration
    openai_api_key: str = Field(
        description="OpenAI API key for embeddings and chat completions"
    )
    pinecone_api_key: str = Field(
        description="Pinecone API key for vector database"
    )
    pinecone_environment: str = Field(
        default="us-east-1",
        description="Pinecone environment region"
    )
    pinecone_index_name: str = Field(
        default="garmin-fitness-activities",
        description="Pinecone index name for storing activity embeddings"
    )

    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: Optional[str] = None

    # CORS
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    allowed_methods: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: list[str] = ["*"]

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Ensure secret key is sufficiently complex in production."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v

    @field_validator("garmin_encryption_key")
    @classmethod
    def validate_encryption_key(cls, v: str) -> str:
        """Ensure encryption key is exactly 32 characters."""
        if len(v) != 32:
            raise ValueError("Garmin encryption key must be exactly 32 characters long")
        return v

    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.database_url,
            "echo": self.database_echo,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
        }

    @property
    def cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration dictionary."""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": True,
            "allow_methods": self.allowed_methods,
            "allow_headers": self.allowed_headers,
        }



@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()