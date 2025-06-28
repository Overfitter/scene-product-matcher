"""
Configuration management for Ultimate Scene Matcher API
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # File Paths
    catalog_path: str = Field(default="data/product-catalog.csv", env="CATALOG_PATH")
    cache_dir: str = Field(default="./cache", env="CACHE_DIR")
    
    # Performance Settings
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Recommendation Settings
    default_k: int = Field(default=5, env="DEFAULT_K")
    max_k: int = Field(default=20, env="MAX_K")
    
    # Image Processing
    max_image_size: int = Field(default=10 * 1024 * 1024, env="MAX_IMAGE_SIZE")  # 10MB
    allowed_image_types: list = Field(
        default=["image/jpeg", "image/jpg", "image/png", "image/webp"],
        env="ALLOWED_IMAGE_TYPES"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # CORS Settings
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    
    # Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    api_keys: Optional[list] = Field(default=None, env="API_KEYS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Environment-specific configurations
def get_development_settings() -> Settings:
    """Development environment settings"""
    settings = get_settings()
    settings.debug = True
    settings.log_level = "DEBUG"
    return settings

def get_production_settings() -> Settings:
    """Production environment settings"""
    settings = get_settings()
    settings.debug = False
    settings.log_level = "INFO"
    settings.cors_origins = ["https://yourdomain.com"]  # Configure for production
    return settings

def get_test_settings() -> Settings:
    """Test environment settings"""
    settings = get_settings()
    settings.debug = True
    settings.catalog_path = "tests/test_catalog.csv"
    settings.cache_dir = "./test_cache"
    return settings

# Validation functions
def validate_settings(settings: Settings) -> bool:
    """Validate settings configuration"""
    
    # Check required files exist
    if not os.path.exists(settings.catalog_path):
        raise FileNotFoundError(f"Catalog file not found: {settings.catalog_path}")
    
    # Validate OpenAI API key format
    if not settings.openai_api_key.startswith("sk-"):
        raise ValueError("Invalid OpenAI API key format")
    
    # Validate rate limiting
    if settings.rate_limit_requests <= 0 or settings.rate_limit_window <= 0:
        raise ValueError("Invalid rate limiting configuration")
    
    # Validate recommendation settings
    if settings.max_k <= 0 or settings.default_k <= 0:
        raise ValueError("Invalid recommendation settings")
    
    if settings.default_k > settings.max_k:
        raise ValueError("Default k cannot be greater than max k")
    
    return True

# Initialize settings validation on import
try:
    _settings = get_settings()
    validate_settings(_settings)
except Exception as e:
    print(f"⚠️ Settings validation warning: {e}")