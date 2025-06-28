"""
API Configuration using Pydantic Settings
Environment-based configuration for different deployment scenarios
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    """API Configuration Settings"""
    
    # API Server Settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Scene Matcher Settings
    cache_dir: str = Field(default="./cache", description="Cache directory path")
    catalog_path: str = Field(default="./data/product-catalog.csv", description="Product catalog path")
    batch_size: int = Field(default=64, description="Batch size for processing")
    quality_target: float = Field(default=0.75, description="Quality target threshold")
    
    # API Limits
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max upload file size (10MB)")
    max_recommendations: int = Field(default=50, description="Maximum recommendations per request")
    
    # Performance Settings
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(default=10, description="Max concurrent requests")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_file = ".env"
        env_prefix = "SCENE_MATCHER_"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Development settings
def get_dev_settings() -> Settings:
    """Development environment settings"""
    return Settings(
        debug=True,
        host="127.0.0.1",
        port=8000,
        log_level="DEBUG",
        batch_size=32  # Smaller for development
    )

# Production settings  
def get_prod_settings() -> Settings:
    """Production environment settings"""
    return Settings(
        debug=False,
        host="0.0.0.0",
        port=8000,
        log_level="INFO",
        batch_size=128,  # Larger for production
        max_concurrent_requests=50
    )