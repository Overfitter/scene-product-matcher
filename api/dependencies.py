"""
FastAPI Dependencies
Dependency injection for scene matcher and other services
"""

from fastapi import Depends, HTTPException, status
import sys
from pathlib import Path
from typing import Optional

from src.core.matcher import SceneProductMatcher

# Global matcher instance (will be set during app startup)
_matcher_instance: Optional[SceneProductMatcher] = None

def set_matcher_instance(matcher: SceneProductMatcher):
    """Set the global matcher instance (called during startup)"""
    global _matcher_instance
    _matcher_instance = matcher
    print(f"✅ Matcher instance set: {matcher is not None}")

def get_matcher() -> SceneProductMatcher:
    """
    Dependency to get the scene matcher instance
    
    Raises HTTPException if matcher is not initialized
    """
    global _matcher_instance
    
    if _matcher_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scene matcher is not initialized. Please try again later."
        )
    
    return _matcher_instance

def get_matcher_optional() -> Optional[SceneProductMatcher]:
    """
    Dependency to get the scene matcher instance (optional)
    
    Returns None if matcher is not initialized (used for health checks)
    """
    global _matcher_instance
    return _matcher_instance

# Request validation dependencies
def validate_image_file(file_size: int = 0) -> bool:
    """
    Validate uploaded image file
    """
    max_size = 10 * 1024 * 1024  # 10MB
    
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
        )
    
    return True

def validate_recommendations_count(k: int) -> int:
    """
    Validate recommendations count parameter
    """
    if k < 1 or k > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of recommendations must be between 1 and 50"
        )
    
    return k

def validate_confidence_threshold(confidence: Optional[float]) -> Optional[float]:
    """
    Validate confidence threshold parameter
    """
    if confidence is not None:
        if confidence < 0.0 or confidence > 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Confidence threshold must be between 0.0 and 1.0"
            )
    
    return confidence