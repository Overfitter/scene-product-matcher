"""
Dependency injection for Ultimate Scene Matcher API
"""

from fastapi import Depends, HTTPException, Request, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Annotated
import time
import asyncio
import uuid
from functools import lru_cache

from api.config import get_settings, Settings
from src.core.llm_matcher import SceneProductMatcher

# Security
security = HTTPBearer(auto_error=False)

def get_settings_dependency() -> Settings:
    """Get settings dependency"""
    return get_settings()

def get_matcher_service(request: Request) -> SceneProductMatcher:
    """Get the Ultimate Matcher service instance"""
    try:
        matcher = request.app.state.get_matcher()
        return matcher
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Matcher service unavailable"
        )

def generate_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())

def get_request_id() -> str:
    """Dependency to generate request ID"""
    return generate_request_id()

# Rate limiting store (in production, use Redis)
request_counts = {}
request_times = {}

def rate_limit_dependency(
    request: Request,
    settings: Settings = Depends(get_settings_dependency)
) -> bool:
    """Rate limiting dependency"""
    
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    if client_ip in request_times:
        request_times[client_ip] = [
            req_time for req_time in request_times[client_ip]
            if current_time - req_time < settings.rate_limit_window
        ]
    else:
        request_times[client_ip] = []
    
    # Check rate limit
    if len(request_times[client_ip]) >= settings.rate_limit_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
    # Add current request
    request_times[client_ip].append(current_time)
    
    return True

def validate_api_key(
    settings: Settings = Depends(get_settings_dependency),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(default=None)
) -> bool:
    """Validate API key if configured"""
    
    # If no API keys configured, allow all requests
    if not settings.api_keys:
        return True
    
    # Check header-based API key
    if x_api_key and x_api_key in settings.api_keys:
        return True
    
    # Check bearer token
    if credentials and credentials.credentials in settings.api_keys:
        return True
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key"
    )

def validate_image_size(
    content_length: Optional[int] = Header(default=None),
    settings: Settings = Depends(get_settings_dependency)
) -> bool:
    """Validate image size"""
    
    if content_length and content_length > settings.max_image_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large. Maximum size: {settings.max_image_size / (1024*1024):.1f}MB"
        )
    
    return True

def validate_k_parameter(k: Optional[int] = None) -> int:
    """Validate and normalize k parameter"""
    settings = get_settings()
    
    if k is None:
        return settings.default_k
    
    if k < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'k' must be at least 1"
        )
    
    if k > settings.max_k:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Parameter 'k' cannot exceed {settings.max_k}"
        )
    
    return k

# Concurrent request limiting
active_requests = set()

async def concurrent_request_limit(
    request_id: str = Depends(get_request_id),
    settings: Settings = Depends(get_settings_dependency)
) -> str:
    """Limit concurrent requests"""
    
    if len(active_requests) >= settings.max_concurrent_requests:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Too many concurrent requests. Please try again later."
        )
    
    active_requests.add(request_id)
    
    try:
        yield request_id
    finally:
        active_requests.discard(request_id)

# Health check dependencies
async def check_matcher_health(
    matcher: SceneProductMatcher = Depends(get_matcher_service)
) -> dict:
    """Check matcher service health"""
    
    try:
        # Basic health check
        health_info = {
            "matcher_initialized": matcher is not None,
            "total_products": len(matcher.products) if matcher else 0,
            "indexes_built": (
                matcher.visual_index is not None and 
                matcher.text_index is not None
            ) if matcher else False
        }
        
        return health_info
        
    except Exception as e:
        return {
            "matcher_initialized": False,
            "error": str(e)
        }

async def check_openai_health(
    settings: Settings = Depends(get_settings_dependency)
) -> dict:
    """Check OpenAI API health"""
    
    try:
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
        
        # Simple API test
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",
                messages=[{"role": "user", "content": "Health check"}],
                max_tokens=5
            ),
            timeout=5.0
        )
        
        return {
            "openai_api": "healthy",
            "response_time": "< 5s"
        }
        
    except asyncio.TimeoutError:
        return {
            "openai_api": "timeout",
            "error": "API request timed out"
        }
    except Exception as e:
        return {
            "openai_api": "unhealthy",
            "error": str(e)
        }

# Performance monitoring
performance_metrics = {
    "request_count": 0,
    "total_response_time": 0.0,
    "error_count": 0,
    "start_time": time.time()
}

def update_performance_metrics(response_time: float, error: bool = False):
    """Update performance metrics"""
    performance_metrics["request_count"] += 1
    performance_metrics["total_response_time"] += response_time
    
    if error:
        performance_metrics["error_count"] += 1

def get_performance_metrics() -> dict:
    """Get current performance metrics"""
    
    uptime = time.time() - performance_metrics["start_time"]
    avg_response_time = (
        performance_metrics["total_response_time"] / 
        max(1, performance_metrics["request_count"])
    )
    error_rate = (
        performance_metrics["error_count"] / 
        max(1, performance_metrics["request_count"])
    )
    
    return {
        "uptime_seconds": uptime,
        "total_requests": performance_metrics["request_count"],
        "avg_response_time": avg_response_time,
        "error_rate": error_rate,
        "requests_per_minute": performance_metrics["request_count"] / (uptime / 60) if uptime > 0 else 0
    }

# Image validation
import base64
from PIL import Image
from io import BytesIO

def validate_and_decode_image(image_data: str) -> Image.Image:
    """Validate and decode base64 image"""
    
    try:
        # Handle data URL format
        if image_data.startswith('data:image/'):
            header, data = image_data.split(',', 1)
            image_bytes = base64.b64decode(data)
        else:
            image_bytes = base64.b64decode(image_data)
        
        # Validate image size
        if len(image_bytes) > get_settings().max_image_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image too large"
            )
        
        # Load and validate image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Validate image dimensions
        if min(image.size) < 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image too small (minimum 32x32 pixels)"
            )
        
        if max(image.size) > 4096:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image too large (maximum 4096x4096 pixels)"
            )
        
        return image
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image data: {str(e)}"
        )

# Dependency combinations for common use cases
CommonDeps = Annotated[
    tuple,
    Depends(lambda: (
        Depends(validate_api_key),
        Depends(rate_limit_dependency),
        Depends(validate_image_size)
    ))
]

MatcherDeps = Annotated[
    tuple,
    Depends(lambda: (
        Depends(get_matcher_service),
        Depends(get_request_id)
    ))
]