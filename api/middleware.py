"""
Custom middleware for Ultimate Scene Matcher API
"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import logging
import json
import traceback
from typing import Callable
import uuid

from api.dependencies import update_performance_metrics

logger = logging.getLogger("ultimate_matcher_api")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        
        logger.info(
            f"Request started | {request_id} | {request.method} {request.url.path} | "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed | {request_id} | {response.status_code} | "
                f"Time: {process_time:.3f}s"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            # Update metrics
            update_performance_metrics(process_time, error=response.status_code >= 400)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"Request failed | {request_id} | Error: {str(e)} | "
                f"Time: {process_time:.3f}s"
            )
            
            # Update metrics
            update_performance_metrics(process_time, error=True)
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "message": "Internal server error",
                    "request_id": request_id,
                    "timestamp": time.time()
                },
                headers={"X-Request-ID": request_id}
            )

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_counts = {}
        self.request_times = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Rate limiting is handled in dependencies
        # This middleware just adds rate limit headers
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = "100"
        response.headers["X-RateLimit-Window"] = "60"
        response.headers["X-RateLimit-Remaining"] = "99"  # Simplified
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
            
        except Exception as e:
            # Get request ID if available
            request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
            
            # Log the error
            logger.error(
                f"Unhandled exception | {request_id} | {type(e).__name__}: {str(e)}"
            )
            logger.debug(f"Traceback | {request_id} | {traceback.format_exc()}")
            
            # Determine error type and response
            if isinstance(e, ValueError):
                status_code = 400
                message = f"Invalid input: {str(e)}"
            elif isinstance(e, FileNotFoundError):
                status_code = 404
                message = "Resource not found"
            elif isinstance(e, PermissionError):
                status_code = 403
                message = "Permission denied"
            elif isinstance(e, TimeoutError):
                status_code = 408
                message = "Request timeout"
            else:
                status_code = 500
                message = "Internal server error"
            
            # Return structured error response
            error_response = {
                "error": True,
                "message": message,
                "error_type": type(e).__name__,
                "status_code": status_code,
                "request_id": request_id,
                "timestamp": time.time()
            }
            
            # Add debug info in development
            try:
                from api.config import get_settings
                settings = get_settings()
                if settings.debug:
                    error_response["debug"] = {
                        "exception": str(e),
                        "traceback": traceback.format_exc().split('\n')
                    }
            except:
                pass
            
            return JSONResponse(
                status_code=status_code,
                content=error_response,
                headers={"X-Request-ID": request_id}
            )

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

class CacheControlMiddleware(BaseHTTPMiddleware):
    """Cache control middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Set cache headers based on endpoint
        if request.url.path.startswith("/health"):
            # Don't cache health checks
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        elif request.url.path.startswith("/api/v1/recommend"):
            # Don't cache recommendations (they're personalized)
            response.headers["Cache-Control"] = "no-cache, private"
        elif request.url.path in ["/", "/docs", "/redoc"]:
            # Cache static content for 1 hour
            response.headers["Cache-Control"] = "public, max-age=3600"
        else:
            # Default: cache for 5 minutes
            response.headers["Cache-Control"] = "public, max-age=300"
        
        return response

class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Request size limiting middleware"""
    
    def __init__(self, app: ASGIApp, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_size:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": True,
                            "message": f"Request too large. Maximum size: {self.max_size / (1024*1024):.1f}MB",
                            "status_code": 413,
                            "timestamp": time.time()
                        }
                    )
            except ValueError:
                pass
        
        return await call_next(request)

class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Health check middleware for load balancers"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Simple health check endpoint for load balancers
        if request.url.path == "/ping":
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "timestamp": time.time()}
            )
        
        return await call_next(request)