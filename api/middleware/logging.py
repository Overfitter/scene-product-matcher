"""
Request Logging Middleware
Log all API requests with timing and performance metrics
"""

import time
import logging
import uuid
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

logger = logging.getLogger("api.requests")

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests with performance metrics
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Start timing
        start_time = time.time()
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request start
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {client_ip} - {user_agent}"
        )
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"[{request_id}] {response.status_code} "
                f"in {process_time*1000:.1f}ms"
            )
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Calculate error time
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"[{request_id}] ERROR {str(e)} "
                f"in {process_time*1000:.1f}ms"
            )
            
            raise

class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """
    Advanced middleware for detailed performance logging
    """
    
    def __init__(self, app, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Get request details
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        content_length = request.headers.get("content-length", "0")
        
        try:
            response = await call_next(request)
            
            # Calculate metrics
            process_time = time.time() - start_time
            status_code = response.status_code
            
            # Log detailed metrics
            log_data = {
                "method": method,
                "path": path,
                "query_params": query_params,
                "status_code": status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "request_size_bytes": content_length,
                "response_size_bytes": getattr(response, "headers", {}).get("content-length", "unknown")
            }
            
            # Log level based on performance
            if process_time > self.slow_request_threshold:
                logger.warning(f"SLOW REQUEST: {log_data}")
            elif status_code >= 400:
                logger.warning(f"ERROR REQUEST: {log_data}")
            else:
                logger.debug(f"REQUEST: {log_data}")
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(f"FAILED REQUEST: {method} {path} - {str(e)} in {process_time*1000:.1f}ms")
            raise

def setup_logging():
    """
    Setup logging configuration for the API
    """
    
    # Configure API request logger
    api_logger = logging.getLogger("api.requests")
    api_logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    api_logger.addHandler(console_handler)
    
    # File handler for API logs
    file_handler = logging.FileHandler('logs/api_requests.log')
    file_handler.setFormatter(formatter)
    api_logger.addHandler(file_handler)
    
    return api_logger