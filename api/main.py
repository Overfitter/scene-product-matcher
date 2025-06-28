"""
FastAPI Main Application for Ultimate Scene Matcher
Production-ready API service with monitoring, validation, and error handling
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import os
from pathlib import Path

from api.config import get_settings
from api.dependencies import get_matcher_service
from api.middleware import (
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware
)
from api.routers import health, scene_matching

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("ultimate_matcher_api")

# Global matcher instance
matcher_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global matcher_service
    
    logger.info("🚀 Starting Ultimate Scene Matcher API...")
    
    try:
        # Initialize the matcher service
        settings = get_settings()
        
        # Import here to avoid circular imports
        from src.core.llm_matcher import build_ultimate_matcher
        
        logger.info("🔧 Initializing Ultimate Matcher...")
        matcher_service = await build_ultimate_matcher(
            openai_api_key=settings.openai_api_key,
            catalog_path=settings.catalog_path,
            cache_dir=settings.cache_dir
        )
        
        logger.info("✅ Ultimate Matcher API ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize matcher: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down Ultimate Scene Matcher API...")

# Create FastAPI app
app = FastAPI(
    title="Ultimate Scene Matcher API",
    description="AI-powered scene-to-product recommendation system with LLM intelligence and FAISS retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(scene_matching.router, prefix="/api/v1", tags=["Scene Matching"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Scene Product Matcher API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "capabilities": [
            "Scene analysis with GPT-4o vision",
            "LLM-enhanced product categorization",
            "FAISS-powered similarity search",
            "Contextual re-ranking",
            "Multi-dimensional scoring"
        ]
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

def get_matcher():
    """Get the global matcher instance"""
    global matcher_service
    if matcher_service is None:
        raise HTTPException(status_code=503, detail="Matcher service not initialized")
    return matcher_service

# Make matcher available to dependencies
app.state.get_matcher = get_matcher

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1,  # Use 1 worker due to global state
        log_level="info"
    )