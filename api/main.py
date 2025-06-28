"""
FastAPI Application for Ultimate Scene Matcher
Enterprise-grade API with async support and comprehensive error handling
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import sys
import logging
from pathlib import Path

from src.core.matcher import SceneProductMatcher
from .routers import scene_matching, health
from .middleware.logging import LoggingMiddleware
from .config import get_settings
from .dependencies import set_matcher_instance  # Import the setter function

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown events"""
    
    # Startup
    logger.info("🚀 Starting Ultimate Scene Matcher API...")
    try:
        settings = get_settings()
        
        # Initialize the matcher
        logger.info("🔧 Initializing Scene Matcher...")
        matcher_instance = SceneProductMatcher(
            cache_dir=settings.cache_dir,
            batch_size=settings.batch_size,
            quality_target=settings.quality_target
        )
        
        # Load catalog and build embeddings
        logger.info("📂 Loading product catalog...")
        matcher_instance.load_and_process_catalog(settings.catalog_path)
        
        logger.info("🔥 Building embeddings...")
        await matcher_instance.build_embeddings_async()
        
        # IMPORTANT: Set the global instance in dependencies
        set_matcher_instance(matcher_instance)
        
        logger.info("✅ Ultimate Scene Matcher API ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize matcher: {e}")
        raise RuntimeError(f"Startup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Ultimate Scene Matcher API...")
    # Clear the global instance
    set_matcher_instance(None)

# Create FastAPI app with lifespan
app = FastAPI(
    title="Ultimate Scene Matcher API",
    description="Enterprise-grade scene-to-product matching with AI-powered recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom logging middleware
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health Check"]
)

app.include_router(
    scene_matching.router,
    prefix="/api/v1",
    tags=["Scene Matching"]
)

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if get_settings().debug else "An unexpected error occurred",
            "status_code": 500
        }
    )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Ultimate Scene Matcher API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health/",
            "docs": "/docs",
            "scene_matching": "/api/v1/match-scene",
            "analytics": "/api/v1/analytics"
        }
    }

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )