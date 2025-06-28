"""
Health Check API Endpoints
System status and monitoring endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
import psutil
import time
import sys
from pathlib import Path

from src.core.matcher import SceneProductMatcher
from ..models.response_models import HealthResponse
from ..dependencies import get_matcher_optional
from ..config import get_settings

# CREATE THE ROUTER - This was missing!
router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check(matcher: SceneProductMatcher = Depends(get_matcher_optional)):
    """
    Basic health check endpoint
    
    **Returns:**
    - Service status and basic system information
    - Scene matcher readiness status
    - Memory usage and cache status
    """
    
    # Check matcher status
    matcher_ready = matcher is not None
    cache_status = "unknown"
    
    if matcher_ready:
        try:
            # Check if embeddings are loaded
            if hasattr(matcher, 'visual_embeddings') and matcher.visual_embeddings is not None:
                cache_status = "loaded"
            else:
                cache_status = "not_loaded"
        except:
            cache_status = "error"
    
    # Get memory usage
    memory_usage_mb = None
    try:
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        pass
    
    return HealthResponse(
        status="healthy" if matcher_ready else "initializing",
        version="1.0.0",
        matcher_ready=matcher_ready,
        cache_status=cache_status,
        memory_usage_mb=memory_usage_mb
    )

@router.get("/detailed")
async def detailed_health_check(matcher: SceneProductMatcher = Depends(get_matcher_optional)):
    """
    Detailed health check with comprehensive system information
    
    **Returns:**
    - Detailed system metrics and configuration
    - Model loading status and performance indicators
    """
    
    settings = get_settings()
    
    # System information
    system_info = {}
    try:
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "python_version": sys.version,
            "platform": sys.platform
        }
    except:
        system_info = {"error": "Could not retrieve system information"}
    
    # Matcher status
    matcher_info = {
        "ready": False,
        "models_loaded": False,
        "embeddings_loaded": False,
        "product_count": 0,
        "cache_dir_exists": Path(settings.cache_dir).exists()
    }
    
    if matcher:
        try:
            matcher_info.update({
                "ready": True,
                "models_loaded": hasattr(matcher, 'clip_model') and matcher.clip_model is not None,
                "embeddings_loaded": hasattr(matcher, 'visual_embeddings') and matcher.visual_embeddings is not None,
                "product_count": len(matcher.products) if hasattr(matcher, 'products') else 0
            })
        except Exception as e:
            matcher_info["error"] = str(e)
    
    # Configuration
    config_info = {
        "cache_dir": settings.cache_dir,
        "batch_size": settings.batch_size,
        "quality_target": settings.quality_target,
        "max_file_size_mb": settings.max_file_size // (1024 * 1024),
        "max_recommendations": settings.max_recommendations
    }
    
    return {
        "status": "healthy" if matcher_info["ready"] else "initializing",
        "timestamp": time.time(),
        "system_info": system_info,
        "matcher_info": matcher_info,
        "configuration": config_info,
        "api_version": "1.0.0"
    }

@router.get("/ready")
async def readiness_check(matcher: SceneProductMatcher = Depends(get_matcher_optional)):
    """
    Kubernetes-style readiness probe
    
    **Returns:**
    - Simple ready/not-ready status for load balancers
    """
    
    if matcher is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Check if embeddings are loaded
    if not hasattr(matcher, 'visual_embeddings') or matcher.visual_embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not loaded")
    
    return {"status": "ready"}

@router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe
    
    **Returns:**
    - Simple alive status for health monitoring
    """
    
    return {"status": "alive", "timestamp": time.time()}

@router.get("/metrics")
async def get_metrics(matcher: SceneProductMatcher = Depends(get_matcher_optional)):
    """
    Prometheus-style metrics endpoint
    
    **Returns:**
    - Key performance metrics in a structured format
    """
    
    metrics = {
        "scene_matcher_requests_total": 0,
        "scene_matcher_errors_total": 0,
        "scene_matcher_avg_confidence": 0.0,
        "scene_matcher_avg_processing_time_ms": 0.0,
        "scene_matcher_memory_usage_bytes": 0,
        "scene_matcher_products_loaded": 0
    }
    
    if matcher:
        try:
            # Get metrics from matcher performance history
            if hasattr(matcher, 'performance_history') and matcher.performance_history:
                history = matcher.performance_history
                metrics.update({
                    "scene_matcher_requests_total": len(history),
                    "scene_matcher_avg_confidence": sum(h.avg_confidence for h in history) / len(history),
                    "scene_matcher_avg_processing_time_ms": sum(h.processing_time_ms for h in history) / len(history)
                })
            
            # Error count
            if hasattr(matcher, 'error_count'):
                metrics["scene_matcher_errors_total"] = sum(matcher.error_count.values())
            
            # Product count
            if hasattr(matcher, 'products'):
                metrics["scene_matcher_products_loaded"] = len(matcher.products)
            
            # Memory usage
            try:
                process = psutil.Process()
                metrics["scene_matcher_memory_usage_bytes"] = process.memory_info().rss
            except:
                pass
                
        except Exception as e:
            metrics["error"] = str(e)
    
    return metrics