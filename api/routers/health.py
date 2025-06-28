"""
Health check router for Ultimate Scene Matcher API
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import asyncio
import time
import psutil
import sys

from api.models.request_models import HealthCheckRequest
from api.models.response_models import HealthCheckResponse
from api.dependencies import (
    check_matcher_health,
    check_openai_health,
    get_performance_metrics,
    get_settings_dependency
)
from api.config import Settings

router = APIRouter()

@router.get("/", response_model=HealthCheckResponse)
async def health_check(
    include_detailed: bool = False,
    check_dependencies: bool = True,
    matcher_health: dict = Depends(check_matcher_health),
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Basic health check endpoint
    
    Returns the health status of the API service and its dependencies.
    """
    
    # Basic system health
    system_health = {
        "api_status": "healthy",
        "matcher_service": "healthy" if matcher_health.get("matcher_initialized") else "unhealthy",
        "total_products": matcher_health.get("total_products", 0),
        "indexes_status": "ready" if matcher_health.get("indexes_built") else "not_ready"
    }
    
    # Determine overall status
    if not matcher_health.get("matcher_initialized") or not matcher_health.get("indexes_built"):
        status = "unhealthy"
    else:
        status = "healthy"
    
    # Get performance metrics
    performance = get_performance_metrics()
    
    response_data = {
        "status": status,
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "uptime_seconds": performance.get("uptime_seconds"),
        "system_health": system_health
    }
    
    # Add detailed information if requested
    if include_detailed:
        response_data["performance"] = performance
        
        # System resource information
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            response_data["system_health"].update({
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_usage_percent": cpu_percent,
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "python_version": sys.version
            })
        except Exception as e:
            response_data["system_health"]["resource_error"] = str(e)
    
    # Check dependencies if requested
    if check_dependencies:
        dependencies = {}
        
        # Check OpenAI API
        try:
            openai_health = await asyncio.wait_for(
                check_openai_health(settings),
                timeout=10.0
            )
            dependencies["openai"] = openai_health
            
            # Update overall status if OpenAI is unhealthy
            if openai_health.get("openai_api") != "healthy":
                status = "degraded"
                
        except asyncio.TimeoutError:
            dependencies["openai"] = {
                "openai_api": "timeout",
                "error": "Health check timed out"
            }
            status = "degraded"
        except Exception as e:
            dependencies["openai"] = {
                "openai_api": "error",
                "error": str(e)
            }
            status = "degraded"
        
        response_data["dependencies"] = dependencies
        response_data["status"] = status
    
    return HealthCheckResponse(**response_data)

@router.get("/ready", response_model=dict)
async def readiness_check(
    matcher_health: dict = Depends(check_matcher_health)
):
    """
    Readiness probe for Kubernetes/container orchestration
    
    Returns 200 if the service is ready to accept requests.
    """
    
    if not matcher_health.get("matcher_initialized") or not matcher_health.get("indexes_built"):
        raise HTTPException(
            status_code=503,
            detail="Service not ready - matcher not initialized or indexes not built"
        )
    
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/live", response_model=dict)
async def liveness_check():
    """
    Liveness probe for Kubernetes/container orchestration
    
    Returns 200 if the service is alive (basic health check).
    """
    
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/detailed", response_model=HealthCheckResponse)
async def detailed_health_check(
    matcher_health: dict = Depends(check_matcher_health),
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Detailed health check with comprehensive system information
    """
    
    return await health_check(
        include_detailed=True,
        check_dependencies=True,
        matcher_health=matcher_health,
        settings=settings
    )

@router.get("/dependencies", response_model=dict)
async def dependencies_health_check(
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Check health of external dependencies only
    """
    
    dependencies = {}
    
    # Check OpenAI API
    try:
        openai_health = await asyncio.wait_for(
            check_openai_health(settings),
            timeout=10.0
        )
        dependencies["openai"] = openai_health
    except Exception as e:
        dependencies["openai"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check file system access
    try:
        import os
        from pathlib import Path
        
        catalog_exists = os.path.exists(settings.catalog_path)
        cache_writable = os.access(settings.cache_dir, os.W_OK)
        
        dependencies["filesystem"] = {
            "status": "healthy" if catalog_exists and cache_writable else "unhealthy",
            "catalog_exists": catalog_exists,
            "cache_writable": cache_writable,
            "catalog_path": settings.catalog_path,
            "cache_dir": settings.cache_dir
        }
    except Exception as e:
        dependencies["filesystem"] = {
            "status": "error",
            "error": str(e)
        }
    
    return {
        "dependencies": dependencies,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics", response_model=dict)
async def performance_metrics():
    """
    Get current performance metrics
    """
    
    metrics = get_performance_metrics()
    
    # Add additional metrics
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        metrics.update({
            "memory_usage_percent": memory.percent,
            "cpu_usage_percent": cpu_percent,
            "memory_available_mb": memory.available / (1024**2)
        })
    except Exception as e:
        metrics["resource_error"] = str(e)
    
    return {
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/version", response_model=dict)
async def version_info():
    """
    Get version and build information
    """
    
    return {
        "service": "Ultimate Scene Matcher API",
        "version": "1.0.0",
        "build_date": "2024-01-01",  # Would be set during build
        "git_commit": "abc123",      # Would be set during build
        "python_version": sys.version,
        "environment": "production",  # Would be set via environment
        "timestamp": datetime.utcnow().isoformat()
    }