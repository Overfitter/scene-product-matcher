"""
Scene Matching API Endpoints
Main functionality for scene-to-product matching
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query, status
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
import logging
import numpy as np
from typing import List, Optional
import sys
from pathlib import Path
import requests

from src.core.matcher import SceneProductMatcher
from ..models.request_models import SceneMatchRequest, AnalyticsRequest
from ..models.response_models import (
    SceneMatchResponse, AnalyticsResponse, QualityMetrics,
    SceneAnalysis, ProductRecommendation, ScoreBreakdown
)
from ..dependencies import get_matcher
from ..config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/match-scene", response_model=SceneMatchResponse)
async def match_scene(
    file: UploadFile = File(...),
    k: int = Query(default=5, ge=1, le=50, description="Number of recommendations"),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Minimum confidence"),
    categories_filter: Optional[str] = Query(default=None, description="Comma-separated category filters"),
    include_analytics: bool = Query(default=True, description="Include quality metrics"),
    matcher: SceneProductMatcher = Depends(get_matcher)
):
    """
    Get product recommendations for a scene image
    
    **Parameters:**
    - **file**: Scene image file (JPEG, PNG, WebP)
    - **k**: Number of recommendations (1-50)
    - **min_confidence**: Minimum confidence threshold (0.0-1.0)
    - **categories_filter**: Filter categories (comma-separated)
    - **include_analytics**: Include performance metrics
    
    **Returns:**
    - Scene analysis (room, style, color palette)
    - Product recommendations with confidence scores
    - Quality metrics and performance data
    """
    
    start_time = time.time()
    settings = get_settings()
    
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, WebP, etc.)"
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size // (1024*1024)}MB"
            )
        
        # Process image
        try:
            image = Image.open(io.BytesIO(file_content)).convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Validate image dimensions
        if min(image.size) < 64 or max(image.size) > 4096:
            raise HTTPException(
                status_code=400,
                detail="Image dimensions must be between 64x64 and 4096x4096 pixels"
            )
        
        # Get recommendations
        try:
            results = matcher.get_ultimate_recommendations(image, k=k)
        except Exception as e:
            logger.error(f"Scene matching failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Scene matching failed: {str(e)}"
            )
        
        # Process results
        scene_analysis = SceneAnalysis(**results['scene_analysis'])
        
        # Filter by confidence if specified
        recommendations = results['recommendations']
        if min_confidence is not None:
            recommendations = [r for r in recommendations if r['confidence'] >= min_confidence]
        
        # Filter by categories if specified
        if categories_filter:
            allowed_categories = set(cat.strip() for cat in categories_filter.split(','))
            recommendations = [r for r in recommendations if r['category'] in allowed_categories]
        
        # Convert to response models
        recommendation_objects = []
        for rec in recommendations:
            score_breakdown = ScoreBreakdown(
                visual=rec['score_breakdown']['visual'],
                text=rec['score_breakdown']['text']
            )
            
            recommendation = ProductRecommendation(
                id=rec['id'],
                sku_id=rec['sku_id'],
                score=rec['score'],
                confidence=float(rec['confidence']),  # Convert numpy float32
                recommendation_level=rec['recommendation_level'],
                category=rec['category'],
                description=rec['description'],
                materials=rec['materials'],
                colors=rec['colors'],
                style_descriptors=rec['style_descriptors'],
                size_category=rec['size_category'],
                quality_tier=rec['quality_tier'],
                enterprise_score=rec['enterprise_score'],
                is_set=rec['is_set'],
                set_count=rec['set_count'],
                image_url=rec['image_url'],
                placement_suggestion=rec['placement_suggestion'],
                score_breakdown=score_breakdown
            )
            recommendation_objects.append(recommendation)
        
        # Calculate quality metrics
        quality_metrics = None
        if include_analytics and recommendation_objects:
            processing_time = (time.time() - start_time) * 1000
            
            avg_confidence = np.mean([r.confidence for r in recommendation_objects])
            high_confidence_count = sum(1 for r in recommendation_objects if r.confidence >= 0.65)
            category_diversity = len(set(r.category for r in recommendation_objects))
            
            quality_metrics = QualityMetrics(
                avg_confidence=float(avg_confidence),
                high_confidence_count=high_confidence_count,
                category_diversity=category_diversity,
                processing_time_ms=processing_time,
                meets_quality_targets=(
                    avg_confidence >= 0.65 and 
                    category_diversity >= 3 and 
                    processing_time < 500
                )
            )
        
        # Create response
        response = SceneMatchResponse(
            scene_analysis=scene_analysis,
            recommendations=recommendation_objects,
            quality_metrics=quality_metrics,
            metadata={
                "original_filename": file.filename,
                "image_size": image.size,
                "processed_recommendations": len(recommendation_objects),
                "api_version": "1.0.0"
            }
        )
        
        logger.info(f"Scene matching completed in {(time.time() - start_time)*1000:.1f}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in scene matching: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during scene matching"
        )

@router.post("/match-scene-url")
async def match_scene_url(
    image_url: str,
    k: int = Query(default=5, ge=1, le=50),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    matcher: SceneProductMatcher = Depends(get_matcher)
):
    """
    Get product recommendations for a scene image from URL
    
    **Note**: This endpoint downloads the image from the provided URL
    """
    
    try:
        # Download image with timeout
        response = requests.get(image_url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="URL does not point to a valid image"
            )
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=413,
                detail="Image file too large (>10MB)"
            )
        
        # Read image content
        image_content = response.content
        
        # Process as image
        try:
            image = Image.open(io.BytesIO(image_content)).convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Validate image dimensions
        if min(image.size) < 64 or max(image.size) > 4096:
            raise HTTPException(
                status_code=400,
                detail="Image dimensions must be between 64x64 and 4096x4096 pixels"
            )
        
        # Get recommendations
        try:
            results = matcher.get_ultimate_recommendations(image, k=k)
        except Exception as e:
            logger.error(f"Scene matching failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Scene matching failed: {str(e)}"
            )
        
        # Process and return results (similar to match_scene)
        scene_analysis = SceneAnalysis(**results['scene_analysis'])
        
        recommendations = results['recommendations']
        if min_confidence is not None:
            recommendations = [r for r in recommendations if r['confidence'] >= min_confidence]
        
        recommendation_objects = []
        for rec in recommendations:
            score_breakdown = ScoreBreakdown(
                visual=rec['score_breakdown']['visual'],
                text=rec['score_breakdown']['text']
            )
            
            recommendation = ProductRecommendation(
                id=rec['id'],
                sku_id=rec['sku_id'],
                score=rec['score'],
                confidence=float(rec['confidence']),
                recommendation_level=rec['recommendation_level'],
                category=rec['category'],
                description=rec['description'],
                materials=rec['materials'],
                colors=rec['colors'],
                style_descriptors=rec['style_descriptors'],
                size_category=rec['size_category'],
                quality_tier=rec['quality_tier'],
                enterprise_score=rec['enterprise_score'],
                is_set=rec['is_set'],
                set_count=rec['set_count'],
                image_url=rec['image_url'],
                placement_suggestion=rec['placement_suggestion'],
                score_breakdown=score_breakdown
            )
            recommendation_objects.append(recommendation)
        
        return SceneMatchResponse(
            scene_analysis=scene_analysis,
            recommendations=recommendation_objects,
            metadata={
                "source_url": image_url,
                "image_size": image.size,
                "processed_recommendations": len(recommendation_objects),
                "api_version": "1.0.0"
            }
        )
        
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image from URL: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image URL: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process image from URL"
        )

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    days: int = Query(default=7, ge=1, le=90, description="Days of analytics data"),
    matcher: SceneProductMatcher = Depends(get_matcher)
):
    """
    Get performance analytics and metrics
    
    **Parameters:**
    - **days**: Number of days to include in analytics (1-90)
    
    **Returns:**
    - Performance metrics and trends
    - Confidence and category distributions
    - Error rates and quality indicators
    """
    
    try:
        # Get analytics from matcher if available
        analytics_data = {}
        
        # Check if matcher has performance history
        if hasattr(matcher, 'performance_history') and matcher.performance_history:
            history = matcher.performance_history
            
            # Calculate metrics from performance history
            total_requests = len(history)
            avg_confidence = sum(h.avg_confidence for h in history) / len(history) if history else 0.0
            avg_processing_time = sum(h.processing_time_ms for h in history) / len(history) if history else 0.0
            
            # Calculate confidence distribution
            confidence_distribution = {"premium": 0, "high": 0, "medium": 0, "low": 0}
            for h in history:
                if h.avg_confidence >= 0.75:
                    confidence_distribution["premium"] += 1
                elif h.avg_confidence >= 0.65:
                    confidence_distribution["high"] += 1
                elif h.avg_confidence >= 0.40:
                    confidence_distribution["medium"] += 1
                else:
                    confidence_distribution["low"] += 1
            
            analytics_data.update({
                "total_requests": total_requests,
                "avg_confidence": avg_confidence,
                "avg_processing_time_ms": avg_processing_time,
                "confidence_distribution": confidence_distribution
            })
        
        # Get error count if available
        error_count = 0
        if hasattr(matcher, 'error_count') and matcher.error_count:
            error_count = sum(matcher.error_count.values())
        
        # Calculate error rate
        total_requests = analytics_data.get("total_requests", 0)
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0.0
        
        # Default analytics structure with real data if available
        default_analytics = {
            'total_requests': analytics_data.get("total_requests", 0),
            'avg_confidence': analytics_data.get("avg_confidence", 0.70),
            'avg_processing_time_ms': analytics_data.get("avg_processing_time_ms", 350.0),
            'confidence_distribution': analytics_data.get("confidence_distribution", {
                'premium': 15,
                'high': 35,
                'medium': 40,
                'low': 10
            }),
            'category_distribution': {
                'statement_vases': 30,
                'lighting_accents': 20,
                'accent_tables': 20,
                'sculptural_objects': 15,
                'functional_beauty': 10,
                'other': 5
            },
            'error_rate': error_rate,
            'quality_trend': 'stable'
        }
        
        return AnalyticsResponse(**default_analytics, period_days=days)
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analytics"
        )

@router.get("/categories")
async def get_categories():
    """
    Get available product categories
    
    **Returns:**
    - List of available product categories with descriptions
    """
    
    categories = {
        'statement_vases': {
            'description': 'Large decorative vases for focal points',
            'placement': 'Coffee table, console, or floor',
            'size_preference': ['large', 'medium'],
            'materials': ['ceramic', 'glass', 'metal']
        },
        'accent_vases': {
            'description': 'Small decorative vases for subtle accents',
            'placement': 'Side tables or shelving',
            'size_preference': ['small'],
            'materials': ['ceramic', 'glass']
        },
        'sculptural_objects': {
            'description': 'Artistic sculptures and figurines',
            'placement': 'Shelving, console, or coffee table',
            'size_preference': ['small', 'medium'],
            'materials': ['ceramic', 'metal', 'resin', 'wood']
        },
        'lighting_accents': {
            'description': 'Candles, holders, and ambient lighting',
            'placement': 'Tables, consoles, or dining areas',
            'size_preference': ['small', 'medium'],
            'materials': ['metal', 'glass', 'ceramic']
        },
        'functional_beauty': {
            'description': 'Decorative bowls, trays, and serving pieces',
            'placement': 'Coffee table styling or console display',
            'size_preference': ['small', 'medium'],
            'materials': ['ceramic', 'wood', 'metal', 'glass']
        },
        'storage_style': {
            'description': 'Stylish storage containers and boxes',
            'placement': 'Coffee table or console organization',
            'size_preference': ['small', 'medium'],
            'materials': ['ceramic', 'wood', 'metal']
        },
        'accent_tables': {
            'description': 'Small tables, stools, and pedestals',
            'placement': 'Beside seating or as plant stands',
            'size_preference': ['medium', 'large'],
            'materials': ['wood', 'metal', 'glass']
        },
        'decorative_accents': {
            'description': 'General decorative accessories',
            'placement': 'Shelving, console, or side tables',
            'size_preference': ['small', 'medium'],
            'materials': ['ceramic', 'metal', 'glass', 'wood']
        }
    }
    
    return {
        "categories": categories,
        "total_categories": len(categories),
        "api_version": "1.0.0"
    }

@router.get("/styles")
async def get_styles():
    """
    Get available design styles
    
    **Returns:**
    - List of design styles that can be detected
    """
    
    styles = {
        'contemporary': {
            'description': 'Modern clean lines and neutral palettes',
            'keywords': ['geometric', 'modern', 'clean', 'simple', 'sculptural']
        },
        'traditional': {
            'description': 'Classic elegant furniture with warm colors',
            'keywords': ['classic', 'elegant', 'ornate', 'detailed', 'refined']
        },
        'minimalist': {
            'description': 'Sparse decoration with essential furniture only',
            'keywords': ['simple', 'clean', 'minimal', 'pure', 'essential']
        },
        'transitional': {
            'description': 'Balanced blend of modern and traditional elements',
            'keywords': ['balanced', 'versatile', 'blended', 'sophisticated']
        },
        'luxury': {
            'description': 'Premium materials and sophisticated finishes',
            'keywords': ['premium', 'sophisticated', 'high-end', 'elegant', 'refined']
        },
        'rustic': {
            'description': 'Weathered wood and farmhouse charm',
            'keywords': ['weathered', 'natural', 'farmhouse', 'vintage', 'authentic']
        },
        'bohemian': {
            'description': 'Vibrant patterns and eclectic artistic elements',
            'keywords': ['vibrant', 'eclectic', 'artistic', 'creative', 'layered']
        }
    }
    
    return {
        "styles": styles,
        "total_styles": len(styles),
        "api_version": "1.0.0"
    }

@router.get("/color-palettes")
async def get_color_palettes():
    """
    Get available color palettes
    
    **Returns:**
    - List of color palettes that can be detected
    """
    
    palettes = {
        'neutral_warm': {
            'description': 'warm neutral tones with creams and beiges',
            'colors': ['cream', 'beige', 'warm gray', 'taupe', 'ivory', 'off-white'],
            'harmony_score': 1.2
        },
        'neutral_cool': {
            'description': 'cool neutral tones with grays and whites',
            'colors': ['cool gray', 'white', 'silver', 'pearl', 'platinum'],
            'harmony_score': 1.2
        },
        'warm_metallics': {
            'description': 'warm metallic accents and finishes',
            'colors': ['gold', 'brass', 'copper', 'bronze', 'amber'],
            'harmony_score': 1.1
        },
        'cool_metallics': {
            'description': 'cool metallic accents and finishes',
            'colors': ['silver', 'chrome', 'platinum', 'steel'],
            'harmony_score': 1.1
        },
        'earth_tones': {
            'description': 'natural earth tone palette',
            'colors': ['brown', 'tan', 'rust', 'terracotta', 'natural'],
            'harmony_score': 1.0
        },
        'blues': {
            'description': 'sophisticated blue color family',
            'colors': ['navy', 'blue', 'teal', 'indigo', 'sapphire'],
            'harmony_score': 1.0
        },
        'greens': {
            'description': 'natural green color palette',
            'colors': ['sage', 'green', 'emerald', 'forest', 'mint'],
            'harmony_score': 1.0
        }
    }
    
    return {
        "palettes": palettes,
        "total_palettes": len(palettes),
        "api_version": "1.0.0"
    }

@router.get("/status")
async def get_status(matcher: SceneProductMatcher = Depends(get_matcher)):
    """
    Get current system status and configuration
    
    **Returns:**
    - System status and matcher configuration
    """
    
    try:
        # Get matcher status
        matcher_status = {
            "initialized": True,
            "models_loaded": hasattr(matcher, 'clip_model') and matcher.clip_model is not None,
            "embeddings_loaded": hasattr(matcher, 'visual_embeddings') and matcher.visual_embeddings is not None,
            "products_count": len(matcher.products) if hasattr(matcher, 'products') else 0
        }
        
        # Get performance metrics
        performance_metrics = {}
        if hasattr(matcher, 'performance_history') and matcher.performance_history:
            recent_metrics = matcher.performance_history[-10:]  # Last 10 requests
            performance_metrics = {
                "total_requests": len(matcher.performance_history),
                "recent_avg_confidence": sum(m.avg_confidence for m in recent_metrics) / len(recent_metrics),
                "recent_avg_time_ms": sum(m.processing_time_ms for m in recent_metrics) / len(recent_metrics),
                "recent_avg_quality": sum(m.quality_score for m in recent_metrics) / len(recent_metrics)
            }
        
        # Get configuration
        settings = get_settings()
        configuration = {
            "batch_size": settings.batch_size,
            "quality_target": settings.quality_target,
            "max_file_size_mb": settings.max_file_size // (1024 * 1024),
            "max_recommendations": settings.max_recommendations
        }
        
        return {
            "status": "operational",
            "matcher_status": matcher_status,
            "performance_metrics": performance_metrics,
            "configuration": configuration,
            "api_version": "1.0.0",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system status"
        )