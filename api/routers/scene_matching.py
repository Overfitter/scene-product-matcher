"""
Scene matching router for Ultimate Scene Matcher API
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
import asyncio
import time
import logging
from typing import List, Optional
import aiohttp
from io import BytesIO
from PIL import Image

from api.models.request_models import (
    RecommendationRequest,
    RecommendationURLRequest,
    BatchRecommendationRequest,
    SystemStatsRequest,
    CatalogUpdateRequest,
    FileUploadRecommendationRequest
)
from api.models.response_models import (
    RecommendationResponse,
    BatchRecommendationResponse,
    SystemStatsResponse,
    CatalogUpdateResponse,
    ErrorResponse
)
from api.dependencies import (
    get_matcher_service,
    get_request_id,
    validate_api_key,
    rate_limit_dependency,
    validate_k_parameter,
    validate_and_decode_image,
    concurrent_request_limit,
    get_settings_dependency
)
from api.config import Settings
from src.core.llm_matcher import SceneProductMatcher

router = APIRouter()
logger = logging.getLogger("ultimate_matcher_api")

@router.post("/recommend-upload", response_model=RecommendationResponse)
async def get_recommendations_from_upload(
    # File upload
    file: UploadFile = File(..., description="Scene image file (JPEG, PNG, WebP)"),
    
    # Form parameters
    k: Optional[int] = Form(default=5, description="Number of recommendations (1-20)"),
    room_type: Optional[str] = Form(default=None, description="Room type hint"),
    style_preference: Optional[str] = Form(default=None, description="Style preference"),
    enable_llm_reranking: Optional[bool] = Form(default=True, description="Enable LLM re-ranking"),
    include_reasoning: Optional[bool] = Form(default=True, description="Include reasoning"),
    
    # Dependencies
    matcher: SceneProductMatcher = Depends(get_matcher_service),
    request_id: str = Depends(get_request_id),
    _: bool = Depends(validate_api_key),
    __: bool = Depends(rate_limit_dependency),
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Get scene-to-product recommendations from uploaded image file
    
    Upload an image file directly and get product recommendations.
    Supports JPEG, PNG, and WebP formats.
    """
    
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG, PNG, WebP)"
            )
        
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Read and validate file size
        file_content = await file.read()
        
        if len(file_content) > settings.max_image_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image too large. Maximum size: {settings.max_image_size / (1024*1024):.1f}MB"
            )
        
        if len(file_content) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image file appears to be corrupted or too small"
            )
        
        # Load and validate image
        try:
            scene_image = Image.open(BytesIO(file_content)).convert('RGB')
            
            # Validate dimensions
            if min(scene_image.size) < 32:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Image too small (minimum 32x32 pixels)"
                )
            
            if max(scene_image.size) > 4096:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Image too large (maximum 4096x4096 pixels)"
                )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Validate k parameter
        k = validate_k_parameter(k)
        
        logger.info(
            f"Processing file upload recommendation | {request_id} | "
            f"File: {file.filename} ({file.content_type}) | "
            f"Size: {len(file_content)} bytes | k={k}"
        )
        
        # Get recommendations from matcher
        recommendations = await matcher.get_recommendations(
            scene_image_path=scene_image,
            k=k
        )
        
        # Add request metadata
        recommendations['request_id'] = request_id
        recommendations['timestamp'] = time.time()
        recommendations['upload_info'] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'file_size_bytes': len(file_content),
            'image_dimensions': f"{scene_image.width}x{scene_image.height}"
        }
        
        # Add request context if provided
        if room_type or style_preference:
            recommendations['request_context'] = {
                'room_type_hint': room_type,
                'style_preference': style_preference
            }
        
        # Process LLM re-ranking option
        if not enable_llm_reranking:
            for rec in recommendations['recommendations']:
                if 'llm_insights' in rec:
                    rec['llm_insights']['reranked'] = False
        
        # Remove reasoning if not requested
        if not include_reasoning:
            for rec in recommendations['recommendations']:
                rec.pop('reasoning', None)
        
        processing_time = time.time() - start_time
        logger.info(
            f"Upload recommendation completed | {request_id} | "
            f"Time: {processing_time:.3f}s | "
            f"Results: {len(recommendations['recommendations'])}"
        )
        
        return RecommendationResponse(**recommendations)
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Upload recommendation failed | {request_id} | "
            f"Time: {processing_time:.3f}s | Error: {str(e)}"
        )
        
        if isinstance(e, HTTPException):
            raise e
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload recommendation processing failed: {str(e)}"
        )

@router.post("/recommend-url", response_model=RecommendationResponse)
async def get_recommendations_from_url(
    request: RecommendationURLRequest,
    matcher: SceneProductMatcher = Depends(get_matcher_service),
    request_id: str = Depends(get_request_id),
    _: bool = Depends(validate_api_key),
    __: bool = Depends(rate_limit_dependency),
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Get scene-to-product recommendations from image URL
    
    Downloads an image from the provided URL and returns product recommendations.
    """
    
    start_time = time.time()
    
    try:
        # Download image from URL
        async with aiohttp.ClientSession() as session:
            async with session.get(request.image_url, timeout=10) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download image: HTTP {response.status}"
                    )
                
                image_data = await response.read()
                
                # Validate image size
                if len(image_data) > settings.max_image_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Downloaded image too large"
                    )
                
                # Load image
                scene_image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Validate k parameter
        k = validate_k_parameter(request.k)
        
        logger.info(f"Processing URL recommendation request | {request_id} | URL: {request.image_url} | k={k}")
        
        # Get recommendations
        recommendations = await matcher.get_recommendations(
            scene_image_path=scene_image,
            k=k
        )
        
        # Add request metadata
        recommendations['request_id'] = request_id
        recommendations['timestamp'] = time.time()
        recommendations['source_url'] = request.image_url
        
        processing_time = time.time() - start_time
        logger.info(f"URL recommendation completed | {request_id} | Time: {processing_time:.3f}s")
        
        return RecommendationResponse(**recommendations)
        
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download image: {str(e)}"
        )
    except Exception as e:
        logger.error(f"URL recommendation failed | {request_id} | Error: {str(e)}")
        
        if isinstance(e, HTTPException):
            raise e
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL recommendation processing failed: {str(e)}"
        )

async def _process_single_recommendation(
    scene_request: RecommendationRequest,
    matcher: SceneProductMatcher,
    sub_request_id: str
) -> RecommendationResponse:
    """Process a single recommendation within a batch"""
    
    try:
        # Validate and decode image
        scene_image = validate_and_decode_image(scene_request.image_data)
        
        # Get recommendations
        recommendations = await matcher.get_recommendations(
            scene_image_path=scene_image,
            k=scene_request.k or 5
        )
        
        # Add metadata
        recommendations['request_id'] = sub_request_id
        recommendations['timestamp'] = time.time()
        
        return RecommendationResponse(**recommendations)
        
    except Exception as e:
        logger.error(f"Single recommendation failed | {sub_request_id} | Error: {str(e)}")
        raise e

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    request: SystemStatsRequest = Depends(),
    matcher: SceneProductMatcher = Depends(get_matcher_service),
    _: bool = Depends(validate_api_key)
):
    """
    Get system statistics and performance metrics
    """
    
    try:
        # Get matcher stats
        matcher_stats = matcher.stats if hasattr(matcher, 'stats') else {}
        
        # System information
        system_info = {
            'service_name': 'Ultimate Scene Matcher API',
            'version': '1.0.0',
            'total_products': len(matcher.products),
            'matcher_initialized': True,
            'llm_reranking_enabled': True
        }
        
        # Performance metrics
        performance_metrics = None
        if request.include_performance:
            from api.dependencies import get_performance_metrics
            performance_metrics = get_performance_metrics()
        
        # Usage statistics (simplified)
        usage_stats = None
        if request.include_usage:
            usage_stats = {
                'total_requests': matcher_stats.get('total_requests', 0),
                'avg_response_time': matcher_stats.get('avg_response_time', 0),
                'avg_confidence': matcher_stats.get('avg_confidence', 0)
            }
        
        # Resource usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            resource_usage = {
                'memory_usage_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'cpu_usage_percent': cpu_percent
            }
        except Exception:
            resource_usage = {'error': 'Resource monitoring unavailable'}
        
        return SystemStatsResponse(
            success=True,
            system_info=system_info,
            performance_metrics=performance_metrics,
            usage_stats=usage_stats,
            resource_usage=resource_usage
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system stats: {str(e)}"
        )

@router.post("/catalog/update", response_model=CatalogUpdateResponse)
async def update_catalog(
    request: CatalogUpdateRequest,
    background_tasks: BackgroundTasks,
    matcher: SceneProductMatcher = Depends(get_matcher_service),
    request_id: str = Depends(get_request_id),
    _: bool = Depends(validate_api_key),
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Update the product catalog and rebuild indexes
    
    This is a potentially long-running operation that rebuilds the entire
    product catalog and FAISS indexes.
    """
    
    try:
        catalog_path = request.catalog_path or settings.catalog_path
        
        # Validate catalog file exists
        import os
        if not os.path.exists(catalog_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Catalog file not found: {catalog_path}"
            )
        
        logger.info(f"Starting catalog update | {request_id} | Path: {catalog_path}")
        
        # Add background task for catalog update
        background_tasks.add_task(
            _update_catalog_background,
            matcher,
            catalog_path,
            request.force_rebuild,
            request_id
        )
        
        return CatalogUpdateResponse(
            success=True,
            message="Catalog update started in background",
            products_processed=0,  # Will be updated in background
            processing_time_seconds=0,
            indexes_rebuilt=False,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Catalog update failed to start | {request_id} | Error: {str(e)}")
        
        if isinstance(e, HTTPException):
            raise e
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start catalog update: {str(e)}"
        )

async def _update_catalog_background(
    matcher: SceneProductMatcher,
    catalog_path: str,
    force_rebuild: bool,
    request_id: str
):
    """Background task to update catalog"""
    
    try:
        start_time = time.time()
        logger.info(f"Background catalog update started | {request_id}")
        
        # Process catalog
        products = await matcher.process_product_catalog(catalog_path)
        
        # Rebuild indexes
        matcher.build_faiss_indexes(products)
        
        # Save indexes
        matcher.save_indexes(str(matcher.cache_dir / "indexes"))
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Background catalog update completed | {request_id} | "
            f"Products: {len(products)} | Time: {processing_time:.1f}s"
        )
        
    except Exception as e:
        logger.error(f"Background catalog update failed | {request_id} | Error: {str(e)}")

@router.get("/catalog/info", response_model=dict)
async def get_catalog_info(
    matcher: SceneProductMatcher = Depends(get_matcher_service),
    _: bool = Depends(validate_api_key)
):
    """
    Get information about the current catalog
    """
    
    try:
        total_products = len(matcher.products)
        
        # Category distribution
        category_dist = {}
        style_dist = {}
        quality_dist = {}
        
        for product in matcher.products:
            # Category distribution
            category = product.category
            category_dist[category] = category_dist.get(category, 0) + 1
            
            # Style distribution
            style = product.style
            style_dist[style] = style_dist.get(style, 0) + 1
            
            # Quality distribution
            quality = product.quality
            quality_dist[quality] = quality_dist.get(quality, 0) + 1
        
        return {
            'total_products': total_products,
            'category_distribution': category_dist,
            'style_distribution': style_dist,
            'quality_distribution': quality_dist,
            'indexes_status': {
                'visual_index_ready': matcher.visual_index is not None,
                'text_index_ready': matcher.text_index is not None,
                'total_vectors': total_products
            },
            'timestamp': time.time()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get catalog info: {str(e)}"
        )

@router.get("/models/info", response_model=dict)
async def get_models_info(
    matcher: SceneProductMatcher = Depends(get_matcher_service),
    _: bool = Depends(validate_api_key)
):
    """
    Get information about the loaded models
    """
    
    try:
        return {
            'clip_model': 'ViT-B/32',
            'text_model': 'sentence-transformers/paraphrase-mpnet-base-v2',
            'llm_model': 'gpt-4o',
            'device': matcher.device,
            'faiss_indexes': {
                'visual_dimension': 512,
                'text_dimension': 768,
                'index_type': 'IndexFlatIP'
            },
            'capabilities': [
                'Scene analysis with GPT-4o vision',
                'Visual similarity search with CLIP',
                'Semantic similarity search',
                'LLM-powered re-ranking',
                'Multi-dimensional product categorization'
            ],
            'timestamp': time.time()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models info: {str(e)}"
        )