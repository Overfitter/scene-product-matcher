"""
Optimized Image Processing Utilities for Ultimate Matcher
"""

import asyncio
import aiohttp
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from typing import List, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """High-performance image processing for the Ultimate Matcher"""
    
    def __init__(self, timeout: int = 5, max_concurrent: int = 50):
        self.timeout = timeout
        self.max_concurrent = max_concurrent
    
    async def download_images_optimized(self, image_urls: List[str]) -> Dict[int, bytes]:
        """Download images efficiently with high concurrency"""
        
        if not image_urls:
            return {}
        
        # Filter valid URLs
        valid_urls = []
        url_indices = []
        for i, url in enumerate(image_urls):
            if url and isinstance(url, str) and url.startswith(('http://', 'https://')):
                valid_urls.append(url)
                url_indices.append(i)
        
        if not valid_urls:
            return {}
        
        # Configure session for high performance
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=20,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async def download_single(session, url, index):
            """Download single image with retries"""
            for attempt in range(2):
                try:
                    async with session.get(url, timeout=timeout) as response:
                        if response.status == 200:
                            data = await response.read()
                            if len(data) > 100:  # Basic validation
                                return index, data
                except Exception as e:
                    if attempt == 0:
                        await asyncio.sleep(0.1)
                    logger.debug(f"Failed to download {url}: {e}")
            return index, None
        
        # Download all images concurrently
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                download_single(session, url, url_indices[i])
                for i, url in enumerate(valid_urls)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        image_data = {}
        success_count = 0
        
        for result in results:
            if isinstance(result, tuple) and result[1] is not None:
                index, data = result
                image_data[index] = data
                success_count += 1
        
        logger.info(f"📥 Downloaded {success_count}/{len(valid_urls)} images successfully")
        return image_data
    
    def process_images_batch_optimized(self, 
                                     image_data_batch: List[Optional[bytes]], 
                                     clip_model, 
                                     clip_preprocess, 
                                     device: str) -> np.ndarray:
        """Process image batch with CLIP model efficiently"""
        
        batch_size = len(image_data_batch)
        feature_dim = 512  # CLIP ViT-B/32 feature dimension
        
        # Initialize output array
        batch_features = np.zeros((batch_size, feature_dim), dtype='float32')
        
        # Process valid images
        valid_images = []
        valid_indices = []
        
        for i, image_data in enumerate(image_data_batch):
            if image_data:
                try:
                    # Load and validate image
                    image = Image.open(BytesIO(image_data)).convert('RGB')
                    
                    # Size validation
                    if min(image.size) >= 32 and max(image.size) <= 4096:
                        # Preprocess
                        processed = clip_preprocess(image)
                        valid_images.append(processed)
                        valid_indices.append(i)
                        
                except Exception as e:
                    logger.debug(f"Failed to process image {i}: {e}")
                    continue
        
        # Process valid images with CLIP
        if valid_images:
            try:
                with torch.no_grad():
                    # Stack images into batch tensor
                    batch_tensor = torch.stack(valid_images).to(device)
                    
                    # Get features
                    features = clip_model.encode_image(batch_tensor)
                    features_np = features.cpu().numpy()
                    
                    # Assign to output array
                    for i, valid_idx in enumerate(valid_indices):
                        batch_features[valid_idx] = features_np[i]
                        
            except Exception as e:
                logger.error(f"CLIP processing failed: {e}")
        
        return batch_features
    
    def preprocess_single_image(self, image: Image.Image, clip_preprocess) -> torch.Tensor:
        """Preprocess single image for CLIP"""
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Size optimization
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return clip_preprocess(image)