"""
Image processing utilities 
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from typing import List, Dict, Optional
import time

class ImageProcessor:
    """Image processing methods from original code"""
    
    def __init__(self, image_timeout: int = 3):
        self.image_timeout = image_timeout
    
    async def download_images_optimized(self, image_urls: List[str]) -> Dict[int, bytes]:
        """Ultimate image processing with enterprise optimization"""
        
        # Enterprise-grade connector
        connector = aiohttp.TCPConnector(
            limit=200,
            limit_per_host=50,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        async def download_with_retry(session, url, index, max_retries=2):
            for attempt in range(max_retries + 1):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.image_timeout)) as response:
                        if response.status == 200:
                            data = await response.read()
                            if len(data) > 500:  # Quality threshold
                                return index, data
                except:
                    if attempt < max_retries:
                        await asyncio.sleep(0.1 * (attempt + 1))
            return index, None
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i, url in enumerate(image_urls):
                if url and pd.notna(url):
                    tasks.append(download_with_retry(session, url, i))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            download_time = time.time() - start_time
            
            image_data = {}
            success_count = 0
            
            for result in results:
                if isinstance(result, tuple) and result[1] is not None:
                    image_data[result[0]] = result[1]
                    success_count += 1
            
            success_rate = success_count / len(tasks) if tasks else 0
            speed = success_count / download_time if download_time > 0 else 0
            
            return image_data
    
    def process_images_batch_optimized(self, image_data_batch: List[Optional[bytes]], 
                                     clip_model, clip_preprocess, device) -> np.ndarray:
        """Ultimate batch processing with maximum GPU efficiency"""
        
        valid_images = []
        valid_indices = []
        
        # Pre-process with quality validation
        for i, data in enumerate(image_data_batch):
            if data:
                try:
                    image = Image.open(BytesIO(data)).convert('RGB')
                    # Quality thresholds
                    if min(image.size) >= 64 and max(image.size) <= 2048:
                        processed = clip_preprocess(image)
                        valid_images.append(processed)
                        valid_indices.append(i)
                except:
                    continue
        
        # Initialize with enterprise defaults
        batch_features = [np.zeros(512, dtype='float32')] * len(image_data_batch)
        
        if valid_images:
            with torch.no_grad():
                # Enterprise batch processing
                batch_input = torch.stack(valid_images).to(device)
                batch_embeddings = clip_model.encode_image(batch_input)
                batch_embeddings = batch_embeddings.cpu().numpy()
                
                # Map to original positions
                for idx, valid_idx in enumerate(valid_indices):
                    batch_features[valid_idx] = batch_embeddings[idx]
        
        return np.array(batch_features, dtype='float32')