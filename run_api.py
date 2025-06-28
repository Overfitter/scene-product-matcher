#!/usr/bin/env python3
"""
Clean startup script for Ultimate Scene Matcher API
Suppresses FAISS warnings and provides clean output
"""

import os
import sys
import logging
from pathlib import Path

# Suppress FAISS GPU warnings before any imports
os.environ['FAISS_ENABLE_GPU'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices from FAISS

def setup_environment():
    """Setup environment and paths"""
    
    # Add project root to Python path
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    # Set default environment variables
    os.environ.setdefault('CATALOG_PATH', str(project_root / 'data' / 'product-catalog.csv'))
    os.environ.setdefault('CACHE_DIR', str(project_root / 'cache'))
    os.environ.setdefault('API_HOST', '0.0.0.0')
    os.environ.setdefault('API_PORT', '8000')
    os.environ.setdefault('DEBUG', 'false')

def check_requirements():
    """Check if all requirements are met"""
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("💡 Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # Check catalog file
    catalog_path = os.getenv('CATALOG_PATH')
    if not Path(catalog_path).exists():
        print(f"❌ Error: Catalog file not found: {catalog_path}")
        print("💡 Make sure the product catalog CSV file exists")
        return False
    
    # Test critical imports
    try:
        import torch
        import clip
        import faiss
        import openai
        import fastapi
        import uvicorn
        print("✅ All required packages available")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Install with: pip install -r requirements-cpu.txt")
        return False

def start_api():
    """Start the API with clean output"""
    
    print("🚀 Ultimate Scene Matcher API")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Configure logging to suppress unwanted messages
    logging.getLogger('faiss.loader').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)
    
    # API configuration
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print(f"🌐 Starting server on http://{host}:{port}")
    print(f"📚 API docs available at http://{host}:{port}/docs")
    print(f"❤️  Health check at http://{host}:{port}/health")
    print(f"🔧 Mode: {'Debug' if debug else 'Production'}")
    print("=" * 50)
    
    try:
        import uvicorn
        
        # Configure uvicorn logging
        log_config = uvicorn.config.LOGGING_CONFIG
        log_config["loggers"]["uvicorn"]["level"] = "INFO"
        log_config["loggers"]["uvicorn.access"]["level"] = "INFO"
        
        # Start the server
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=debug,
            workers=1,
            log_level="info",
            access_log=True,
            log_config=log_config
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_api()