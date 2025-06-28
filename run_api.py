#!/usr/bin/env python3
"""
API Runner Script
Start the Ultimate Scene Matcher API with proper configuration
"""

import sys
import os
import logging
import uvicorn
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required files and directories exist"""
    
    required_paths = [
        "src",
        "data",
        "cache"
    ]
    
    missing = []
    for path in required_paths:
        if not (project_root / path).exists():
            missing.append(path)
    
    if missing:
        logger.error(f"Missing required paths: {missing}")
        logger.info("Please ensure your project structure is correct:")
        logger.info("  src/  - Scene matcher source code")
        logger.info("  data/               - Data directory (for catalog)")
        logger.info("  cache/              - Cache directory (will be created)")
        return False
    
    return True

def setup_environment():
    """Setup environment variables and directories"""
    
    # Create cache directory if it doesn't exist
    cache_dir = project_root / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Set environment variables if not already set
    env_vars = {
        "SCENE_MATCHER_CACHE_DIR": str(cache_dir),
        "SCENE_MATCHER_CATALOG_PATH": str(project_root / "data" / "product-catalog.csv"),
        "SCENE_MATCHER_HOST": "0.0.0.0",
        "SCENE_MATCHER_PORT": "8000",
        "SCENE_MATCHER_DEBUG": "false"
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

def main():
    """Main function to start the API"""
    
    logger.info("🚀 Starting Ultimate Scene Matcher API...")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Get configuration
    host = os.environ.get("SCENE_MATCHER_HOST", "0.0.0.0")
    port = int(os.environ.get("SCENE_MATCHER_PORT", "8000"))
    debug = os.environ.get("SCENE_MATCHER_DEBUG", "false").lower() == "true"
    
    logger.info(f"🌐 Server will start on http://{host}:{port}")
    logger.info(f"📚 API docs will be available at http://{host}:{port}/docs")
    logger.info(f"🔍 Debug mode: {debug}")
    
    # Start the server
    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info" if not debug else "debug",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down API server...")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()