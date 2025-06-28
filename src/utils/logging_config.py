"""
Logging configuration - extracted from original code
"""

import logging
import sys
import warnings

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ultimate_matcher.log')
        ]
    )
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    return logging.getLogger(__name__)