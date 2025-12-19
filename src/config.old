# =====================================
# config.py - Configuration Settings
# =====================================
"""
Configuration for OSINT Aggregator
"""

import os
from typing import Dict, List, Tuple

class Config:
    """Central configuration for the project"""
    
    # Database settings
    DATABASE_PATH = os.environ.get('OSINT_DB_PATH', 'people_search.db')
    
    # Rate limiting (seconds between requests per site)
    RATE_LIMITS = {
        'truepeoplesearch': 3,
        'whitepages': 5,
        'spokeo': 7,
        'beenverified': 5,
        'intelius': 6
    }
    
    # Request delays (min, max in seconds)
    REQUEST_DELAY_RANGE = (2.0, 5.0)
    BULK_SEARCH_DELAY_RANGE = (5.0, 15.0)
    
    # Scraper settings
    USE_SELENIUM = {
        'truepeoplesearch': False,
        'whitepages': True,
        'spokeo': False,
        'beenverified': False
    }
    
    # Matching thresholds
    MATCH_THRESHOLD = 0.85
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    
    # Entropy thresholds
    NAME_ENTROPY_SUSPICIOUS = 2.0
    TEMPORAL_ENTROPY_SUSPICIOUS = 0.1
    
    # Proxy settings (if using)
    USE_PROXIES = False
    PROXY_LIST = []
    
    # User agent rotation
    ROTATE_USER_AGENTS = True
    
    # Maximum results per search
    MAX_RESULTS_PER_SITE = 10
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'osint_aggregator.log'
