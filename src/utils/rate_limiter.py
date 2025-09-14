# =====================================
# utils/rate_limiter.py
# =====================================
"""
Rate limiting functionality
"""

import time
from typing import Dict
from config import Config

class RateLimiter:
    """Manage rate limiting per site"""
    
    def __init__(self):
        self.last_request_times: Dict[str, float] = {}
        self.rate_limits = Config.RATE_LIMITS
    
    def wait_if_needed(self, site: str):
        """Wait if necessary to respect rate limits"""
        
        if site in self.last_request_times:
            elapsed = time.time() - self.last_request_times[site]
            required_delay = self.rate_limits.get(site, 5)
            
            if elapsed < required_delay:
                sleep_time = required_delay - elapsed
                print(f"[Rate Limit] Waiting {sleep_time:.1f}s for {site}")
                time.sleep(sleep_time)
        
        self.last_request_times[site] = time.time()
    
    def get_time_until_ready(self, site: str) -> float:
        """Get seconds until site is ready for next request"""
        
        if site not in self.last_request_times:
            return 0.0
        
        elapsed = time.time() - self.last_request_times[site]
        required_delay = self.rate_limits.get(site, 5)
        
        if elapsed < required_delay:
            return required_delay - elapsed
        
        return 0.0