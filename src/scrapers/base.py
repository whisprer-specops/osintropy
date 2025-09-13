# =====================================
# scrapers/base.py - Base Scraper Class
# =====================================
"""
Base scraper class with common functionality
"""

import time
import random
import requests
from typing import Dict, List, Optional
from fake_useragent import UserAgent
from selenium import webdriver
import undetected_chromedriver as uc

ua = UserAgent()

class BaseScraper:
    """Base class for all people search scrapers"""
    
    def __init__(self, use_selenium: bool = False):
        self.use_selenium = use_selenium
        self.session = requests.Session()
        self.driver = None
        self.request_count = 0
        self.last_request_time = 0
        
        if use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Setup undetected Chrome driver"""
        options = uc.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument(f'user-agent={ua.random}')
        # Uncomment for headless mode
        # options.add_argument('--headless')
        
        self.driver = uc.Chrome(options=options)
    
    def _get_headers(self) -> Dict:
        """Get randomized headers"""
        return {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def _add_delay(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """Add random delay with entropy"""
        delay = random.uniform(min_sec, max_sec)
        # Add micro-jitter for more entropy
        delay += random.random() * 0.1
        time.sleep(delay)
    
    def search(self, **kwargs) -> List[Dict]:
        """Search method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement search()")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
        if self.session:
            self.session.close()
