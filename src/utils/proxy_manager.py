"""
Proxy manager for rotating proxies to avoid rate limiting and detection.
Supports multiple proxy sources and automatic rotation strategies.
"""

import time
import random
from typing import List, Optional, Dict, Any
from collections import deque
import requests
from requests.exceptions import RequestException, Timeout

from utils.logger import get_logger

logger = get_logger(__name__)


class ProxyManager:
    """
    Manages proxy rotation for web scraping operations.
    Supports proxy validation, rotation strategies, and failure tracking.
    """
    
    def __init__(self, proxies: Optional[List[str]] = None, 
                 rotation_strategy: str = 'round_robin'):
        """
        Initialize proxy manager.
        
        Args:
            proxies: List of proxy URLs (format: http://host:port or http://user:pass@host:port)
            rotation_strategy: Rotation strategy ('round_robin', 'random', 'least_used')
        """
        self.proxies = proxies or []
        self.rotation_strategy = rotation_strategy
        self.proxy_stats = {proxy: {'uses': 0, 'failures': 0, 'last_used': 0} 
                           for proxy in self.proxies}
        self.proxy_queue = deque(self.proxies)
        self.current_proxy = None
        self.enabled = len(self.proxies) > 0
        
        if self.enabled:
            logger.info(f"ProxyManager initialized with {len(self.proxies)} proxies")
        else:
            logger.warning("ProxyManager initialized with no proxies - direct connections only")
    
    def get_proxy(self) -> Optional[str]:
        """
        Get next proxy based on rotation strategy.
        
        Returns:
            Proxy URL or None if no proxies available
        """
        if not self.enabled:
            return None
        
        if not self.proxies:
            logger.warning("No proxies available")
            return None
        
        if self.rotation_strategy == 'round_robin':
            proxy = self._round_robin()
        elif self.rotation_strategy == 'random':
            proxy = self._random_selection()
        elif self.rotation_strategy == 'least_used':
            proxy = self._least_used()
        else:
            proxy = self._round_robin()
        
        if proxy:
            self.current_proxy = proxy
            self.proxy_stats[proxy]['uses'] += 1
            self.proxy_stats[proxy]['last_used'] = time.time()
        
        return proxy
    
    def _round_robin(self) -> Optional[str]:
        """Round-robin proxy selection."""
        if not self.proxy_queue:
            self.proxy_queue = deque(self.proxies)
        
        proxy = self.proxy_queue.popleft()
        self.proxy_queue.append(proxy)
        return proxy
    
    def _random_selection(self) -> Optional[str]:
        """Random proxy selection."""
        return random.choice(self.proxies)
    
    def _least_used(self) -> Optional[str]:
        """Select least-used proxy."""
        # Sort by usage count, then by last used time
        sorted_proxies = sorted(
            self.proxies,
            key=lambda p: (self.proxy_stats[p]['uses'], self.proxy_stats[p]['last_used'])
        )
        return sorted_proxies[0] if sorted_proxies else None
    
    def report_failure(self, proxy: Optional[str] = None):
        """
        Report proxy failure.
        
        Args:
            proxy: Proxy that failed (uses current if None)
        """
        proxy = proxy or self.current_proxy
        
        if proxy and proxy in self.proxy_stats:
            self.proxy_stats[proxy]['failures'] += 1
            logger.warning(f"Proxy failure reported: {proxy} "
                          f"(total failures: {self.proxy_stats[proxy]['failures']})")
            
            # Remove proxy if too many failures
            if self.proxy_stats[proxy]['failures'] > 5:
                self.remove_proxy(proxy)
    
    def report_success(self, proxy: Optional[str] = None):
        """
        Report successful proxy use (resets failure count).
        
        Args:
            proxy: Proxy that succeeded (uses current if None)
        """
        proxy = proxy or self.current_proxy
        
        if proxy and proxy in self.proxy_stats:
            # Reset failures on success
            if self.proxy_stats[proxy]['failures'] > 0:
                self.proxy_stats[proxy]['failures'] = max(0, self.proxy_stats[proxy]['failures'] - 1)
    
    def add_proxy(self, proxy: str):
        """
        Add new proxy to pool.
        
        Args:
            proxy: Proxy URL to add
        """
        if proxy not in self.proxies:
            self.proxies.append(proxy)
            self.proxy_stats[proxy] = {'uses': 0, 'failures': 0, 'last_used': 0}
            self.proxy_queue.append(proxy)
            self.enabled = True
            logger.info(f"Added proxy: {proxy}")
    
    def remove_proxy(self, proxy: str):
        """
        Remove proxy from pool.
        
        Args:
            proxy: Proxy URL to remove
        """
        if proxy in self.proxies:
            self.proxies.remove(proxy)
            del self.proxy_stats[proxy]
            
            # Remove from queue if present
            if proxy in self.proxy_queue:
                self.proxy_queue.remove(proxy)
            
            logger.warning(f"Removed proxy: {proxy}")
            
            if not self.proxies:
                self.enabled = False
                logger.warning("No proxies remaining - switching to direct connections")
    
    def validate_proxy(self, proxy: str, test_url: str = 'http://httpbin.org/ip',
                       timeout: int = 10) -> bool:
        """
        Validate proxy by testing connection.
        
        Args:
            proxy: Proxy URL to test
            test_url: URL to test against
            timeout: Request timeout in seconds
            
        Returns:
            True if proxy is working
        """
        try:
            proxies = {
                'http': proxy,
                'https': proxy
            }
            
            response = requests.get(
                test_url,
                proxies=proxies,
                timeout=timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Proxy validated: {proxy}")
                return True
            else:
                logger.warning(f"Proxy validation failed: {proxy} (status: {response.status_code})")
                return False
                
        except (RequestException, Timeout) as e:
            logger.warning(f"Proxy validation failed: {proxy} ({e})")
            return False
    
    def validate_all_proxies(self, test_url: str = 'http://httpbin.org/ip',
                            timeout: int = 10) -> Dict[str, bool]:
        """
        Validate all proxies in pool.
        
        Args:
            test_url: URL to test against
            timeout: Request timeout
            
        Returns:
            Dictionary of proxy: is_valid
        """
        logger.info(f"Validating {len(self.proxies)} proxies...")
        
        results = {}
        invalid_proxies = []
        
        for proxy in self.proxies:
            is_valid = self.validate_proxy(proxy, test_url, timeout)
            results[proxy] = is_valid
            
            if not is_valid:
                invalid_proxies.append(proxy)
            
            # Small delay between tests
            time.sleep(0.5)
        
        # Remove invalid proxies
        for proxy in invalid_proxies:
            self.remove_proxy(proxy)
        
        valid_count = sum(1 for v in results.values() if v)
        logger.info(f"Proxy validation complete: {valid_count}/{len(results)} valid")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get proxy usage statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.enabled:
            return {
                'enabled': False,
                'total_proxies': 0
            }
        
        total_uses = sum(stats['uses'] for stats in self.proxy_stats.values())
        total_failures = sum(stats['failures'] for stats in self.proxy_stats.values())
        
        return {
            'enabled': True,
            'total_proxies': len(self.proxies),
            'rotation_strategy': self.rotation_strategy,
            'total_uses': total_uses,
            'total_failures': total_failures,
            'failure_rate': total_failures / max(total_uses, 1),
            'proxy_details': self.proxy_stats
        }
    
    def reset_stats(self):
        """Reset all proxy statistics."""
        for proxy in self.proxy_stats:
            self.proxy_stats[proxy] = {'uses': 0, 'failures': 0, 'last_used': 0}
        logger.info("Proxy statistics reset")
    
    @staticmethod
    def load_from_file(filepath: str) -> 'ProxyManager':
        """
        Load proxies from file (one per line).
        
        Args:
            filepath: Path to proxy list file
            
        Returns:
            ProxyManager instance
        """
        try:
            with open(filepath, 'r') as f:
                proxies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            logger.info(f"Loaded {len(proxies)} proxies from {filepath}")
            return ProxyManager(proxies)
            
        except Exception as e:
            logger.error(f"Failed to load proxies from {filepath}: {e}")
            return ProxyManager([])
    
    @staticmethod
    def load_from_api(api_url: str, api_key: Optional[str] = None) -> 'ProxyManager':
        """
        Load proxies from API endpoint.
        
        Args:
            api_url: API endpoint URL
            api_key: Optional API key
            
        Returns:
            ProxyManager instance
        """
        try:
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            response = requests.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle different API response formats
                if isinstance(data, list):
                    proxies = data
                elif isinstance(data, dict) and 'proxies' in data:
                    proxies = data['proxies']
                else:
                    proxies = []
                
                logger.info(f"Loaded {len(proxies)} proxies from API")
                return ProxyManager(proxies)
            else:
                logger.error(f"API returned status {response.status_code}")
                return ProxyManager([])
                
        except Exception as e:
            logger.error(f"Failed to load proxies from API: {e}")
            return ProxyManager([])
