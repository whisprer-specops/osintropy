"""
Base scraper class providing common functionality for all OSINT scrapers.
"""

import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseScraper(ABC):
    """
    Abstract base class for all OSINT scrapers.
    Provides rate limiting, common utilities, and interface contract.
    """
    
    def __init__(self, source_name: str, base_url: str, rate_limit: float = 1.0):
        """
        Initialize base scraper.
        
        Args:
            source_name: Name of the data source
            base_url: Base URL for the source
            rate_limit: Minimum seconds between requests
        """
        self.source_name = source_name
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.last_request_time = 0
    
    def _respect_rate_limit(self):
        """
        Enforce rate limiting between requests.
        Sleeps if necessary to maintain minimum time between requests.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @abstractmethod
    def search_person(self, first_name: str, last_name: str, 
                     location: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for a person by name and optional location.
        
        Args:
            first_name: First name to search
            last_name: Last name to search
            location: Optional location string
            
        Returns:
            Dictionary containing search results
        """
        pass
    
    def search(self, first_name: str, last_name: str, 
               location: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generic search method that delegates to search_person.
        This method is called by the aggregator for compatibility.
        
        Args:
            first_name: First name to search
            last_name: Last name to search
            location: Optional location string
            **kwargs: Additional keyword arguments (ignored by default)
            
        Returns:
            Dictionary containing search results
        """
        return self.search_person(first_name, last_name, location)
    
    def get_source_name(self) -> str:
        """Get the source name."""
        return self.source_name
    
    def get_base_url(self) -> str:
        """Get the base URL."""
        return self.base_url
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(source='{self.source_name}')"
