"""
TruePeopleSearch scraper for OSINT data collection.
"""

import time
import re
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException, Timeout

from .base_scraper import BaseScraper
from utils.logger import get_logger

logger = get_logger(__name__)

class TruePeopleSearchScraper(BaseScraper):
    """
    Scraper for TruePeopleSearch.com - free people search engine.
    """
    
    def __init__(self, proxy_manager=None, rate_limit: float = 2.0):
        """
        Initialize TruePeopleSearch scraper.
        
        Args:
            proxy_manager: Optional ProxyManager instance
            rate_limit: Minimum seconds between requests
        """
        super().__init__(
            source_name="truepeoplesearch",
            base_url="https://www.truepeoplesearch.com",
            rate_limit=rate_limit
        )
        self.proxy_manager = proxy_manager
        self.session = requests.Session()
        
        # Headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def search_person(self, first_name: str, last_name: str, 
                     location: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for a person by name and location.
        
        Args:
            first_name: First name
            last_name: Last name
            location: Optional location (City, State)
            
        Returns:
            Search results dictionary
        """
        # Build search URL
        search_url = f"{self.base_url}/results"
        params = {
            'name': f"{first_name} {last_name}",
            'citystatezip': location if location else ''
        }
        
        return self._execute_search(search_url, params)
    
    def search_phone(self, phone_number: str) -> Dict[str, Any]:
        """
        Reverse phone lookup.
        
        Args:
            phone_number: Phone number to search
            
        Returns:
            Search results
        """
        clean_phone = re.sub(r'\D', '', phone_number)
        search_url = f"{self.base_url}/results"
        params = {'phoneno': clean_phone}
        
        return self._execute_search(search_url, params)
    
    def search_address(self, address: str) -> Dict[str, Any]:
        """
        Reverse address lookup.
        
        Args:
            address: Full address string
            
        Returns:
            Search results
        """
        search_url = f"{self.base_url}/results"
        params = {'streetaddress': address}
        
        return self._execute_search(search_url, params)
    
    def _execute_search(self, url: str, params: Dict) -> Dict[str, Any]:
        """Execute search and parse results."""
        self._respect_rate_limit()
        
        try:
            proxies = None
            if self.proxy_manager:
                proxy = self.proxy_manager.get_proxy()
                proxies = {'http': proxy, 'https': proxy}
            
            logger.info(f"TruePeopleSearch search: {url}")
            response = self.session.get(
                url,
                params=params,
                proxies=proxies,
                timeout=15,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                return self._parse_response(response.text)
            elif response.status_code == 429:
                logger.warning("Rate limited by TruePeopleSearch")
                time.sleep(10)
                return self._create_empty_result("Rate limited")
            else:
                logger.error(f"TruePeopleSearch returned status {response.status_code}")
                return self._create_empty_result(f"HTTP {response.status_code}")
                
        except Timeout:
            logger.error("Request timed out")
            return self._create_empty_result("Timeout")
        except RequestException as e:
            logger.error(f"Request failed: {e}")
            return self._create_empty_result(str(e))
    
    def _parse_response(self, html: str) -> Dict[str, Any]:
        """Parse HTML response."""
        soup = BeautifulSoup(html, 'html.parser')
        
        results = {
            'source': self.source_name,
            'timestamp': time.time(),
            'records': []
        }
        
        # Find result cards
        cards = soup.find_all('div', class_='card')
        
        for card in cards[:10]:
            record = self._extract_record(card)
            if record:
                results['records'].append(record)
        
        results['entropy_score'] = self._calculate_entropy(results['records'])
        
        logger.info(f"Found {len(results['records'])} records")
        return results
    
    def _extract_record(self, card) -> Optional[Dict[str, Any]]:
        """Extract record from card element."""
        try:
            record = {}
            
            # Extract name
            name_elem = card.find('a', class_='link-to-details')
            if name_elem:
                record['name'] = name_elem.get_text(strip=True)
            
            # Extract age
            age_elem = card.find('div', class_='age')
            if age_elem:
                age_text = age_elem.get_text()
                age_match = re.search(r'\d+', age_text)
                if age_match:
                    record['age'] = int(age_match.group())
            
            # Extract location
            loc_elem = card.find('div', class_='content-location')
            if loc_elem:
                record['location'] = loc_elem.get_text(strip=True)
            
            # Extract phones
            phone_elems = card.find_all('div', class_='phone')
            if phone_elems:
                record['phones'] = [p.get_text(strip=True) for p in phone_elems]
            
            return record if record else None
            
        except Exception as e:
            logger.debug(f"Error extracting record: {e}")
            return None
    
    def _calculate_entropy(self, records: List[Dict]) -> float:
        """Calculate entropy for records."""
        if not records:
            return 0.0
        
        unique_values = set()
        total_fields = 0
        
        for record in records:
            for value in record.values():
                if isinstance(value, list):
                    unique_values.update(str(v) for v in value)
                    total_fields += len(value)
                else:
                    unique_values.add(str(value))
                    total_fields += 1
        
        if total_fields == 0:
            return 0.0
        
        return round(len(unique_values) / total_fields, 3)
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result with error."""
        return {
            'source': self.source_name,
            'timestamp': time.time(),
            'records': [],
            'error': reason,
            'entropy_score': 0.0
        }
