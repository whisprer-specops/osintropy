"""
WhitePages scraper for OSINT data collection with entropy analysis.
Handles name, phone, address, and reverse lookups with rate limiting and proxy support.
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


class WhitepagesScraper(BaseScraper):
    """
    Scraper for WhitePages.com - extracts person data including addresses,
    phone numbers, relatives, and associated records.
    """
    
    def __init__(self, proxy_manager=None, rate_limit: float = 2.0):
        """
        Initialize WhitePages scraper.
        
        Args:
            proxy_manager: Optional ProxyManager instance for rotation
            rate_limit: Minimum seconds between requests (default: 2.0)
        """
        super().__init__(
            source_name="whitepages",
            base_url="https://www.whitepages.com",
            rate_limit=rate_limit
        )
        self.proxy_manager = proxy_manager
        self.session = requests.Session()
        
        # Realistic headers to avoid detection
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
    def search_person(self, first_name: str, last_name: str, 
                     location: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for a person by name and optional location.
        
        Args:
            first_name: First name to search
            last_name: Last name to search
            location: Optional location (city, state format)
            
        Returns:
            Dictionary containing search results with entropy metadata
        """
        query_params = {
            'firstname': first_name,
            'lastname': last_name
        }
        
        if location:
            query_params['where'] = location
            
        search_url = f"{self.base_url}/name/{first_name}-{last_name}"
        if location:
            # Format: /name/John-Doe/Miami-FL
            location_slug = location.replace(' ', '-').replace(',', '')
            search_url = f"{search_url}/{location_slug}"
            
        return self._execute_search(search_url, query_params, 'person')
    
    def reverse_phone(self, phone_number: str) -> Dict[str, Any]:
        """
        Perform reverse phone lookup.
        
        Args:
            phone_number: Phone number to lookup (digits only or formatted)
            
        Returns:
            Dictionary containing phone owner information
        """
        # Clean phone number
        clean_phone = re.sub(r'\D', '', phone_number)
        
        if len(clean_phone) != 10:
            logger.warning(f"Invalid phone number format: {phone_number}")
            return self._create_empty_result(f"Invalid phone: {phone_number}")
        
        search_url = f"{self.base_url}/phone/1-{clean_phone}"
        return self._execute_search(search_url, {}, 'phone')
    
    def reverse_address(self, street: str, city: str, state: str) -> Dict[str, Any]:
        """
        Perform reverse address lookup.
        
        Args:
            street: Street address
            city: City name
            state: State abbreviation
            
        Returns:
            Dictionary containing address resident information
        """
        # Format address for URL
        address_slug = f"{street}/{city}/{state}".replace(' ', '-')
        search_url = f"{self.base_url}/address/{address_slug}"
        
        return self._execute_search(search_url, {}, 'address')
    
    def _execute_search(self, url: str, params: Dict, 
                       search_type: str) -> Dict[str, Any]:
        """
        Execute search request and parse results.
        
        Args:
            url: Search URL
            params: Query parameters
            search_type: Type of search (person, phone, address)
            
        Returns:
            Parsed results dictionary
        """
        self._respect_rate_limit()
        
        try:
            proxies = None
            if self.proxy_manager:
                proxy = self.proxy_manager.get_proxy()
                proxies = {'http': proxy, 'https': proxy}
            
            logger.info(f"WhitePages search: {url}")
            response = self.session.get(
                url,
                params=params,
                proxies=proxies,
                timeout=15,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                return self._parse_response(response.text, search_type)
            elif response.status_code == 429:
                logger.warning("Rate limited by WhitePages, backing off...")
                time.sleep(10)
                return self._create_empty_result("Rate limited")
            else:
                logger.error(f"WhitePages returned status {response.status_code}")
                return self._create_empty_result(f"HTTP {response.status_code}")
                
        except Timeout:
            logger.error("WhitePages request timed out")
            return self._create_empty_result("Timeout")
        except RequestException as e:
            logger.error(f"WhitePages request failed: {e}")
            return self._create_empty_result(str(e))
    
    def _parse_response(self, html: str, search_type: str) -> Dict[str, Any]:
        """
        Parse HTML response and extract data.
        
        Args:
            html: HTML content
            search_type: Type of search performed
            
        Returns:
            Extracted data dictionary
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        results = {
            'source': self.source_name,
            'search_type': search_type,
            'timestamp': time.time(),
            'records': []
        }
        
        # Find result cards - WhitePages uses various div classes
        result_cards = soup.find_all('div', class_=re.compile(r'result|card|person'))
        
        if not result_cards:
            # Try alternative selectors
            result_cards = soup.find_all('div', attrs={'data-person': True})
        
        for card in result_cards[:10]:  # Limit to first 10 results
            record = self._extract_record(card, search_type)
            if record:
                results['records'].append(record)
        
        # Calculate entropy for the dataset
        results['entropy_score'] = self._calculate_result_entropy(results['records'])
        
        logger.info(f"WhitePages found {len(results['records'])} records")
        return results
    
    def _extract_record(self, card, search_type: str) -> Optional[Dict[str, Any]]:
        """
        Extract individual record from result card.
        
        Args:
            card: BeautifulSoup element containing record
            search_type: Type of search
            
        Returns:
            Record dictionary or None
        """
        try:
            record = {}
            
            # Extract name
            name_elem = card.find(['h3', 'h2', 'div'], class_=re.compile(r'name'))
            if name_elem:
                record['name'] = name_elem.get_text(strip=True)
            
            # Extract age
            age_elem = card.find(text=re.compile(r'Age:?\s*\d+'))
            if age_elem:
                age_match = re.search(r'\d+', age_elem)
                if age_match:
                    record['age'] = int(age_match.group())
            
            # Extract addresses
            address_elems = card.find_all(['div', 'span'], class_=re.compile(r'address|location'))
            addresses = []
            for addr in address_elems:
                addr_text = addr.get_text(strip=True)
                if addr_text and len(addr_text) > 5:
                    addresses.append(addr_text)
            if addresses:
                record['addresses'] = addresses
            
            # Extract phone numbers
            phone_elems = card.find_all('a', href=re.compile(r'tel:'))
            phones = []
            for phone in phone_elems:
                phone_text = phone.get_text(strip=True)
                if phone_text:
                    phones.append(phone_text)
            if phones:
                record['phones'] = phones
            
            # Extract relatives/associates
            relatives_section = card.find(text=re.compile(r'Relatives|Associates'))
            if relatives_section:
                relatives_elem = relatives_section.find_parent().find_next_sibling()
                if relatives_elem:
                    relatives = [r.strip() for r in relatives_elem.get_text().split(',')]
                    record['relatives'] = relatives
            
            # Extract email if available
            email_elem = card.find('a', href=re.compile(r'mailto:'))
            if email_elem:
                record['email'] = email_elem.get('href').replace('mailto:', '')
            
            # Only return if we extracted meaningful data
            if len(record) > 1:
                return record
                
        except Exception as e:
            logger.debug(f"Error parsing WhitePages record: {e}")
        
        return None
    
    def _calculate_result_entropy(self, records: List[Dict]) -> float:
        """
        Calculate information entropy for result set.
        
        Args:
            records: List of extracted records
            
        Returns:
            Entropy score (0.0 - 1.0)
        """
        if not records:
            return 0.0
        
        # Count unique values across fields
        field_diversity = {}
        total_fields = 0
        
        for record in records:
            for key, value in record.items():
                if key not in field_diversity:
                    field_diversity[key] = set()
                
                if isinstance(value, list):
                    field_diversity[key].update(value)
                else:
                    field_diversity[key].add(str(value))
                
                total_fields += 1
        
        # Calculate entropy based on uniqueness
        if total_fields == 0:
            return 0.0
        
        unique_values = sum(len(values) for values in field_diversity.values())
        entropy = min(unique_values / max(total_fields, 1), 1.0)
        
        return round(entropy, 3)
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result structure with error reason."""
        return {
            'source': self.source_name,
            'timestamp': time.time(),
            'records': [],
            'error': reason,
            'entropy_score': 0.0
        }
