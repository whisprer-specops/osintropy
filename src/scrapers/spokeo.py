"""
Spokeo scraper for comprehensive people search with enhanced data points.
Supports name, phone, email, and username searches with entropy scoring.
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


class SpokeoScraper(BaseScraper):
    """
    Scraper for Spokeo.com - extracts comprehensive person data including
    social media profiles, criminal records, and employment history.
    """
    
    def __init__(self, proxy_manager=None, rate_limit: float = 2.5):
        """
        Initialize Spokeo scraper.
        
        Args:
            proxy_manager: Optional ProxyManager instance
            rate_limit: Minimum seconds between requests (default: 2.5)
        """
        super().__init__(
            source_name="spokeo",
            base_url="https://www.spokeo.com",
            rate_limit=rate_limit
        )
        self.proxy_manager = proxy_manager
        self.session = requests.Session()
        
        # Enhanced headers for Spokeo
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Referer': 'https://www.google.com/'
        })
    
    def search_person(self, first_name: str, last_name: str,
                     location: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for person by name and location.
        
        Args:
            first_name: First name
            last_name: Last name
            location: Optional location (City, State)
            
        Returns:
            Search results with entropy analysis
        """
        # Spokeo name search URL format
        search_url = f"{self.base_url}/{first_name}-{last_name}"
        
        if location:
            # Add location to search
            location_formatted = location.replace(' ', '-').replace(',', '')
            search_url = f"{search_url}/{location_formatted}"
        
        return self._execute_search(search_url, 'person')
    
    def search_phone(self, phone_number: str) -> Dict[str, Any]:
        """
        Reverse phone lookup.
        
        Args:
            phone_number: Phone number to search
            
        Returns:
            Phone owner information
        """
        clean_phone = re.sub(r'\D', '', phone_number)
        
        if len(clean_phone) != 10:
            logger.warning(f"Invalid phone number: {phone_number}")
            return self._create_empty_result(f"Invalid phone: {phone_number}")
        
        # Format: (XXX) XXX-XXXX
        formatted = f"({clean_phone[:3]}) {clean_phone[3:6]}-{clean_phone[6:]}"
        search_url = f"{self.base_url}/phone/{formatted}"
        
        return self._execute_search(search_url, 'phone')
    
    def search_email(self, email: str) -> Dict[str, Any]:
        """
        Search by email address.
        
        Args:
            email: Email address to search
            
        Returns:
            Associated person information
        """
        if '@' not in email:
            logger.warning(f"Invalid email format: {email}")
            return self._create_empty_result(f"Invalid email: {email}")
        
        search_url = f"{self.base_url}/email/{email}"
        return self._execute_search(search_url, 'email')
    
    def search_username(self, username: str) -> Dict[str, Any]:
        """
        Search by username across social platforms.
        
        Args:
            username: Username to search
            
        Returns:
            Social media profiles and associated data
        """
        search_url = f"{self.base_url}/username/{username}"
        return self._execute_search(search_url, 'username')
    
    def search_address(self, street: str, city: str, state: str) -> Dict[str, Any]:
        """
        Reverse address lookup.
        
        Args:
            street: Street address
            city: City
            state: State abbreviation
            
        Returns:
            Resident information
        """
        address_formatted = f"{street}-{city}-{state}".replace(' ', '-')
        search_url = f"{self.base_url}/address/{address_formatted}"
        
        return self._execute_search(search_url, 'address')
    
    def _execute_search(self, url: str, search_type: str) -> Dict[str, Any]:
        """
        Execute search request and parse results.
        
        Args:
            url: Search URL
            search_type: Type of search
            
        Returns:
            Parsed results
        """
        self._respect_rate_limit()
        
        try:
            proxies = None
            if self.proxy_manager:
                proxy = self.proxy_manager.get_proxy()
                proxies = {'http': proxy, 'https': proxy}
            
            logger.info(f"Spokeo search: {url}")
            response = self.session.get(
                url,
                proxies=proxies,
                timeout=20,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                return self._parse_response(response.text, search_type, url)
            elif response.status_code == 429:
                logger.warning("Rate limited by Spokeo")
                time.sleep(15)
                return self._create_empty_result("Rate limited")
            elif response.status_code == 404:
                logger.info("No results found on Spokeo")
                return self._create_empty_result("Not found")
            else:
                logger.error(f"Spokeo returned status {response.status_code}")
                return self._create_empty_result(f"HTTP {response.status_code}")
                
        except Timeout:
            logger.error("Spokeo request timed out")
            return self._create_empty_result("Timeout")
        except RequestException as e:
            logger.error(f"Spokeo request failed: {e}")
            return self._create_empty_result(str(e))
    
    def _parse_response(self, html: str, search_type: str, url: str) -> Dict[str, Any]:
        """
        Parse Spokeo HTML response.
        
        Args:
            html: HTML content
            search_type: Type of search
            url: Original search URL
            
        Returns:
            Extracted data
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        results = {
            'source': self.source_name,
            'search_type': search_type,
            'timestamp': time.time(),
            'search_url': url,
            'records': []
        }
        
        # Check if paywall/login required
        if soup.find(text=re.compile(r'Sign Up|Create Account|Subscribe')):
            logger.warning("Spokeo requires authentication for full results")
            results['limited'] = True
        
        # Find result containers - Spokeo uses card-based layout
        result_cards = soup.find_all('div', class_=re.compile(r'card|result|person-card'))
        
        # Alternative: look for data attributes
        if not result_cards:
            result_cards = soup.find_all('div', attrs={'data-person-id': True})
        
        # Extract teaser data (available without login)
        teaser_data = self._extract_teaser_data(soup, search_type)
        if teaser_data:
            results['records'].append(teaser_data)
        
        # Parse individual cards
        for card in result_cards[:15]:
            record = self._extract_record(card, search_type)
            if record and record not in results['records']:
                results['records'].append(record)
        
        # Calculate entropy
        results['entropy_score'] = self._calculate_entropy(results['records'])
        
        logger.info(f"Spokeo extracted {len(results['records'])} records")
        return results
    
    def _extract_teaser_data(self, soup, search_type: str) -> Optional[Dict[str, Any]]:
        """
        Extract preview/teaser data visible without subscription.
        
        Args:
            soup: BeautifulSoup object
            search_type: Type of search
            
        Returns:
            Teaser record or None
        """
        try:
            teaser = {}
            
            # Name extraction
            name_elem = soup.find(['h1', 'h2'], class_=re.compile(r'name|title'))
            if name_elem:
                teaser['name'] = name_elem.get_text(strip=True)
            
            # Age/DOB
            age_elem = soup.find(text=re.compile(r'Age:?\s*\d+|Born:?'))
            if age_elem:
                age_match = re.search(r'\d+', age_elem)
                if age_match:
                    teaser['age'] = int(age_match.group())
            
            # Locations lived
            location_section = soup.find(text=re.compile(r'Lives in|Current Address|Locations?'))
            if location_section:
                loc_parent = location_section.find_parent()
                if loc_parent:
                    locations = []
                    for loc in loc_parent.find_all(['li', 'div', 'span']):
                        loc_text = loc.get_text(strip=True)
                        if loc_text and len(loc_text) > 3:
                            locations.append(loc_text)
                    if locations:
                        teaser['locations'] = locations[:5]
            
            # Phone numbers
            phone_section = soup.find(text=re.compile(r'Phone|Contact'))
            if phone_section:
                phone_parent = phone_section.find_parent()
                if phone_parent:
                    phones = []
                    for phone in phone_parent.find_all(text=re.compile(r'\d{3}[-.]?\d{3}[-.]?\d{4}')):
                        phones.append(phone.strip())
                    if phones:
                        teaser['phones'] = phones
            
            # Email addresses
            emails = soup.find_all('a', href=re.compile(r'mailto:'))
            if emails:
                teaser['emails'] = [e.get('href').replace('mailto:', '') for e in emails[:3]]
            
            # Social media
            social_section = soup.find(text=re.compile(r'Social|Profiles|Networks'))
            if social_section:
                social_parent = social_section.find_parent()
                if social_parent:
                    platforms = []
                    for link in social_parent.find_all('a', href=True):
                        href = link.get('href', '')
                        if any(platform in href.lower() for platform in 
                              ['facebook', 'twitter', 'linkedin', 'instagram', 'tiktok']):
                            platforms.append({
                                'platform': self._identify_platform(href),
                                'url': href
                            })
                    if platforms:
                        teaser['social_media'] = platforms
            
            # Relatives
            relatives_section = soup.find(text=re.compile(r'Relatives|Associates|Family'))
            if relatives_section:
                rel_parent = relatives_section.find_parent()
                if rel_parent:
                    relatives = []
                    for rel in rel_parent.find_all(['li', 'a']):
                        rel_text = rel.get_text(strip=True)
                        if rel_text and len(rel_text) > 2:
                            relatives.append(rel_text)
                    if relatives:
                        teaser['relatives'] = relatives[:10]
            
            # Work/Education
            work_section = soup.find(text=re.compile(r'Work|Education|Employment'))
            if work_section:
                work_parent = work_section.find_parent()
                if work_parent:
                    work_info = work_parent.get_text(strip=True)
                    if work_info:
                        teaser['employment'] = work_info
            
            return teaser if len(teaser) > 1 else None
            
        except Exception as e:
            logger.debug(f"Error extracting Spokeo teaser: {e}")
            return None
    
    def _extract_record(self, card, search_type: str) -> Optional[Dict[str, Any]]:
        """
        Extract individual record from result card.
        
        Args:
            card: BeautifulSoup element
            search_type: Search type
            
        Returns:
            Record dictionary or None
        """
        try:
            record = {}
            
            # Extract all text content
            name = card.find(['h3', 'h4', 'a'], class_=re.compile(r'name|title'))
            if name:
                record['name'] = name.get_text(strip=True)
            
            # Age
            age_text = card.find(text=re.compile(r'Age:?\s*\d+'))
            if age_text:
                age_match = re.search(r'\d+', age_text)
                if age_match:
                    record['age'] = int(age_match.group())
            
            # Location
            location = card.find(['div', 'span'], class_=re.compile(r'location|address|city'))
            if location:
                record['location'] = location.get_text(strip=True)
            
            # Phone
            phone = card.find(text=re.compile(r'\(\d{3}\)\s*\d{3}-\d{4}'))
            if phone:
                record['phone'] = phone.strip()
            
            return record if len(record) > 0 else None
            
        except Exception as e:
            logger.debug(f"Error parsing Spokeo record: {e}")
            return None
    
    def _identify_platform(self, url: str) -> str:
        """Identify social media platform from URL."""
        url_lower = url.lower()
        if 'facebook' in url_lower:
            return 'Facebook'
        elif 'twitter' in url_lower or 'x.com' in url_lower:
            return 'Twitter/X'
        elif 'linkedin' in url_lower:
            return 'LinkedIn'
        elif 'instagram' in url_lower:
            return 'Instagram'
        elif 'tiktok' in url_lower:
            return 'TikTok'
        elif 'youtube' in url_lower:
            return 'YouTube'
        else:
            return 'Unknown'
    
    def _calculate_entropy(self, records: List[Dict]) -> float:
        """
        Calculate information entropy for Spokeo results.
        
        Args:
            records: List of records
            
        Returns:
            Entropy score (0.0-1.0)
        """
        if not records:
            return 0.0
        
        # Count unique data points
        all_values = set()
        total_fields = 0
        
        for record in records:
            for key, value in record.items():
                if isinstance(value, list):
                    all_values.update(str(v) for v in value)
                    total_fields += len(value)
                elif isinstance(value, dict):
                    all_values.update(str(v) for v in value.values())
                    total_fields += len(value)
                else:
                    all_values.add(str(value))
                    total_fields += 1
        
        if total_fields == 0:
            return 0.0
        
        # Entropy based on uniqueness ratio
        entropy = min(len(all_values) / total_fields, 1.0)
        return round(entropy, 3)
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result with error reason."""
        return {
            'source': self.source_name,
            'timestamp': time.time(),
            'records': [],
            'error': reason,
            'entropy_score': 0.0
        }
