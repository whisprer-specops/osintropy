"""
BeenVerified scraper for background check and people search data.
Provides criminal records, property ownership, and comprehensive contact info.
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


class BeenVerifiedScraper(BaseScraper):
    """
    Scraper for BeenVerified.com - extracts background check data,
    criminal records, property records, and contact information.
    """
    
    def __init__(self, proxy_manager=None, rate_limit: float = 3.0):
        """
        Initialize BeenVerified scraper.
        
        Args:
            proxy_manager: Optional ProxyManager instance
            rate_limit: Minimum seconds between requests (default: 3.0)
        """
        super().__init__(
            source_name="beenverified",
            base_url="https://www.beenverified.com",
            rate_limit=rate_limit
        )
        self.proxy_manager = proxy_manager
        self.session = requests.Session()
        
        # Headers optimized for BeenVerified
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'TE': 'trailers'
        })
    
    def search_person(self, first_name: str, last_name: str,
                     city: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for person by name and location.
        
        Args:
            first_name: First name
            last_name: Last name
            city: Optional city
            state: Optional state
            
        Returns:
            Search results with background data
        """
        # BeenVerified search format
        search_url = f"{self.base_url}/people/{first_name}-{last_name}"
        
        if city and state:
            search_url = f"{search_url}/{city}-{state}"
        elif state:
            search_url = f"{search_url}/{state}"
        
        return self._execute_search(search_url, 'person')
    
    def reverse_phone(self, phone_number: str) -> Dict[str, Any]:
        """
        Reverse phone lookup.
        
        Args:
            phone_number: Phone number to lookup
            
        Returns:
            Phone owner and associated data
        """
        clean_phone = re.sub(r'\D', '', phone_number)
        
        if len(clean_phone) != 10:
            logger.warning(f"Invalid phone number: {phone_number}")
            return self._create_empty_result(f"Invalid phone: {phone_number}")
        
        search_url = f"{self.base_url}/phone-lookup/{clean_phone}"
        return self._execute_search(search_url, 'phone')
    
    def reverse_email(self, email: str) -> Dict[str, Any]:
        """
        Email reverse lookup.
        
        Args:
            email: Email address
            
        Returns:
            Email owner information
        """
        if '@' not in email:
            logger.warning(f"Invalid email: {email}")
            return self._create_empty_result(f"Invalid email: {email}")
        
        search_url = f"{self.base_url}/reverse-email/{email}"
        return self._execute_search(search_url, 'email')
    
    def reverse_address(self, street: str, city: str, state: str,
                       zip_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Reverse address lookup.
        
        Args:
            street: Street address
            city: City
            state: State
            zip_code: Optional ZIP code
            
        Returns:
            Property and resident data
        """
        address_parts = [street, city, state]
        if zip_code:
            address_parts.append(zip_code)
        
        address_slug = '-'.join(address_parts).replace(' ', '-')
        search_url = f"{self.base_url}/address/{address_slug}"
        
        return self._execute_search(search_url, 'address')
    
    def search_property(self, street: str, city: str, state: str) -> Dict[str, Any]:
        """
        Search property records.
        
        Args:
            street: Street address
            city: City
            state: State
            
        Returns:
            Property ownership and history
        """
        address_slug = f"{street}-{city}-{state}".replace(' ', '-')
        search_url = f"{self.base_url}/property/{address_slug}"
        
        return self._execute_search(search_url, 'property')
    
    def _execute_search(self, url: str, search_type: str) -> Dict[str, Any]:
        """
        Execute search and parse results.
        
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
            
            logger.info(f"BeenVerified search: {url}")
            response = self.session.get(
                url,
                proxies=proxies,
                timeout=20,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                return self._parse_response(response.text, search_type, url)
            elif response.status_code == 429:
                logger.warning("Rate limited by BeenVerified")
                time.sleep(20)
                return self._create_empty_result("Rate limited")
            elif response.status_code == 404:
                logger.info("No results on BeenVerified")
                return self._create_empty_result("Not found")
            else:
                logger.error(f"BeenVerified returned status {response.status_code}")
                return self._create_empty_result(f"HTTP {response.status_code}")
                
        except Timeout:
            logger.error("BeenVerified request timed out")
            return self._create_empty_result("Timeout")
        except RequestException as e:
            logger.error(f"BeenVerified request failed: {e}")
            return self._create_empty_result(str(e))
    
    def _parse_response(self, html: str, search_type: str, url: str) -> Dict[str, Any]:
        """
        Parse BeenVerified HTML response.
        
        Args:
            html: HTML content
            search_type: Search type
            url: Original URL
            
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
        
        # Check for paywall
        if soup.find(text=re.compile(r'Start Your Search|Sign Up|Get Full Report')):
            logger.warning("BeenVerified requires subscription for full access")
            results['paywall'] = True
        
        # Extract preview/teaser data
        preview_data = self._extract_preview_data(soup, search_type)
        if preview_data:
            results['records'].append(preview_data)
        
        # Find result listings
        result_items = soup.find_all('div', class_=re.compile(r'result|record|person'))
        
        for item in result_items[:10]:
            record = self._extract_record(item, search_type)
            if record and record not in results['records']:
                results['records'].append(record)
        
        # Calculate entropy
        results['entropy_score'] = self._calculate_data_entropy(results['records'])
        
        logger.info(f"BeenVerified extracted {len(results['records'])} records")
        return results
    
    def _extract_preview_data(self, soup, search_type: str) -> Optional[Dict[str, Any]]:
        """
        Extract preview data visible without subscription.
        
        Args:
            soup: BeautifulSoup object
            search_type: Search type
            
        Returns:
            Preview record or None
        """
        try:
            preview = {}
            
            # Name
            name_elem = soup.find(['h1', 'h2'], class_=re.compile(r'name|person|title'))
            if name_elem:
                preview['name'] = name_elem.get_text(strip=True)
            
            # Age
            age_elem = soup.find(text=re.compile(r'Age:?\s*\d+'))
            if age_elem:
                age_match = re.search(r'\d+', age_elem)
                if age_match:
                    preview['age'] = int(age_match.group())
            
            # Current address/location
            location_elem = soup.find(text=re.compile(r'Current Address|Lives in|Location'))
            if location_elem:
                loc_parent = location_elem.find_parent()
                if loc_parent:
                    location_text = loc_parent.get_text(strip=True)
                    preview['current_location'] = location_text
            
            # Previous addresses
            prev_addr_section = soup.find(text=re.compile(r'Previous Addresses|Past Locations'))
            if prev_addr_section:
                addr_parent = prev_addr_section.find_parent()
                if addr_parent:
                    addresses = []
                    for addr in addr_parent.find_all(['li', 'div']):
                        addr_text = addr.get_text(strip=True)
                        if addr_text and len(addr_text) > 5:
                            addresses.append(addr_text)
                    if addresses:
                        preview['previous_addresses'] = addresses[:5]
            
            # Phone numbers
            phone_section = soup.find(text=re.compile(r'Phone Numbers|Contact'))
            if phone_section:
                phone_parent = phone_section.find_parent()
                if phone_parent:
                    phones = []
                    for phone_text in phone_parent.find_all(text=re.compile(r'\d{3}[-.]?\d{3}[-.]?\d{4}')):
                        phones.append(phone_text.strip())
                    if phones:
                        preview['phones'] = phones
            
            # Email addresses
            email_section = soup.find(text=re.compile(r'Email|E-mail'))
            if email_section:
                email_parent = email_section.find_parent()
                if email_parent:
                    emails = []
                    for email in email_parent.find_all('a', href=re.compile(r'mailto:')):
                        emails.append(email.get('href').replace('mailto:', ''))
                    if emails:
                        preview['emails'] = emails
            
            # Relatives
            relatives_section = soup.find(text=re.compile(r'Relatives|Family Members|Associates'))
            if relatives_section:
                rel_parent = relatives_section.find_parent()
                if rel_parent:
                    relatives = []
                    for rel in rel_parent.find_all(['li', 'a', 'span']):
                        rel_text = rel.get_text(strip=True)
                        if rel_text and len(rel_text) > 2 and not re.match(r'^\d+$', rel_text):
                            relatives.append(rel_text)
                    if relatives:
                        preview['relatives'] = relatives[:8]
            
            # Criminal records indicator
            criminal_section = soup.find(text=re.compile(r'Criminal|Arrest|Court|Traffic'))
            if criminal_section:
                preview['has_criminal_records'] = True
            
            # Property records
            property_section = soup.find(text=re.compile(r'Property|Real Estate|Ownership'))
            if property_section:
                prop_parent = property_section.find_parent()
                if prop_parent:
                    prop_text = prop_parent.get_text(strip=True)
                    if prop_text:
                        preview['property_records'] = prop_text
            
            # Business affiliations
            business_section = soup.find(text=re.compile(r'Business|Employment|Work'))
            if business_section:
                biz_parent = business_section.find_parent()
                if biz_parent:
                    biz_text = biz_parent.get_text(strip=True)
                    if biz_text:
                        preview['business'] = biz_text
            
            # Education
            edu_section = soup.find(text=re.compile(r'Education|School|University'))
            if edu_section:
                edu_parent = edu_section.find_parent()
                if edu_parent:
                    edu_text = edu_parent.get_text(strip=True)
                    if edu_text:
                        preview['education'] = edu_text
            
            # Social media
            social_links = soup.find_all('a', href=re.compile(r'facebook|twitter|linkedin|instagram'))
            if social_links:
                preview['social_media'] = [
                    {
                        'platform': self._identify_platform(link.get('href')),
                        'url': link.get('href')
                    }
                    for link in social_links[:5]
                ]
            
            return preview if len(preview) > 1 else None
            
        except Exception as e:
            logger.debug(f"Error extracting BeenVerified preview: {e}")
            return None
    
    def _extract_record(self, element, search_type: str) -> Optional[Dict[str, Any]]:
        """
        Extract individual record from element.
        
        Args:
            element: BeautifulSoup element
            search_type: Search type
            
        Returns:
            Record dictionary or None
        """
        try:
            record = {}
            
            # Name
            name = element.find(['h3', 'h4', 'a'], class_=re.compile(r'name'))
            if name:
                record['name'] = name.get_text(strip=True)
            
            # Age
            age_text = element.find(text=re.compile(r'\d+\s*years old|Age:\s*\d+'))
            if age_text:
                age_match = re.search(r'\d+', age_text)
                if age_match:
                    record['age'] = int(age_match.group())
            
            # Location
            location = element.find(['div', 'span'], class_=re.compile(r'location|city|address'))
            if location:
                record['location'] = location.get_text(strip=True)
            
            return record if len(record) > 0 else None
            
        except Exception as e:
            logger.debug(f"Error parsing BeenVerified record: {e}")
            return None
    
    def _identify_platform(self, url: str) -> str:
        """Identify social media platform from URL."""
        url_lower = url.lower()
        platforms = {
            'facebook': 'Facebook',
            'twitter': 'Twitter',
            'x.com': 'Twitter/X',
            'linkedin': 'LinkedIn',
            'instagram': 'Instagram',
            'tiktok': 'TikTok',
            'youtube': 'YouTube'
        }
        
        for key, value in platforms.items():
            if key in url_lower:
                return value
        
        return 'Unknown'
    
    def _calculate_data_entropy(self, records: List[Dict]) -> float:
        """
        Calculate information entropy for dataset.
        
        Args:
            records: List of records
            
        Returns:
            Entropy score (0.0-1.0)
        """
        if not records:
            return 0.0
        
        unique_values = set()
        total_fields = 0
        
        for record in records:
            for key, value in record.items():
                if isinstance(value, (list, tuple)):
                    unique_values.update(str(v) for v in value)
                    total_fields += len(value)
                elif isinstance(value, dict):
                    unique_values.update(str(v) for v in value.values())
                    total_fields += len(value)
                else:
                    unique_values.add(str(value))
                    total_fields += 1
        
        if total_fields == 0:
            return 0.0
        
        entropy = min(len(unique_values) / max(total_fields, 1), 1.0)
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
