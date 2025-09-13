#!/usr/bin/env python3
"""
Site-Specific Scrapers for People Search Sites
EDUCATIONAL PURPOSES ONLY - Respect Terms of Service and Privacy Laws

This demonstrates the technical approach to scraping these sites.
In production, consider using official APIs where available.
"""

import requests
from bs4 import BeautifulSoup
import time
import random
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlencode
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import undetected_chromedriver as uc
from fake_useragent import UserAgent

ua = UserAgent()

class BasePeopleSearchScraper:
    """Base class for people search scrapers"""
    
    def __init__(self, use_selenium: bool = False):
        self.use_selenium = use_selenium
        self.session = requests.Session()
        self.driver = None
        
        if use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Setup undetected Chrome driver for sites with strong anti-bot"""
        options = uc.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument(f'user-agent={ua.random}')
        # Run headless for production
        # options.add_argument('--headless')
        
        self.driver = uc.Chrome(options=options)
    
    def _add_human_behavior(self):
        """Add random human-like behavior to avoid detection"""
        if self.driver:
            # Random mouse movements
            action = webdriver.ActionChains(self.driver)
            for _ in range(random.randint(2, 5)):
                x = random.randint(100, 800)
                y = random.randint(100, 600)
                action.move_by_offset(x, y)
                action.perform()
                time.sleep(random.uniform(0.1, 0.3))
    
    def _random_delay(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """Add random delay between requests"""
        time.sleep(random.uniform(min_sec, max_sec))

class TruePeopleSearchScraper(BasePeopleSearchScraper):
    """Scraper for TruePeopleSearch (one of the easier sites)"""
    
    BASE_URL = "https://www.truepeoplesearch.com"
    
    def search(self, first_name: str, last_name: str, city: str = "", state: str = "") -> List[Dict]:
        """Search for a person on TruePeopleSearch"""
        
        results = []
        
        # Build search URL
        search_url = f"{self.BASE_URL}/results"
        params = {
            'name': f"{first_name} {last_name}",
            'citystatezip': f"{city}, {state}" if city and state else ""
        }
        
        url = f"{search_url}?{urlencode(params)}"
        
        try:
            # Add headers to look more legitimate
            headers = {
                'User-Agent': ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            
            response = self.session.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all result cards
                result_cards = soup.find_all('div', class_='card')
                
                for card in result_cards[:10]:  # Limit to first 10 results
                    person_data = self._extract_person_data(card)
                    if person_data:
                        results.append(person_data)
            
        except Exception as e:
            print(f"Error scraping TruePeopleSearch: {e}")
        
        return results
    
    def _extract_person_data(self, card) -> Optional[Dict]:
        """Extract person data from a result card"""
        
        data = {'source': 'truepeoplesearch'}
        
        try:
            # Name
            name_elem = card.find('h2', class_='h4')
            if name_elem:
                data['name'] = name_elem.get_text(strip=True)
            
            # Age
            age_elem = card.find('span', string=re.compile(r'Age \d+'))
            if age_elem:
                data['age'] = age_elem.get_text(strip=True).replace('Age ', '')
            
            # Current address
            addr_elem = card.find('div', {'itemprop': 'address'})
            if addr_elem:
                data['current_address'] = addr_elem.get_text(strip=True)
            
            # Phone numbers
            phones = []
            phone_elems = card.find_all('span', {'itemprop': 'telephone'})
            for phone in phone_elems:
                phones.append(phone.get_text(strip=True))
            if phones:
                data['phones'] = phones
            
            # Relatives
            relatives = []
            relative_section = card.find('div', string=re.compile('Relatives'))
            if relative_section:
                relative_links = relative_section.find_next_sibling('div').find_all('a')
                for link in relative_links:
                    relatives.append(link.get_text(strip=True))
            if relatives:
                data['relatives'] = relatives
            
            # Previous addresses
            prev_addrs = []
            prev_addr_section = card.find('div', string=re.compile('Previous Addresses'))
            if prev_addr_section:
                addr_divs = prev_addr_section.find_next_siblings('div', limit=5)
                for div in addr_divs:
                    prev_addrs.append(div.get_text(strip=True))
            if prev_addrs:
                data['previous_addresses'] = prev_addrs
            
            return data if data.get('name') else None
            
        except Exception as e:
            print(f"Error extracting data: {e}")
            return None

class WhitepagesScraper(BasePeopleSearchScraper):
    """Scraper for Whitepages (uses more anti-bot measures)"""
    
    BASE_URL = "https://www.whitepages.com"
    
    def __init__(self):
        # Whitepages often requires Selenium due to JavaScript rendering
        super().__init__(use_selenium=True)
    
    def search(self, first_name: str, last_name: str, location: str = "") -> List[Dict]:
        """Search on Whitepages using Selenium"""
        
        results = []
        
        try:
            # Navigate to search page
            search_url = f"{self.BASE_URL}/name/{first_name}-{last_name}"
            if location:
                search_url += f"/{location.replace(' ', '-')}"
            
            self.driver.get(search_url)
            
            # Wait for results to load
            wait = WebDriverWait(self.driver, 10)
            
            # Add human-like behavior
            self._add_human_behavior()
            self._random_delay(2, 4)
            
            # Check for CAPTCHA
            if self._check_for_captcha():
                print("[!] CAPTCHA detected on Whitepages")
                return results
            
            # Wait for and extract results
            try:
                results_container = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "results-container"))
                )
                
                # Extract person cards
                person_cards = self.driver.find_elements(By.CLASS_NAME, "person-card")
                
                for card in person_cards[:10]:
                    person_data = self._extract_person_data_selenium(card)
                    if person_data:
                        results.append(person_data)
                        
            except TimeoutException:
                print("Results did not load in time")
                
        except Exception as e:
            print(f"Error scraping Whitepages: {e}")
        
        return results
    
    def _check_for_captcha(self) -> bool:
        """Check if CAPTCHA is present"""
        captcha_indicators = [
            "captcha",
            "recaptcha",
            "challenge",
            "verify you're human"
        ]
        
        page_source = self.driver.page_source.lower()
        return any(indicator in page_source for indicator in captcha_indicators)
    
    def _extract_person_data_selenium(self, card) -> Optional[Dict]:
        """Extract data from Selenium WebElement"""
        
        data = {'source': 'whitepages'}
        
        try:
            # Name
            name_elem = card.find_element(By.CLASS_NAME, "person-name")
            data['name'] = name_elem.text
            
            # Age
            try:
                age_elem = card.find_element(By.CLASS_NAME, "person-age")
                data['age'] = age_elem.text.replace("Age", "").strip()
            except:
                pass
            
            # Location
            try:
                location_elem = card.find_element(By.CLASS_NAME, "person-location")
                data['location'] = location_elem.text
            except:
                pass
            
            # Phone (if visible)
            try:
                phone_elem = card.find_element(By.CLASS_NAME, "phone-number")
                data['phone'] = phone_elem.text
            except:
                pass
            
            return data if data.get('name') else None
            
        except Exception as e:
            return None

class SpokeoScraper(BasePeopleSearchScraper):
    """Scraper for Spokeo (requires careful handling)"""
    
    BASE_URL = "https://www.spokeo.com"
    
    def search(self, name: str, location: str = "") -> List[Dict]:
        """Search Spokeo (note: many features require login)"""
        
        results = []
        
        # Spokeo is particularly aggressive about blocking scrapers
        # This shows the structure but would need more sophistication
        
        search_url = f"{self.BASE_URL}/search"
        
        # Use specific headers that Spokeo expects
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Origin': 'https://www.spokeo.com',
            'Connection': 'keep-alive',
            'Referer': 'https://www.spokeo.com/',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
        }
        
        # First get the search page to establish session
        self.session.get(self.BASE_URL, headers=headers)
        self._random_delay(1, 2)
        
        # Then perform search
        params = {
            'q': name,
            'loaded': '1'
        }
        
        try:
            response = self.session.get(search_url, params=params, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for result containers
                # Note: Spokeo's structure changes frequently
                result_divs = soup.find_all('div', class_='search-result')
                
                for div in result_divs[:5]:
                    data = {
                        'source': 'spokeo',
                        'name': '',
                        'preview_available': True
                    }
                    
                    # Extract what's available without login
                    name_elem = div.find('h2', class_='name')
                    if name_elem:
                        data['name'] = name_elem.get_text(strip=True)
                    
                    # Location
                    location_elem = div.find('span', class_='location')
                    if location_elem:
                        data['location'] = location_elem.get_text(strip=True)
                    
                    # Note: Most data requires paid access
                    data['note'] = 'Full details require Spokeo subscription'
                    
                    if data['name']:
                        results.append(data)
                        
        except Exception as e:
            print(f"Error with Spokeo search: {e}")
        
        return results

class BeenVerifiedScraper(BasePeopleSearchScraper):
    """Scraper for BeenVerified (mostly requires payment)"""
    
    BASE_URL = "https://www.beenverified.com"
    
    def search(self, first_name: str, last_name: str) -> List[Dict]:
        """Search BeenVerified - note most data requires payment"""
        
        # BeenVerified shows very limited free preview
        results = []
        
        search_url = f"{self.BASE_URL}/people/{first_name}-{last_name}/"
        
        headers = {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        
        try:
            response = self.session.get(search_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # BeenVerified shows minimal free info
                preview_cards = soup.find_all('div', class_='record-preview')
                
                for card in preview_cards[:5]:
                    data = {
                        'source': 'beenverified',
                        'name': '',
                        'requires_payment': True
                    }
                    
                    # Get name if available
                    name_elem = card.find('span', class_='full-name')
                    if name_elem:
                        data['name'] = name_elem.get_text(strip=True)
                    
                    # Age range (often shown in preview)
                    age_elem = card.find('span', class_='age')
                    if age_elem:
                        data['age_range'] = age_elem.get_text(strip=True)
                    
                    # Location hint
                    location_elem = card.find('span', class_='location')
                    if location_elem:
                        data['location_hint'] = location_elem.get_text(strip=True)
                    
                    data['note'] = 'Full report requires BeenVerified subscription'
                    
                    if data['name']:
                        results.append(data)
                        
        except Exception as e:
            print(f"Error with BeenVerified: {e}")
        
        return results

class AggregatedPeopleSearch:
    """Aggregate searches across multiple sites with rate limiting"""
    
    def __init__(self):
        self.scrapers = {
            'truepeoplesearch': TruePeopleSearchScraper(),
            'whitepages': WhitepagesScraper(),
            'spokeo': SpokeoScraper(),
            'beenverified': BeenVerifiedScraper()
        }
        
        # Track last request time per site
        self.last_request = {}
        
        # Minimum delays between requests per site (in seconds)
        self.rate_limits = {
            'truepeoplesearch': 3,
            'whitepages': 5,
            'spokeo': 7,
            'beenverified': 5
        }
    
    def search_all(self, first_name: str, last_name: str, 
                   location: str = "") -> Dict[str, List[Dict]]:
        """Search across all available scrapers"""
        
        all_results = {}
        
        for site_name, scraper in self.scrapers.items():
            print(f"\n[*] Searching {site_name}...")
            
            # Rate limiting
            self._enforce_rate_limit(site_name)
            
            try:
                if site_name == 'truepeoplesearch':
                    results = scraper.search(first_name, last_name)
                elif site_name == 'whitepages':
                    results = scraper.search(first_name, last_name, location)
                elif site_name == 'spokeo':
                    results = scraper.search(f"{first_name} {last_name}", location)
                elif site_name == 'beenverified':
                    results = scraper.search(first_name, last_name)
                else:
                    results = []
                
                all_results[site_name] = results
                print(f"    Found {len(results)} results")
                
            except Exception as e:
                print(f"    Error: {e}")
                all_results[site_name] = []
            
            # Random delay between sites
            time.sleep(random.uniform(2, 5))
        
        return all_results
    
    def _enforce_rate_limit(self, site_name: str):
        """Enforce rate limiting per site"""
        
        if site_name in self.last_request:
            elapsed = time.time() - self.last_request[site_name]
            required_delay = self.rate_limits.get(site_name, 5)
            
            if elapsed < required_delay:
                sleep_time = required_delay - elapsed
                print(f"    Rate limiting: waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.last_request[site_name] = time.time()

# Anti-detection techniques demo
class AntiDetectionTechniques:
    """Advanced techniques to avoid detection"""
    
    @staticmethod
    def rotate_user_agents() -> str:
        """Rotate through realistic user agents"""
        # You could also maintain a curated list of real user agents
        return ua.random
    
    @staticmethod
    def add_mouse_entropy(driver):
        """Add random mouse movements to seem human"""
        action = webdriver.ActionChains(driver)
        
        # Bezier curve mouse movement (more human-like)
        import numpy as np
        
        points = []
        for _ in range(10):
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            steps = random.randint(5, 10)
            for step in range(steps):
                t = step / steps
                # Simple linear interpolation (could use bezier)
                x = points[i][0] + t * (points[i+1][0] - points[i][0])
                y = points[i][1] + t * (points[i+1][1] - points[i][1])
                
                action.move_to_element_with_offset(
                    driver.find_element(By.TAG_NAME, 'body'),
                    int(x), int(y)
                )
                action.perform()
                time.sleep(random.uniform(0.01, 0.05))
    
    @staticmethod
    def add_typing_entropy(element, text):
        """Type with human-like delays"""
        for char in text:
            element.send_keys(char)
            # Varying delays based on character type
            if char == ' ':
                delay = random.uniform(0.1, 0.2)
            elif char in '.,!?':
                delay = random.uniform(0.2, 0.4)
            else:
                delay = random.uniform(0.05, 0.15)
            time.sleep(delay)
    
    @staticmethod
    def use_proxy_rotation():
        """Rotate through proxy servers"""
        proxies = [
            # Add your proxy list here
            # 'http://proxy1.com:8080',
            # 'http://proxy2.com:8080',
        ]
        return random.choice(proxies) if proxies else None

# Demo usage
def demo_scrapers():
    """Demonstrate the scrapers"""
    
    print("=" * 70)
    print("PEOPLE SEARCH SCRAPER DEMONSTRATION")
    print("Educational Example - Respect Privacy & Terms of Service")
    print("=" * 70)
    
    print("\n[!] Important Notes:")
    print("- These sites actively block scrapers")
    print("- Most require payment for full data")
    print("- Always check and respect robots.txt")
    print("- Consider legal implications in your jurisdiction")
    print("- Use official APIs when available")
    
    print("\n[*] Initializing scrapers...")
    aggregator = AggregatedPeopleSearch()
    
    # Example search (would use real names in practice)
    print("\n[*] Performing example search...")
    print("[*] Note: In production, results would be real")
    
    # Simulate what results might look like
    example_results = {
        'truepeoplesearch': [
            {
                'source': 'truepeoplesearch',
                'name': 'John Smith',
                'age': '45',
                'current_address': '123 Main St, Anytown, NY',
                'phones': ['555-0123', '555-0456'],
                'relatives': ['Jane Smith', 'Bob Smith']
            }
        ],
        'whitepages': [
            {
                'source': 'whitepages',
                'name': 'John Smith',
                'age': '45',
                'location': 'Anytown, NY',
                'phone': '555-0123'
            }
        ],
        'spokeo': [
            {
                'source': 'spokeo',
                'name': 'John Smith',
                'location': 'New York',
                'preview_available': True,
                'note': 'Full details require Spokeo subscription'
            }
        ]
    }
    
    print("\n" + "=" * 70)
    print("EXAMPLE RESULTS STRUCTURE")
    print("=" * 70)
    
    for site, results in example_results.items():
        print(f"\n{site.upper()}:")
        for result in results:
            print(f"  Name: {result.get('name')}")
            print(f"  Source: {result.get('source')}")
            if 'age' in result:
                print(f"  Age: {result.get('age')}")
            if 'phones' in result:
                print(f"  Phones: {', '.join(result.get('phones', []))}")
            if 'note' in result:
                print(f"  Note: {result.get('note')}")
    
    print("\n[*] Remember: This is for educational purposes")
    print("[*] Always operate within legal boundaries")
    print("[*] Respect privacy and terms of service")
    
    return aggregator

if __name__ == "__main__":
    demo_scrapers()