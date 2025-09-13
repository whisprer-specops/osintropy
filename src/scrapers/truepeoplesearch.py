# =====================================
# scrapers/truepeoplesearch.py
# =====================================
"""
TruePeopleSearch scraper implementation
"""

from bs4 import BeautifulSoup
from typing import Dict, List
from urllib.parse import urlencode
import re
from scrapers.base import BaseScraper
from core.models import Address, PhoneNumber
from config import Config

class TruePeopleSearchScraper(BaseScraper):
    """Scraper for TruePeopleSearch.com"""
    
    BASE_URL = "https://www.truepeoplesearch.com"
    
    def __init__(self):
        super().__init__(use_selenium=False)
        self.site_name = 'truepeoplesearch'
    
    def search(self, first_name: str, last_name: str, 
              location: str = None, **kwargs) -> List[Dict]:
        """Search TruePeopleSearch"""
        
        results = []
        
        # Build search URL
        params = {
            'name': f"{first_name} {last_name}",
            'citystatezip': location or ""
        }
        
        search_url = f"{self.BASE_URL}/results?{urlencode(params)}"
        
        try:
            # Add delay before request
            self._add_delay(2, 4)
            
            response = self.session.get(
                search_url,
                headers=self._get_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = self._parse_results(soup)
            
            self.request_count += 1
            
        except Exception as e:
            print(f"[!] Error scraping {self.site_name}: {e}")
        
        return results
    
    def _parse_results(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse search results from HTML"""
        
        results = []
        
        # Find all result cards
        cards = soup.find_all('div', class_=['card', 'result-card'])
        
        for card in cards[:Config.MAX_RESULTS_PER_SITE]:
            person_data = self._extract_person_data(card)
            if person_data:
                person_data['source'] = self.site_name
                results.append(person_data)
        
        return results
    
    def _extract_person_data(self, card) -> Dict:
        """Extract data from a result card"""
        
        data = {}
        
        # Name
        name_elem = card.find(['h2', 'h3'], class_=['h4', 'name'])
        if name_elem:
            data['name'] = name_elem.get_text(strip=True)
        
        # Age
        age_pattern = re.compile(r'Age\s*(\d+)')
        age_elem = card.find(text=age_pattern)
        if age_elem:
            match = age_pattern.search(age_elem)
            if match:
                data['age'] = match.group(1)
        
        # Current address
        addr_elem = card.find(['div', 'span'], {'itemprop': 'address'})
        if addr_elem:
            data['current_address'] = addr_elem.get_text(strip=True)
        
        # Phone numbers
        phones = []
        phone_elems = card.find_all(['span', 'div'], {'itemprop': 'telephone'})
        for phone in phone_elems:
            phone_text = phone.get_text(strip=True)
            if phone_text:
                phones.append(phone_text)
        if phones:
            data['phones'] = phones
        
        # Relatives
        relatives = []
        rel_section = card.find(text=re.compile('Relatives'))
        if rel_section:
            rel_container = rel_section.find_parent().find_next_sibling()
            if rel_container:
                rel_links = rel_container.find_all('a')
                for link in rel_links:
                    relatives.append(link.get_text(strip=True))
        if relatives:
            data['relatives'] = relatives
        
        return data if data.get('name') else None