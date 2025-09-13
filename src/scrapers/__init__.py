# =====================================
# scrapers/__init__.py
# =====================================
"""
Scraper module initialization and factory
"""

from typing import Optional
from scrapers.base import BaseScraper
from scrapers.truepeoplesearch import TruePeopleSearchScraper
from scrapers.whitepages import WhitepagesScraper
from scrapers.spokeo import SpokeoScraper
from scrapers.beenverified import BeenVerifiedScraper

AVAILABLE_SCRAPERS = {
    'truepeoplesearch': TruePeopleSearchScraper,
    'whitepages': WhitepagesScraper,
    'spokeo': SpokeoScraper,
    'beenverified': BeenVerifiedScraper
}

def get_scraper(site_name: str) -> Optional[BaseScraper]:
    """Factory function to get appropriate scraper"""
    scraper_class = AVAILABLE_SCRAPERS.get(site_name)
    if scraper_class:
        return scraper_class()
    return None

def get_all_scrapers():
    """Get instances of all available scrapers"""
    return {name: cls() for name, cls in AVAILABLE_SCRAPERS.items()}