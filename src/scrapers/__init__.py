"""
Web scrapers for various OSINT sources with entropy analysis.
"""

from .truepeoplesearch import TruePeopleSearchScraper
from .whitepages import WhitepagesScraper
from .spokeo import SpokeoScraper
from .beenverified import BeenVerifiedScraper

# Scraper registry
SCRAPERS = {
    'truepeoplesearch': TruePeopleSearchScraper,
    'whitepages': WhitepagesScraper,
    'spokeo': SpokeoScraper,
    'beenverified': BeenVerifiedScraper
}


def get_scraper(source_name: str, **kwargs):
    """
    Get scraper instance by name.
    
    Args:
        source_name: Name of the scraper source
        **kwargs: Arguments to pass to scraper constructor
        
    Returns:
        Scraper instance
        
    Raises:
        ValueError: If scraper not found
    """
    if source_name not in SCRAPERS:
        raise ValueError(f"Unknown scraper: {source_name}. Available: {list(SCRAPERS.keys())}")
    
    return SCRAPERS[source_name](**kwargs)


def list_scrapers():
    """
    Get list of available scrapers.
    
    Returns:
        List of scraper names
    """
    return list(SCRAPERS.keys())


__all__ = [
    'TruePeopleSearchScraper',
    'WhitepagesScraper',
    'SpokeoScraper',
    'BeenVerifiedScraper',
    'get_scraper',
    'list_scrapers',
    'SCRAPERS'
]
