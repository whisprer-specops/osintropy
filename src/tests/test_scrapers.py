"""
Unit tests for web scrapers.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrapers.truepeoplesearch import TruePeopleSearchScraper
from scrapers.whitepages import WhitepagesScraper
from scrapers.spokeo import SpokeoScraper
from scrapers.beenverified import BeenVerifiedScraper


class TestTruePeopleSearchScraper(unittest.TestCase):
    """Test TruePeopleSearch scraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = TruePeopleSearchScraper()
    
    def test_initialization(self):
        """Test scraper initialization."""
        self.assertEqual(self.scraper.source_name, 'truepeoplesearch')
        self.assertIsNotNone(self.scraper.base_url)
    
    def test_search_url_generation(self):
        """Test search URL generation."""
        # This would test URL building logic
        self.assertIn('truepeoplesearch', self.scraper.base_url)
    
    @patch('scrapers.truepeoplesearch.requests.Session.get')
    def test_search_person_success(self, mock_get):
        """Test successful person search."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '<html><body><div class="result"></div></body></html>'
        mock_get.return_value = mock_response
        
        result = self.scraper.search_person('John', 'Doe', 'Miami, FL')
        
        self.assertIn('source', result)
        self.assertEqual(result['source'], 'truepeoplesearch')
        self.assertIn('records', result)
    
    @patch('scrapers.truepeoplesearch.requests.Session.get')
    def test_search_person_rate_limit(self, mock_get):
        """Test rate limit handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response
        
        result = self.scraper.search_person('John', 'Doe')
        
        self.assertIn('error', result)


class TestWhitepagesScraper(unittest.TestCase):
    """Test WhitePages scraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = WhitepagesScraper()
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.scraper.source_name, 'whitepages')
    
    def test_phone_cleaning(self):
        """Test phone number cleaning."""
        result = self.scraper.reverse_phone('(305) 555-1234')
        # Should handle formatted input
        self.assertIn('source', result)


class TestSpokeoScraper(unittest.TestCase):
    """Test Spokeo scraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = SpokeoScraper()
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.scraper.source_name, 'spokeo')
    
    def test_email_validation(self):
        """Test email validation."""
        result = self.scraper.search_email('invalid-email')
        self.assertIn('error', result)


class TestBeenVerifiedScraper(unittest.TestCase):
    """Test BeenVerified scraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = BeenVerifiedScraper()
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.scraper.source_name, 'beenverified')


if __name__ == '__main__':
    unittest.main()
