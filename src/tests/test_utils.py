"""
Unit tests for utility modules.
"""

import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.proxy_manager import ProxyManager
from utils.logger import get_logger


class TestProxyManager(unittest.TestCase):
    """Test proxy manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_proxies = [
            'http://proxy1.example.com:8080',
            'http://proxy2.example.com:8080',
            'http://proxy3.example.com:8080'
        ]
        self.manager = ProxyManager(self.test_proxies)
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(len(self.manager.proxies), 3)
        self.assertTrue(self.manager.enabled)
    
    def test_get_proxy_round_robin(self):
        """Test round-robin proxy selection."""
        manager = ProxyManager(self.test_proxies, rotation_strategy='round_robin')
        
        proxy1 = manager.get_proxy()
        proxy2 = manager.get_proxy()
        proxy3 = manager.get_proxy()
        proxy4 = manager.get_proxy()
        
        # Should cycle back to first proxy
        self.assertEqual(proxy1, proxy4)
    
    def test_get_proxy_random(self):
        """Test random proxy selection."""
        manager = ProxyManager(self.test_proxies, rotation_strategy='random')
        
        proxy = manager.get_proxy()
        self.assertIn(proxy, self.test_proxies)
    
    def test_report_failure(self):
        """Test failure reporting."""
        proxy = self.manager.get_proxy()
        initial_failures = self.manager.proxy_stats[proxy]['failures']
        
        self.manager.report_failure(proxy)
        
        self.assertEqual(
            self.manager.proxy_stats[proxy]['failures'],
            initial_failures + 1
        )
    
    def test_add_remove_proxy(self):
        """Test adding and removing proxies."""
        new_proxy = 'http://proxy4.example.com:8080'
        
        self.manager.add_proxy(new_proxy)
        self.assertEqual(len(self.manager.proxies), 4)
        
        self.manager.remove_proxy(new_proxy)
        self.assertEqual(len(self.manager.proxies), 3)
    
    def test_load_from_file(self):
        """Test loading proxies from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('http://proxy1.com:8080\n')
            f.write('http://proxy2.com:8080\n')
            f.write('# This is a comment\n')
            f.write('http://proxy3.com:8080\n')
            temp_file = f.name
        
        try:
            manager = ProxyManager.load_from_file(temp_file)
            self.assertEqual(len(manager.proxies), 3)
        finally:
            os.unlink(temp_file)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        self.manager.get_proxy()
        self.manager.get_proxy()
        
        stats = self.manager.get_stats()
        
        self.assertIn('total_proxies', stats)
        self.assertIn('total_uses', stats)
        self.assertEqual(stats['total_uses'], 2)


class TestLogger(unittest.TestCase):
    """Test logger utilities."""
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger('test_logger')
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_logger')


if __name__ == '__main__':
    unittest.main()
