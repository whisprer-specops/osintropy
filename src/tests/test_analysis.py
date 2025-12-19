"""
Unit tests for analysis modules.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.entropy_calculator import EntropyCalculator
from analysis.anomaly_detection import AnomalyDetector, PatternMatcher


class TestEntropyCalculator(unittest.TestCase):
    """Test entropy calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = EntropyCalculator()
    
    def test_calculate_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        data = ['a', 'a', 'b', 'b', 'c', 'c']
        entropy = self.calculator.calculate_shannon_entropy(data)
        
        self.assertGreater(entropy, 0)
        self.assertLessEqual(entropy, 1.0)
    
    def test_calculate_shannon_entropy_uniform(self):
        """Test entropy with uniform distribution."""
        data = ['a', 'b', 'c', 'd']
        entropy = self.calculator.calculate_shannon_entropy(data)
        
        # Uniform distribution should have high entropy
        self.assertGreater(entropy, 0.9)
    
    def test_calculate_shannon_entropy_single(self):
        """Test entropy with single value."""
        data = ['a', 'a', 'a', 'a']
        entropy = self.calculator.calculate_shannon_entropy(data)
        
        # Single value should have zero entropy
        self.assertEqual(entropy, 0.0)


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector()
    
    def test_initialization(self):
        """Test initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(len(self.detector.anomalies), 0)
    
    def test_analyze_empty_data(self):
        """Test analysis with empty data."""
        report = self.detector.analyze({'sources': {}})
        
        self.assertIn('total_anomalies', report)
        self.assertIn('overall_anomaly_score', report)
    
    def test_detect_age_inconsistency(self):
        """Test age inconsistency detection."""
        test_data = {
            'sources': {
                'source1': {
                    'records': [{'name': 'John Doe', 'age': 30}]
                },
                'source2': {
                    'records': [{'name': 'John Doe', 'age': 50}]
                }
            }
        }
        
        report = self.detector.analyze(test_data)
        
        # Should detect age discrepancy
        self.assertGreater(report['total_anomalies'], 0)


class TestPatternMatcher(unittest.TestCase):
    """Test pattern matcher."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.matcher = PatternMatcher()
    
    def test_match_phone_pattern(self):
        """Test phone number pattern matching."""
        text = "Call me at (305) 555-1234 or 305-555-5678"
        matches = self.matcher.match_pattern(text, 'phone_formats')
        
        self.assertGreater(len(matches), 0)
    
    def test_match_email_pattern(self):
        """Test email pattern matching."""
        text = "Contact me at test@example.com"
        matches = self.matcher.match_pattern(text, 'email_patterns')
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], 'test@example.com')


if __name__ == '__main__':
    unittest.main()
