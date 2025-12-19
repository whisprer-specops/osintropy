# =====================================
# core/entropy.py
# =====================================
"""
Entropy calculation and analysis functions
"""

import math
import numpy as np
from typing import Set, List, Dict
from collections import Counter

class EntropyAnalyzer:
    """Calculate various entropy metrics"""
    
    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Character frequency
        freq = Counter(text.lower())
        total = len(text)
        
        entropy = 0
        for count in freq.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    @staticmethod
    def calculate_name_entropy(names: Set[str]) -> float:
        """Calculate entropy across name variations"""
        if not names:
            return 0.0
        
        # Combine all names
        all_text = ' '.join(names)
        return EntropyAnalyzer.calculate_shannon_entropy(all_text)
    
    @staticmethod
    def calculate_temporal_entropy(timestamps: List[float]) -> float:
        """Calculate entropy of temporal patterns"""
        if len(timestamps) < 2:
            return 0.0
        
        # Sort and calculate intervals
        sorted_times = sorted(timestamps)
        intervals = np.diff(sorted_times)
        
        if len(intervals) == 0:
            return 0.0
        
        # Normalize intervals
        normalized = intervals / np.sum(intervals)
        
        # Calculate entropy
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        
        return entropy
    
    @staticmethod
    def calculate_geographic_entropy(addresses: List) -> float:
        """Calculate entropy of geographic distribution"""
        if not addresses:
            return 0.0
        
        states = set()
        cities = set()
        
        for addr in addresses:
            if hasattr(addr, 'state') and addr.state:
                states.add(addr.state)
            if hasattr(addr, 'city') and addr.city:
                cities.add(addr.city)
        
        # Normalize by expected maximums
        state_entropy = len(states) / 50.0  # US states
        city_entropy = min(len(cities) / 20.0, 1.0)  # Cap at 20 cities
        
        return (state_entropy + city_entropy) / 2
    
    @staticmethod
    def calculate_source_diversity(sources: Set[str]) -> float:
        """Calculate diversity of data sources"""
        if not sources:
            return 0.0
        
        # More sources = higher confidence
        return min(len(sources) / 5.0, 1.0)
    