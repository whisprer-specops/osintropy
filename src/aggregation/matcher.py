# =====================================
# aggregation/matcher.py
# =====================================
"""
Record matching and deduplication logic
"""

from typing import List, Optional, Dict
from fuzzywuzzy import fuzz
from core.models import PersonRecord
from config import Config

class RecordMatcher:
    """Intelligent record matching and deduplication"""
    
    def __init__(self, threshold: float = None):
        self.threshold = threshold or Config.MATCH_THRESHOLD
    
    def find_best_match(self, search_data: Dict, 
                       candidates: List[PersonRecord]) -> Optional[PersonRecord]:
        """Find the best matching PersonRecord from candidates"""
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = self.calculate_match_score(search_data, candidate)
            
            if score > best_score and score >= self.threshold:
                best_score = score
                best_match = candidate
        
        return best_match
    
    def calculate_match_score(self, search_data: Dict, 
                            record: PersonRecord) -> float:
        """Calculate similarity score between search data and record"""
        
        scores = []
        weights = []
        
        # Name matching (highest weight)
        if 'name' in search_data and record.names:
            name_scores = [
                fuzz.token_set_ratio(search_data['name'], name) / 100.0
                for name in record.names
            ]
            scores.append(max(name_scores)