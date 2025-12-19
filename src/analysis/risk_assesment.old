# =====================================
# analysis/risk_assessment.py
# =====================================
"""
Risk assessment and anomaly detection
"""

from typing import List
from core.models import PersonRecord
from core.entropy import EntropyAnalyzer
from config import Config

class RiskAssessor:
    """Assess risk and identify anomalies in person records"""
    
    def __init__(self):
        self.entropy_analyzer = EntropyAnalyzer()
    
    def assess(self, record: PersonRecord) -> List[str]:
        """Assess risk indicators for a record"""
        
        indicators = []
        
        # Calculate entropy metrics
        name_entropy = self.entropy_analyzer.calculate_name_entropy(record.names)
        geo_entropy = self.entropy_analyzer.calculate_geographic_entropy(record.addresses)
        source_diversity = self.entropy_analyzer.calculate_source_diversity(record.sources)
        
        # Store in record
        record.entropy_profile = {
            'name_entropy': name_entropy,
            'geographic_entropy': geo_entropy,
            'source_diversity': source_diversity
        }
        
        # Check for anomalies
        
        # Name anomalies
        if name_entropy < Config.NAME_ENTROPY_SUSPICIOUS:
            indicators.append("Low name entropy - possible fake identity")
        
        if len(record.names) > 5:
            indicators.append("Excessive name variations - possible identity confusion")
        
        # Geographic anomalies
        if geo_entropy > 0.8:
            indicators.append("High geographic mobility - unusual address patterns")
        
        # Source anomalies
        if source_diversity < 0.4:
            indicators.append("Low source diversity - limited verification")
        
        # Missing data anomalies
        if not record.phone_numbers:
            indicators.append("No phone numbers found - incomplete profile")
        
        if not record.addresses:
            indicators.append("No addresses found - suspicious lack of data")
        
        # Calculate overall confidence
        confidence = 0.0
        confidence += source_diversity * 0.4
        confidence += min(len(record.phone_numbers) / 3.0, 1.0) * 0.3
        confidence += min(len(record.addresses) / 5.0, 1.0) * 0.3
        
        record.confidence_scores['overall'] = confidence
        
        return indicators