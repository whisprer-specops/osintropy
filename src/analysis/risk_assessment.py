"""
Risk assessment module for OSINT data analysis.
Evaluates potential risks and confidence levels in aggregated intelligence.
"""

from typing import Dict, List, Any
from collections import Counter

from utils.logger import get_logger

logger = get_logger(__name__)


class RiskAssessor:
    """
    Assesses risk levels and confidence scores for OSINT findings.
    """
    
    def __init__(self):
        """Initialize risk assessor."""
        self.risk_factors = []
        
    def assess(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk and confidence levels for aggregated data.
        
        Args:
            aggregated_data: Aggregated OSINT data
            
        Returns:
            Risk assessment report
        """
        logger.info("Starting risk assessment...")
        
        self.risk_factors = []
        
        # Assess various risk dimensions
        source_confidence = self._assess_source_confidence(aggregated_data)
        data_consistency = self._assess_data_consistency(aggregated_data)
        coverage_score = self._assess_coverage(aggregated_data)
        recency_score = self._assess_data_recency(aggregated_data)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            source_confidence,
            data_consistency,
            coverage_score,
            recency_score
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_confidence)
        
        report = {
            'overall_confidence': overall_confidence,
            'risk_level': risk_level,
            'source_confidence': source_confidence,
            'data_consistency': data_consistency,
            'coverage_score': coverage_score,
            'recency_score': recency_score,
            'risk_factors': self.risk_factors,
            'recommendations': self._generate_recommendations(overall_confidence, risk_level)
        }
        
        logger.info(f"Risk assessment complete: {risk_level} risk, {overall_confidence:.2f} confidence")
        return report
    
    def _assess_source_confidence(self, data: Dict[str, Any]) -> float:
        """
        Assess confidence based on number and quality of sources.
        
        Args:
            data: Aggregated data
            
        Returns:
            Source confidence score (0.0-1.0)
        """
        sources = data.get('sources', {})
        num_sources = len(sources)
        
        if num_sources == 0:
            self.risk_factors.append("No data sources available")
            return 0.0
        
        # Base score on number of sources
        if num_sources >= 4:
            base_score = 1.0
        elif num_sources == 3:
            base_score = 0.8
        elif num_sources == 2:
            base_score = 0.6
        else:
            base_score = 0.4
            self.risk_factors.append("Limited number of data sources")
        
        # Adjust for source quality (entropy)
        total_entropy = 0.0
        for source_data in sources.values():
            total_entropy += source_data.get('entropy_score', 0.0)
        
        avg_entropy = total_entropy / num_sources if num_sources > 0 else 0.0
        
        # Penalize low entropy
        if avg_entropy < 0.3:
            self.risk_factors.append("Low data entropy indicates poor quality")
            base_score *= 0.7
        
        return round(min(base_score, 1.0), 3)
    
    def _assess_data_consistency(self, data: Dict[str, Any]) -> float:
        """
        Assess consistency of data across sources.
        
        Args:
            data: Aggregated data
            
        Returns:
            Consistency score (0.0-1.0)
        """
        sources = data.get('sources', {})
        
        if len(sources) < 2:
            return 1.0  # Can't measure consistency with single source
        
        # Collect key fields across sources
        names = set()
        ages = []
        locations = set()
        phones = set()
        
        for source_data in sources.values():
            for record in source_data.get('records', []):
                if 'name' in record:
                    names.add(record['name'].lower().strip())
                
                if 'age' in record:
                    ages.append(record['age'])
                
                if 'location' in record:
                    locations.add(str(record['location']).lower())
                
                if 'phones' in record:
                    phone_list = record['phones']
                    if isinstance(phone_list, list):
                        phones.update(phone_list)
                    else:
                        phones.add(str(phone_list))
        
        consistency_scores = []
        
        # Name consistency (should be mostly the same)
        if len(names) > 1:
            name_consistency = 1.0 / len(names)
            consistency_scores.append(name_consistency)
            if len(names) > 2:
                self.risk_factors.append(f"Multiple name variations found: {len(names)}")
        
        # Age consistency (should be within a few years)
        if len(ages) > 1:
            age_range = max(ages) - min(ages)
            if age_range > 5:
                age_consistency = max(0.0, 1.0 - (age_range / 20.0))
                consistency_scores.append(age_consistency)
                self.risk_factors.append(f"Age discrepancy of {age_range} years")
            else:
                consistency_scores.append(1.0)
        
        # Overall consistency
        if consistency_scores:
            return round(sum(consistency_scores) / len(consistency_scores), 3)
        
        return 0.8  # Default moderate consistency
    
    def _assess_coverage(self, data: Dict[str, Any]) -> float:
        """
        Assess data coverage (completeness).
        
        Args:
            data: Aggregated data
            
        Returns:
            Coverage score (0.0-1.0)
        """
        sources = data.get('sources', {})
        
        # Count presence of key data types
        has_name = False
        has_age = False
        has_location = False
        has_phone = False
        has_email = False
        has_relatives = False
        
        for source_data in sources.values():
            for record in source_data.get('records', []):
                if 'name' in record:
                    has_name = True
                if 'age' in record:
                    has_age = True
                if 'location' in record or 'addresses' in record:
                    has_location = True
                if 'phones' in record:
                    has_phone = True
                if 'emails' in record or 'email' in record:
                    has_email = True
                if 'relatives' in record:
                    has_relatives = True
        
        # Calculate coverage
        data_types = [has_name, has_age, has_location, has_phone, has_email, has_relatives]
        coverage = sum(data_types) / len(data_types)
        
        if coverage < 0.5:
            self.risk_factors.append("Low data coverage - many fields missing")
        
        return round(coverage, 3)
    
    def _assess_data_recency(self, data: Dict[str, Any]) -> float:
        """
        Assess how recent the data is.
        
        Args:
            data: Aggregated data
            
        Returns:
            Recency score (0.0-1.0)
        """
        import time
        
        sources = data.get('sources', {})
        current_time = time.time()
        
        timestamps = []
        for source_data in sources.values():
            if 'timestamp' in source_data:
                timestamps.append(source_data['timestamp'])
        
        if not timestamps:
            return 0.5  # Unknown recency
        
        # Get most recent timestamp
        most_recent = max(timestamps)
        age_hours = (current_time - most_recent) / 3600
        
        # Score based on age
        if age_hours < 1:
            score = 1.0
        elif age_hours < 24:
            score = 0.9
        elif age_hours < 168:  # 1 week
            score = 0.7
        elif age_hours < 720:  # 30 days
            score = 0.5
        else:
            score = 0.3
            self.risk_factors.append(f"Data is {int(age_hours/24)} days old")
        
        return round(score, 3)
    
    def _calculate_overall_confidence(self, source_conf: float, 
                                     consistency: float,
                                     coverage: float,
                                     recency: float) -> float:
        """
        Calculate weighted overall confidence score.
        
        Args:
            source_conf: Source confidence
            consistency: Data consistency
            coverage: Data coverage
            recency: Data recency
            
        Returns:
            Overall confidence (0.0-1.0)
        """
        # Weighted average
        weights = {
            'source': 0.3,
            'consistency': 0.3,
            'coverage': 0.25,
            'recency': 0.15
        }
        
        overall = (
            source_conf * weights['source'] +
            consistency * weights['consistency'] +
            coverage * weights['coverage'] +
            recency * weights['recency']
        )
        
        return round(overall, 3)
    
    def _determine_risk_level(self, confidence: float) -> str:
        """
        Determine risk level from confidence score.
        
        Args:
            confidence: Overall confidence score
            
        Returns:
            Risk level string
        """
        if confidence >= 0.8:
            return "LOW"
        elif confidence >= 0.6:
            return "MODERATE"
        elif confidence >= 0.4:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_recommendations(self, confidence: float, 
                                 risk_level: str) -> List[str]:
        """
        Generate recommendations based on assessment.
        
        Args:
            confidence: Confidence score
            risk_level: Risk level
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if risk_level == "CRITICAL":
            recommendations.append("CRITICAL: Data quality is very poor. Do not rely on this intelligence.")
            recommendations.append("Gather additional sources before making any decisions.")
        elif risk_level == "HIGH":
            recommendations.append("HIGH RISK: Significant data quality concerns detected.")
            recommendations.append("Verify all findings through additional independent sources.")
        elif risk_level == "MODERATE":
            recommendations.append("MODERATE RISK: Some data quality issues present.")
            recommendations.append("Cross-reference key findings with additional sources.")
        else:
            recommendations.append("LOW RISK: Data appears reliable.")
            recommendations.append("Confidence is high, but always verify critical findings.")
        
        # Specific recommendations based on risk factors
        if any('entropy' in factor.lower() for factor in self.risk_factors):
            recommendations.append("Seek sources with more diverse and detailed information.")
        
        if any('source' in factor.lower() for factor in self.risk_factors):
            recommendations.append("Increase number of data sources for better coverage.")
        
        if any('age' in factor.lower() or 'discrepancy' in factor.lower() for factor in self.risk_factors):
            recommendations.append("Investigate data inconsistencies manually.")
        
        return recommendations
