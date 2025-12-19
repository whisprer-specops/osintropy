"""
Shannon entropy calculator for OSINT data analysis.
Calculates information entropy to measure data quality and diversity.
"""

import math
from typing import List, Dict, Any, Union
from collections import Counter


class EntropyCalculator:
    """
    Calculates Shannon entropy and related information theory metrics
    for OSINT data quality assessment.
    """
    
    def __init__(self):
        """Initialize entropy calculator."""
        pass
    
    def calculate_shannon_entropy(self, data: List[Any]) -> float:
        """
        Calculate Shannon entropy for a dataset.
        
        Shannon entropy H(X) = -Σ p(x) * log2(p(x))
        where p(x) is the probability of each unique value.
        
        Args:
            data: List of data points
            
        Returns:
            Entropy value normalized to 0.0-1.0 range
        """
        if not data:
            return 0.0
        
        # Count frequencies
        counter = Counter(str(item) for item in data)
        total = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)
        
        # Normalize to 0-1 range
        # Maximum entropy is log2(n) where n is number of unique values
        max_entropy = math.log2(len(counter)) if len(counter) > 1 else 1.0
        
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return round(normalized_entropy, 3)
    
    def calculate_field_entropy(self, records: List[Dict[str, Any]], 
                               field_name: str) -> float:
        """
        Calculate entropy for a specific field across records.
        
        Args:
            records: List of record dictionaries
            field_name: Field name to analyze
            
        Returns:
            Entropy score for the field
        """
        values = []
        
        for record in records:
            if field_name in record:
                value = record[field_name]
                
                # Handle list values
                if isinstance(value, list):
                    values.extend(str(v) for v in value)
                else:
                    values.append(str(value))
        
        return self.calculate_shannon_entropy(values)
    
    def calculate_dataset_entropy(self, records: List[Dict[str, Any]]) -> float:
        """
        Calculate overall entropy for a dataset.
        
        Args:
            records: List of record dictionaries
            
        Returns:
            Average entropy across all fields
        """
        if not records:
            return 0.0
        
        # Collect all field names
        all_fields = set()
        for record in records:
            all_fields.update(record.keys())
        
        # Calculate entropy for each field
        field_entropies = []
        for field in all_fields:
            entropy = self.calculate_field_entropy(records, field)
            field_entropies.append(entropy)
        
        # Return average
        return round(sum(field_entropies) / len(field_entropies), 3) if field_entropies else 0.0
    
    def calculate_uniqueness_ratio(self, data: List[Any]) -> float:
        """
        Calculate ratio of unique values to total values.
        
        Args:
            data: List of data points
            
        Returns:
            Uniqueness ratio (0.0-1.0)
        """
        if not data:
            return 0.0
        
        unique_count = len(set(str(item) for item in data))
        total_count = len(data)
        
        return round(unique_count / total_count, 3)
    
    def calculate_diversity_score(self, records: List[Dict[str, Any]]) -> float:
        """
        Calculate diversity score combining entropy and uniqueness.
        
        Args:
            records: List of record dictionaries
            
        Returns:
            Diversity score (0.0-1.0)
        """
        entropy = self.calculate_dataset_entropy(records)
        
        # Calculate average uniqueness across fields
        all_fields = set()
        for record in records:
            all_fields.update(record.keys())
        
        uniqueness_scores = []
        for field in all_fields:
            values = []
            for record in records:
                if field in record:
                    value = record[field]
                    if isinstance(value, list):
                        values.extend(value)
                    else:
                        values.append(value)
            
            if values:
                uniqueness_scores.append(self.calculate_uniqueness_ratio(values))
        
        avg_uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0.0
        
        # Weighted combination (60% entropy, 40% uniqueness)
        diversity = (0.6 * entropy) + (0.4 * avg_uniqueness)
        
        return round(diversity, 3)
    
    def analyze_data_quality(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis using entropy metrics.
        
        Args:
            records: List of record dictionaries
            
        Returns:
            Quality analysis report
        """
        if not records:
            return {
                'entropy': 0.0,
                'diversity': 0.0,
                'quality_score': 0.0,
                'field_analysis': {}
            }
        
        # Overall metrics
        entropy = self.calculate_dataset_entropy(records)
        diversity = self.calculate_diversity_score(records)
        
        # Per-field analysis
        all_fields = set()
        for record in records:
            all_fields.update(record.keys())
        
        field_analysis = {}
        for field in all_fields:
            values = []
            for record in records:
                if field in record:
                    value = record[field]
                    if isinstance(value, list):
                        values.extend(value)
                    else:
                        values.append(value)
            
            if values:
                field_analysis[field] = {
                    'entropy': self.calculate_shannon_entropy(values),
                    'uniqueness': self.calculate_uniqueness_ratio(values),
                    'total_values': len(values),
                    'unique_values': len(set(str(v) for v in values))
                }
        
        # Overall quality score (combination of metrics)
        quality_score = (entropy + diversity) / 2
        
        return {
            'entropy': entropy,
            'diversity': diversity,
            'quality_score': round(quality_score, 3),
            'total_records': len(records),
            'field_count': len(all_fields),
            'field_analysis': field_analysis
        }
    
    def compare_sources(self, source_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compare entropy across multiple data sources.
        
        Args:
            source_data: Dictionary mapping source names to record lists
            
        Returns:
            Comparison report
        """
        comparison = {}
        
        for source_name, records in source_data.items():
            analysis = self.analyze_data_quality(records)
            comparison[source_name] = {
                'entropy': analysis['entropy'],
                'diversity': analysis['diversity'],
                'quality_score': analysis['quality_score'],
                'record_count': len(records)
            }
        
        # Find best and worst sources
        if comparison:
            sorted_by_quality = sorted(
                comparison.items(),
                key=lambda x: x[1]['quality_score'],
                reverse=True
            )
            
            return {
                'sources': comparison,
                'best_source': sorted_by_quality[0][0] if sorted_by_quality else None,
                'worst_source': sorted_by_quality[-1][0] if sorted_by_quality else None,
                'average_quality': round(
                    sum(s['quality_score'] for s in comparison.values()) / len(comparison),
                    3
                )
            }
        
        return {'sources': {}, 'best_source': None, 'worst_source': None, 'average_quality': 0.0}
