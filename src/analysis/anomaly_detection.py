"""
Anomaly detection for OSINT data using entropy analysis and pattern recognition.
Identifies unusual patterns, inconsistencies, and potential data quality issues.
"""

import math
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import re
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in OSINT data using entropy analysis,
    statistical methods, and pattern recognition.
    """
    
    def __init__(self, sensitivity: float = 0.7):
        """
        Initialize anomaly detector.
        
        Args:
            sensitivity: Detection sensitivity (0.0-1.0, higher = more sensitive)
        """
        self.sensitivity = sensitivity
        self.baseline_patterns = {}
        self.anomalies = []
        
    def analyze(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze aggregated OSINT data for anomalies.
        
        Args:
            aggregated_data: Data from OSINTAggregator
            
        Returns:
            Anomaly report dictionary
        """
        logger.info("Starting anomaly detection analysis...")
        
        self.anomalies = []
        
        # Run various anomaly detection methods
        self._detect_data_inconsistencies(aggregated_data)
        self._detect_entropy_anomalies(aggregated_data)
        self._detect_temporal_anomalies(aggregated_data)
        self._detect_geographic_anomalies(aggregated_data)
        self._detect_relationship_anomalies(aggregated_data)
        self._detect_statistical_outliers(aggregated_data)
        
        # Calculate overall anomaly score
        overall_score = self._calculate_overall_score()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_anomalies': len(self.anomalies),
            'anomalies_by_severity': self._group_by_severity(),
            'anomalies_by_type': self._group_by_type(),
            'overall_anomaly_score': overall_score,
            'anomalies': sorted(self.anomalies, key=lambda x: x['severity'], reverse=True),
            'recommendations': self._generate_recommendations()
        }
        
        logger.info(f"Detected {len(self.anomalies)} anomalies (score: {overall_score:.3f})")
        return report
    
    def _detect_data_inconsistencies(self, data: Dict[str, Any]):
        """
        Detect inconsistencies across data sources.
        
        Args:
            data: Aggregated data
        """
        logger.debug("Checking data inconsistencies...")
        
        sources = data.get('sources', {})
        
        # Cross-reference key fields across sources
        name_variations = set()
        age_values = []
        location_sets = defaultdict(set)
        
        for source_name, source_data in sources.items():
            for record in source_data.get('records', []):
                # Collect names
                if 'name' in record:
                    name_variations.add(record['name'].strip().lower())
                
                # Collect ages
                if 'age' in record:
                    age_values.append((source_name, record['age']))
                
                # Collect locations
                locations = record.get('addresses', []) or record.get('locations', [])
                if isinstance(locations, str):
                    locations = [locations]
                for loc in locations:
                    if isinstance(loc, str):
                        location_sets[source_name].add(loc.lower())
        
        # Detect name inconsistencies
        if len(name_variations) > 1:
            self._add_anomaly(
                'data_inconsistency',
                'Multiple name variations found across sources',
                0.6,
                {'names': list(name_variations)}
            )
        
        # Detect age inconsistencies
        if len(age_values) > 1:
            ages = [age for _, age in age_values]
            age_range = max(ages) - min(ages)
            if age_range > 5:  # More than 5 years difference
                self._add_anomaly(
                    'data_inconsistency',
                    f'Age discrepancy of {age_range} years across sources',
                    0.7,
                    {'age_values': age_values}
                )
        
        # Detect completely disjoint location sets
        if len(location_sets) > 1:
            all_locations = set.union(*location_sets.values())
            for source, locs in location_sets.items():
                if locs.isdisjoint(all_locations - locs):
                    self._add_anomaly(
                        'data_inconsistency',
                        f'No location overlap with other sources for {source}',
                        0.5,
                        {'source': source, 'unique_locations': list(locs)}
                    )
    
    def _detect_entropy_anomalies(self, data: Dict[str, Any]):
        """
        Detect anomalies based on entropy scores.
        
        Args:
            data: Aggregated data
        """
        logger.debug("Analyzing entropy anomalies...")
        
        entropy_scores = []
        
        for source_name, source_data in data.get('sources', {}).items():
            score = source_data.get('entropy_score', 0.0)
            entropy_scores.append((source_name, score))
        
        if not entropy_scores:
            return
        
        # Calculate mean and standard deviation
        scores = [s for _, s in entropy_scores]
        mean_score = sum(scores) / len(scores)
        
        if len(scores) > 1:
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = math.sqrt(variance)
            
            # Detect outliers (> 2 standard deviations)
            for source_name, score in entropy_scores:
                if abs(score - mean_score) > 2 * std_dev:
                    severity = min(abs(score - mean_score) / (3 * std_dev), 1.0)
                    self._add_anomaly(
                        'entropy_anomaly',
                        f'{source_name} has unusual entropy score: {score:.3f}',
                        severity,
                        {'source': source_name, 'score': score, 'mean': mean_score}
                    )
        
        # Detect very low entropy (poor data quality)
        for source_name, score in entropy_scores:
            if score < 0.2:
                self._add_anomaly(
                    'low_entropy',
                    f'{source_name} has very low entropy (poor data diversity)',
                    0.6,
                    {'source': source_name, 'score': score}
                )
    
    def _detect_temporal_anomalies(self, data: Dict[str, Any]):
        """
        Detect temporal anomalies (unusual timing, outdated data, etc.).
        
        Args:
            data: Aggregated data
        """
        logger.debug("Checking temporal anomalies...")
        
        current_time = datetime.now().timestamp()
        
        for source_name, source_data in data.get('sources', {}).items():
            timestamp = source_data.get('timestamp', current_time)
            age_seconds = current_time - timestamp
            
            # Warn if data is very old (> 7 days)
            if age_seconds > 7 * 24 * 3600:
                days_old = age_seconds / (24 * 3600)
                self._add_anomaly(
                    'temporal_anomaly',
                    f'{source_name} data is {days_old:.1f} days old',
                    0.4,
                    {'source': source_name, 'age_days': days_old}
                )
            
            # Check for records with temporal inconsistencies
            for record in source_data.get('records', []):
                age = record.get('age')
                if age and (age < 0 or age > 120):
                    self._add_anomaly(
                        'temporal_anomaly',
                        f'Impossible age value: {age}',
                        0.8,
                        {'source': source_name, 'age': age}
                    )
    
    def _detect_geographic_anomalies(self, data: Dict[str, Any]):
        """
        Detect geographic anomalies (impossible locations, inconsistencies).
        
        Args:
            data: Aggregated data
        """
        logger.debug("Analyzing geographic patterns...")
        
        all_locations = []
        
        for source_name, source_data in data.get('sources', {}).items():
            for record in source_data.get('records', []):
                locations = record.get('addresses', []) or record.get('locations', [])
                if isinstance(locations, str):
                    locations = [locations]
                
                for loc in locations:
                    if isinstance(loc, str):
                        all_locations.append((source_name, loc))
        
        if not all_locations:
            return
        
        # Extract states/countries
        locations_by_region = defaultdict(list)
        
        for source, location in all_locations:
            # Simple state extraction (US focused)
            state_match = re.search(r'\b([A-Z]{2})\b', location)
            if state_match:
                state = state_match.group(1)
                locations_by_region[state].append((source, location))
        
        # Detect if person appears in too many different states
        if len(locations_by_region) > 5:
            self._add_anomaly(
                'geographic_anomaly',
                f'Person associated with {len(locations_by_region)} different states',
                0.6,
                {'state_count': len(locations_by_region), 'states': list(locations_by_region.keys())}
            )
    
    def _detect_relationship_anomalies(self, data: Dict[str, Any]):
        """
        Detect anomalies in relationship data.
        
        Args:
            data: Aggregated data
        """
        logger.debug("Checking relationship anomalies...")
        
        all_relatives = []
        
        for source_name, source_data in data.get('sources', {}).items():
            for record in source_data.get('records', []):
                relatives = record.get('relatives', []) or record.get('associates', [])
                if relatives:
                    all_relatives.extend(relatives)
        
        if not all_relatives:
            return
        
        # Detect unusually high number of relatives
        if len(all_relatives) > 20:
            self._add_anomaly(
                'relationship_anomaly',
                f'Unusually high number of relatives/associates: {len(all_relatives)}',
                0.5,
                {'count': len(all_relatives)}
            )
        
        # Detect duplicate relationships
        relative_counts = Counter(all_relatives)
        duplicates = {name: count for name, count in relative_counts.items() if count > 3}
        
        if duplicates:
            self._add_anomaly(
                'relationship_anomaly',
                'Duplicate relationships found across sources',
                0.4,
                {'duplicates': duplicates}
            )
    
    def _detect_statistical_outliers(self, data: Dict[str, Any]):
        """
        Detect statistical outliers in numeric fields.
        
        Args:
            data: Aggregated data
        """
        logger.debug("Finding statistical outliers...")
        
        # Collect numeric fields
        phone_counts = []
        email_counts = []
        address_counts = []
        
        for source_name, source_data in data.get('sources', {}).items():
            for record in source_data.get('records', []):
                phones = record.get('phones', [])
                if phones:
                    phone_counts.append(len(phones) if isinstance(phones, list) else 1)
                
                emails = record.get('emails', []) or record.get('email', [])
                if emails:
                    email_counts.append(len(emails) if isinstance(emails, list) else 1)
                
                addresses = record.get('addresses', []) or record.get('locations', [])
                if addresses:
                    address_counts.append(len(addresses) if isinstance(addresses, list) else 1)
        
        # Detect outliers
        if phone_counts and max(phone_counts) > 5:
            self._add_anomaly(
                'statistical_outlier',
                f'Unusually high number of phone numbers: {max(phone_counts)}',
                0.5,
                {'max_phones': max(phone_counts)}
            )
        
        if email_counts and max(email_counts) > 5:
            self._add_anomaly(
                'statistical_outlier',
                f'Unusually high number of email addresses: {max(email_counts)}',
                0.5,
                {'max_emails': max(email_counts)}
            )
        
        if address_counts and max(address_counts) > 10:
            self._add_anomaly(
                'statistical_outlier',
                f'Unusually high number of addresses: {max(address_counts)}',
                0.6,
                {'max_addresses': max(address_counts)}
            )
    
    def _add_anomaly(self, anomaly_type: str, description: str, 
                     severity: float, metadata: Dict[str, Any]):
        """
        Add anomaly to detection list.
        
        Args:
            anomaly_type: Type of anomaly
            description: Human-readable description
            severity: Severity score (0.0-1.0)
            metadata: Additional metadata
        """
        anomaly = {
            'type': anomaly_type,
            'description': description,
            'severity': severity,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        self.anomalies.append(anomaly)
    
    def _calculate_overall_score(self) -> float:
        """
        Calculate overall anomaly score.
        
        Returns:
            Aggregated anomaly score (0.0-1.0)
        """
        if not self.anomalies:
            return 0.0
        
        # Weight by severity
        total_severity = sum(a['severity'] for a in self.anomalies)
        
        # Normalize by count (more anomalies = higher score)
        base_score = total_severity / max(len(self.anomalies), 1)
        
        # Apply logarithmic scaling for count
        count_factor = min(math.log(len(self.anomalies) + 1) / 3, 1.0)
        
        return min(base_score * (1 + count_factor), 1.0)
    
    def _group_by_severity(self) -> Dict[str, int]:
        """Group anomalies by severity level."""
        groups = {
            'critical': 0,  # >= 0.8
            'high': 0,      # >= 0.6
            'medium': 0,    # >= 0.4
            'low': 0        # < 0.4
        }
        
        for anomaly in self.anomalies:
            severity = anomaly['severity']
            if severity >= 0.8:
                groups['critical'] += 1
            elif severity >= 0.6:
                groups['high'] += 1
            elif severity >= 0.4:
                groups['medium'] += 1
            else:
                groups['low'] += 1
        
        return groups
    
    def _group_by_type(self) -> Dict[str, int]:
        """Count anomalies by type."""
        type_counts = Counter(a['type'] for a in self.anomalies)
        return dict(type_counts)
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on detected anomalies.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        type_counts = self._group_by_type()
        severity_counts = self._group_by_severity()
        
        # Recommendations based on anomaly types
        if type_counts.get('data_inconsistency', 0) > 2:
            recommendations.append(
                "High number of data inconsistencies detected. "
                "Consider cross-referencing with additional sources or manual verification."
            )
        
        if type_counts.get('low_entropy', 0) > 0:
            recommendations.append(
                "Low entropy detected in some sources. "
                "Data may be incomplete or of poor quality. Seek additional sources."
            )
        
        if type_counts.get('temporal_anomaly', 0) > 0:
            recommendations.append(
                "Temporal anomalies detected. "
                "Consider refreshing data from sources or verifying dates."
            )
        
        if severity_counts.get('critical', 0) > 0:
            recommendations.append(
                f"{severity_counts['critical']} critical anomalies found. "
                "Manual review strongly recommended before acting on this data."
            )
        
        if not self.anomalies:
            recommendations.append(
                "No significant anomalies detected. Data appears consistent and reliable."
            )
        
        return recommendations


class PatternMatcher:
    """
    Pattern matching for identifying known OSINT patterns and signatures.
    """
    
    def __init__(self):
        """Initialize pattern matcher."""
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, List[str]]:
        """
        Load known patterns for detection.
        
        Returns:
            Dictionary of pattern types and regex patterns
        """
        return {
            'phone_formats': [
                r'\(\d{3}\)\s*\d{3}-\d{4}',  # (123) 456-7890
                r'\d{3}-\d{3}-\d{4}',         # 123-456-7890
                r'\d{10}',                     # 1234567890
            ],
            'email_patterns': [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            ],
            'ssn_patterns': [
                r'\d{3}-\d{2}-\d{4}',  # SSN format (handle carefully!)
            ],
            'address_patterns': [
                r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir)',
            ],
            'username_patterns': [
                r'@[a-zA-Z0-9_]{1,15}',  # Twitter-style
                r'[a-zA-Z0-9._]{3,30}',   # Generic username
            ]
        }
    
    def match_pattern(self, text: str, pattern_type: str) -> List[str]:
        """
        Match text against pattern type.
        
        Args:
            text: Text to search
            pattern_type: Type of pattern to match
            
        Returns:
            List of matches
        """
        if pattern_type not in self.patterns:
            return []
        
        matches = []
        for pattern in self.patterns[pattern_type]:
            matches.extend(re.findall(pattern, text))
        
        return matches
    
    def validate_data_format(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate data field formats.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            Dictionary of field: is_valid
        """
        validation = {}
        
        # Validate phone numbers
        if 'phones' in data:
            phones = data['phones'] if isinstance(data['phones'], list) else [data['phones']]
            validation['phones'] = all(
                any(re.match(p, phone) for p in self.patterns['phone_formats'])
                for phone in phones
            )
        
        # Validate emails
        if 'emails' in data or 'email' in data:
            emails = data.get('emails', []) or data.get('email', [])
            if isinstance(emails, str):
                emails = [emails]
            validation['emails'] = all(
                re.match(self.patterns['email_patterns'][0], email)
                for email in emails
            )
        
        return validation
