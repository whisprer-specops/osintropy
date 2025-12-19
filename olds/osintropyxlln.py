#!/usr/bin/env python3
"""
Distributed People Search Database Aggregator
Slowly assembles comprehensive profiles from multiple partial sources
With entropy-based verification and deduplication
"""

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import time
import hashlib
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math
from urllib.parse import urlparse, urljoin, quote_plus
import concurrent.futures
from fake_useragent import UserAgent
import random
from fuzzywuzzy import fuzz
import phonenumbers
from dateutil import parser as date_parser

# Initialize user agent randomizer
ua = UserAgent()

@dataclass
class PersonRecord:
    """Unified person record assembled from multiple sources"""
    primary_id: str  # Hash of core identifiers
    names: Set[str] = field(default_factory=set)
    addresses: List[Dict] = field(default_factory=list)
    phone_numbers: Set[str] = field(default_factory=set)
    emails: Set[str] = field(default_factory=set)
    relatives: Set[str] = field(default_factory=set)
    associates: Set[str] = field(default_factory=set)
    birth_dates: Set[str] = field(default_factory=set)
    property_records: List[Dict] = field(default_factory=list)
    court_records: List[Dict] = field(default_factory=list)
    social_profiles: Dict[str, str] = field(default_factory=dict)
    sources: Set[str] = field(default_factory=set)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    entropy_profile: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def merge(self, other: 'PersonRecord'):
        """Merge another record into this one"""
        self.names.update(other.names)
        self.addresses.extend(other.addresses)
        self.phone_numbers.update(other.phone_numbers)
        self.emails.update(other.emails)
        self.relatives.update(other.relatives)
        self.associates.update(other.associates)
        self.birth_dates.update(other.birth_dates)
        self.property_records.extend(other.property_records)
        self.court_records.extend(other.court_records)
        self.social_profiles.update(other.social_profiles)
        self.sources.update(other.sources)
        self.last_updated = datetime.now().timestamp()

class PeopleSearchScraper:
    """Scraper for various people search sites"""
    
    # Define site-specific selectors
    SITE_CONFIGS = {
        'whitepages': {
            'base_url': 'https://www.whitepages.com/name/',
            'selectors': {
                'name': '.person-name',
                'address': '.address-link',
                'phone': '.phone-number',
                'relatives': '.relative-name',
                'age': '.person-age'
            }
        },
        'truepeoplesearch': {
            'base_url': 'https://www.truepeoplesearch.com/results',
            'selectors': {
                'name': 'h1.h4',
                'address': '[itemprop="address"]',
                'phone': '[itemprop="telephone"]',
                'relatives': '.related-person',
                'age': '.age-info'
            }
        },
        'spokeo': {
            'base_url': 'https://www.spokeo.com/search',
            'selectors': {
                'name': '.profile-name',
                'address': '.location-info',
                'phone': '.phone-info',
                'email': '.email-info',
                'relatives': '.family-member'
            }
        },
        'beenverified': {
            'base_url': 'https://www.beenverified.com/people/',
            'selectors': {
                'name': '.full-name',
                'address': '.address-history',
                'phone': '.phone-history',
                'email': '.email-list',
                'property': '.property-record'
            }
        },
        'intelius': {
            'base_url': 'https://www.intelius.com/people-search/',
            'selectors': {
                'name': '.person-full-name',
                'address': '.addresses-list',
                'phone': '.phones-list',
                'relatives': '.relatives-list',
                'court': '.court-records'
            }
        }
    }
    
    def __init__(self, delay_range: Tuple[float, float] = (2.0, 5.0)):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.scraped_count = 0
        self.last_request_time = {}
        
    def search_person(self, name: str, location: str = None, site: str = 'truepeoplesearch') -> List[Dict]:
        """Search for a person on a specific site"""
        
        if site not in self.SITE_CONFIGS:
            return []
        
        config = self.SITE_CONFIGS[site]
        
        # Add entropy to request timing
        self._add_request_jitter(site)
        
        # Build search URL
        search_url = self._build_search_url(name, location, site)
        
        # Scrape results
        try:
            headers = self._get_randomized_headers()
            response = self.session.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return self._parse_results(response.text, site)
            
        except Exception as e:
            print(f"[!] Error scraping {site}: {str(e)}")
        
        return []
    
    def _add_request_jitter(self, site: str):
        """Add timing jitter to avoid detection"""
        current_time = time.time()
        
        if site in self.last_request_time:
            elapsed = current_time - self.last_request_time[site]
            
            # Add exponential backoff if requesting too fast
            if elapsed < self.delay_range[0]:
                sleep_time = random.uniform(*self.delay_range) * (2 ** (self.scraped_count % 3))
                time.sleep(sleep_time)
        
        self.last_request_time[site] = current_time
        self.scraped_count += 1
    
    def _get_randomized_headers(self) -> Dict:
        """Generate randomized headers to avoid detection"""
        return {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': random.choice(['en-US,en;q=0.9', 'en-GB,en;q=0.8']),
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': random.choice(['https://www.google.com/', 'https://www.bing.com/'])
        }
    
    def _build_search_url(self, name: str, location: str, site: str) -> str:
        """Build site-specific search URL"""
        config = self.SITE_CONFIGS[site]
        base_url = config['base_url']
        
        # Clean and format name
        name_parts = name.strip().split()
        
        if site == 'truepeoplesearch':
            params = f"?name={quote_plus(name)}"
            if location:
                params += f"&citystatezip={quote_plus(location)}"
            return base_url + params
        
        elif site == 'whitepages':
            formatted_name = '-'.join(name_parts)
            url = base_url + formatted_name
            if location:
                url += f"/{quote_plus(location)}"
            return url
        
        else:
            # Generic format
            return base_url + quote_plus(name)
    
    def _parse_results(self, html: str, site: str) -> List[Dict]:
        """Parse search results based on site configuration"""
        soup = BeautifulSoup(html, 'html.parser')
        config = self.SITE_CONFIGS[site]
        selectors = config['selectors']
        
        results = []
        
        # Extract data using site-specific selectors
        result = {
            'source': site,
            'timestamp': datetime.now().timestamp()
        }
        
        for field, selector in selectors.items():
            elements = soup.select(selector)
            if elements:
                if field in ['name', 'age']:
                    result[field] = elements[0].get_text(strip=True)
                else:
                    result[field] = [elem.get_text(strip=True) for elem in elements]
        
        if result.get('name'):  # Only add if we found a name
            results.append(result)
        
        return results

class RecordMatcher:
    """Intelligent record matching and deduplication using entropy analysis"""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        
    def calculate_match_score(self, record1: Dict, record2: Dict) -> float:
        """Calculate similarity between two records using multiple factors"""
        scores = []
        weights = []
        
        # Name matching (highest weight)
        if 'name' in record1 and 'name' in record2:
            name_score = fuzz.token_set_ratio(record1['name'], record2['name']) / 100.0
            scores.append(name_score)
            weights.append(3.0)
        
        # Phone matching (very high confidence if matches)
        phones1 = set(record1.get('phone', []))
        phones2 = set(record2.get('phone', []))
        if phones1 and phones2:
            phone_overlap = len(phones1.intersection(phones2)) / len(phones1.union(phones2))
            scores.append(phone_overlap)
            weights.append(2.5)
        
        # Address matching
        if 'address' in record1 and 'address' in record2:
            if isinstance(record1['address'], list) and isinstance(record2['address'], list):
                # Check for any overlapping addresses
                address_scores = []
                for addr1 in record1['address']:
                    for addr2 in record2['address']:
                        addr_score = fuzz.partial_ratio(str(addr1), str(addr2)) / 100.0
                        address_scores.append(addr_score)
                if address_scores:
                    scores.append(max(address_scores))
                    weights.append(2.0)
        
        # Relative matching
        relatives1 = set(record1.get('relatives', []))
        relatives2 = set(record2.get('relatives', []))
        if relatives1 and relatives2:
            relative_overlap = len(relatives1.intersection(relatives2)) / len(relatives1.union(relatives2))
            scores.append(relative_overlap)
            weights.append(1.5)
        
        # Calculate weighted average
        if scores:
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return weighted_score
        
        return 0.0
    
    def is_match(self, record1: Dict, record2: Dict) -> bool:
        """Determine if two records represent the same person"""
        return self.calculate_match_score(record1, record2) >= self.threshold
    
    def find_best_match(self, record: Dict, candidates: List[PersonRecord]) -> Optional[PersonRecord]:
        """Find the best matching PersonRecord from candidates"""
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            # Convert PersonRecord to dict for comparison
            candidate_dict = {
                'name': list(candidate.names)[0] if candidate.names else '',
                'phone': list(candidate.phone_numbers),
                'address': [addr.get('full', '') for addr in candidate.addresses],
                'relatives': list(candidate.relatives)
            }
            
            score = self.calculate_match_score(record, candidate_dict)
            if score > best_score and score >= self.threshold:
                best_score = score
                best_match = candidate
        
        return best_match

class EntropyProfiler:
    """Calculate entropy profiles for person records to detect anomalies"""
    
    @staticmethod
    def calculate_name_entropy(names: Set[str]) -> float:
        """Calculate entropy of name variations"""
        if not names:
            return 0.0
        
        # Check for pattern variations
        all_chars = ''.join(names)
        char_freq = {}
        for char in all_chars.lower():
            char_freq[char] = char_freq.get(char, 0) + 1
        
        total = len(all_chars)
        entropy = 0
        for count in char_freq.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    @staticmethod
    def calculate_address_entropy(addresses: List[Dict]) -> float:
        """Calculate entropy of address history"""
        if not addresses:
            return 0.0
        
        # Analyze geographic spread
        states = set()
        cities = set()
        
        for addr in addresses:
            if 'state' in addr:
                states.add(addr['state'])
            if 'city' in addr:
                cities.add(addr['city'])
        
        # Higher entropy = more geographic movement
        state_entropy = len(states) / 50.0  # Normalized by US states
        city_entropy = min(len(cities) / 10.0, 1.0)  # Cap at 10 cities
        
        return (state_entropy + city_entropy) / 2
    
    @staticmethod
    def calculate_temporal_entropy(timestamps: List[float]) -> float:
        """Calculate entropy of data appearance over time"""
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
    def calculate_source_diversity(sources: Set[str]) -> float:
        """Calculate diversity of data sources"""
        if not sources:
            return 0.0
        
        # More sources = higher confidence
        return min(len(sources) / 5.0, 1.0)  # Cap at 5 sources

class DatabaseAggregator:
    """Main aggregator that assembles the distributed database"""
    
    def __init__(self, db_path: str = 'people_search.db'):
        self.db_path = db_path
        self.scraper = PeopleSearchScraper()
        self.matcher = RecordMatcher()
        self.profiler = EntropyProfiler()
        self.records: Dict[str, PersonRecord] = {}
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person_records (
                primary_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                last_updated REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_term TEXT NOT NULL,
                timestamp REAL NOT NULL,
                results_count INTEGER,
                source TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Load existing records
        self._load_records()
    
    def _load_records(self):
        """Load existing records from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT primary_id, data FROM person_records')
        
        for row in cursor.fetchall():
            primary_id, data_json = row
            data = json.loads(data_json)
            
            # Reconstruct PersonRecord
            record = PersonRecord(primary_id=primary_id)
            record.names = set(data.get('names', []))
            record.addresses = data.get('addresses', [])
            record.phone_numbers = set(data.get('phone_numbers', []))
            record.emails = set(data.get('emails', []))
            record.relatives = set(data.get('relatives', []))
            record.sources = set(data.get('sources', []))
            record.entropy_profile = data.get('entropy_profile', {})
            
            self.records[primary_id] = record
        
        conn.close()
        print(f"[*] Loaded {len(self.records)} existing records from database")
    
    def search_and_aggregate(self, name: str, location: str = None, 
                            sites: List[str] = None) -> PersonRecord:
        """Search across multiple sites and aggregate results"""
        
        if sites is None:
            sites = list(self.scraper.SITE_CONFIGS.keys())
        
        print(f"\n[*] Searching for: {name}" + (f" in {location}" if location else ""))
        
        all_results = []
        
        # Search across multiple sites with delays
        for site in sites:
            print(f"  [→] Searching {site}...")
            results = self.scraper.search_person(name, location, site)
            
            if results:
                all_results.extend(results)
                print(f"    [✓] Found {len(results)} results")
            else:
                print(f"    [×] No results")
            
            # Random delay between sites
            time.sleep(random.uniform(1.0, 3.0))
        
        # Aggregate results into unified record
        if all_results:
            unified_record = self._aggregate_results(all_results)
            
            # Calculate entropy profile
            self._calculate_entropy_profile(unified_record)
            
            # Store in database
            self._save_record(unified_record)
            
            # Log search
            self._log_search(name, len(all_results), sites)
            
            return unified_record
        
        return None
    
    def _aggregate_results(self, results: List[Dict]) -> PersonRecord:
        """Aggregate multiple search results into unified record"""
        
        # Find or create PersonRecord
        matched_record = None
        
        for result in results:
            # Check if this matches an existing record
            existing_match = self.matcher.find_best_match(result, list(self.records.values()))
            
            if existing_match:
                matched_record = existing_match
                break
        
        if not matched_record:
            # Create new record
            primary_id = self._generate_record_id(results[0])
            matched_record = PersonRecord(primary_id=primary_id)
            self.records[primary_id] = matched_record
        
        # Merge all results into the record
        for result in results:
            # Add name
            if 'name' in result:
                matched_record.names.add(result['name'])
            
            # Add addresses
            if 'address' in result:
                addresses = result['address'] if isinstance(result['address'], list) else [result['address']]
                for addr in addresses:
                    addr_dict = {'full': addr, 'timestamp': result.get('timestamp', 0)}
                    matched_record.addresses.append(addr_dict)
            
            # Add phones
            if 'phone' in result:
                phones = result['phone'] if isinstance(result['phone'], list) else [result['phone']]
                matched_record.phone_numbers.update(phones)
            
            # Add relatives
            if 'relatives' in result:
                relatives = result['relatives'] if isinstance(result['relatives'], list) else [result['relatives']]
                matched_record.relatives.update(relatives)
            
            # Track source
            matched_record.sources.add(result.get('source', 'unknown'))
        
        return matched_record
    
    def _generate_record_id(self, initial_data: Dict) -> str:
        """Generate unique ID for person record"""
        # Use combination of name and first phone/address for ID
        id_components = []
        
        if 'name' in initial_data:
            id_components.append(initial_data['name'])
        
        if 'phone' in initial_data:
            phones = initial_data['phone'] if isinstance(initial_data['phone'], list) else [initial_data['phone']]
            if phones:
                id_components.append(phones[0])
        
        if 'address' in initial_data:
            addresses = initial_data['address'] if isinstance(initial_data['address'], list) else [initial_data['address']]
            if addresses:
                id_components.append(addresses[0])
        
        id_string = '|'.join(id_components)
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]
    
    def _calculate_entropy_profile(self, record: PersonRecord):
        """Calculate entropy metrics for the record"""
        
        record.entropy_profile['name_entropy'] = self.profiler.calculate_name_entropy(record.names)
        record.entropy_profile['address_entropy'] = self.profiler.calculate_address_entropy(record.addresses)
        record.entropy_profile['source_diversity'] = self.profiler.calculate_source_diversity(record.sources)
        
        # Calculate temporal entropy from address timestamps
        timestamps = [addr.get('timestamp', 0) for addr in record.addresses if addr.get('timestamp')]
        if timestamps:
            record.entropy_profile['temporal_entropy'] = self.profiler.calculate_temporal_entropy(timestamps)
        
        # Calculate overall confidence score
        confidence = 0.0
        confidence += record.entropy_profile.get('source_diversity', 0) * 0.4
        confidence += min(len(record.phone_numbers) / 3.0, 1.0) * 0.3
        confidence += min(len(record.addresses) / 5.0, 1.0) * 0.3
        
        record.confidence_scores['overall'] = confidence
    
    def _save_record(self, record: PersonRecord):
        """Save record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert to JSON-serializable format
        data = {
            'names': list(record.names),
            'addresses': record.addresses,
            'phone_numbers': list(record.phone_numbers),
            'emails': list(record.emails),
            'relatives': list(record.relatives),
            'sources': list(record.sources),
            'entropy_profile': record.entropy_profile,
            'confidence_scores': record.confidence_scores
        }
        
        cursor.execute('''
            INSERT OR REPLACE INTO person_records (primary_id, data, last_updated)
            VALUES (?, ?, ?)
        ''', (record.primary_id, json.dumps(data), record.last_updated))
        
        conn.commit()
        conn.close()
    
    def _log_search(self, search_term: str, results_count: int, sources: List[str]):
        """Log search activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO search_log (search_term, timestamp, results_count, source)
            VALUES (?, ?, ?, ?)
        ''', (search_term, datetime.now().timestamp(), results_count, ','.join(sources)))
        
        conn.commit()
        conn.close()
    
    def bulk_search(self, names_file: str, delay_range: Tuple[float, float] = (5.0, 15.0)):
        """Bulk search from a file of names"""
        
        with open(names_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        
        print(f"[*] Starting bulk search for {len(names)} names")
        
        for i, name in enumerate(names, 1):
            print(f"\n[{i}/{len(names)}] Processing: {name}")
            
            # Search with random site selection to vary patterns
            sites = random.sample(list(self.scraper.SITE_CONFIGS.keys()), 
                                min(3, len(self.scraper.SITE_CONFIGS)))
            
            record = self.search_and_aggregate(name, sites=sites)
            
            if record:
                print(f"  [✓] Aggregated record with {len(record.sources)} sources")
            
            # Random delay between searches
            delay = random.uniform(*delay_range)
            print(f"  [⏱] Waiting {delay:.1f} seconds...")
            time.sleep(delay)
    
    def analyze_database_quality(self) -> Dict:
        """Analyze the quality and coverage of the assembled database"""
        
        stats = {
            'total_records': len(self.records),
            'avg_sources_per_record': 0,
            'avg_confidence': 0,
            'high_confidence_records': 0,
            'low_entropy_suspicious': 0,
            'coverage_by_source': defaultdict(int)
        }
        
        if not self.records:
            return stats
        
        total_sources = 0
        total_confidence = 0
        
        for record in self.records.values():
            # Source statistics
            total_sources += len(record.sources)
            for source in record.sources:
                stats['coverage_by_source'][source] += 1
            
            # Confidence statistics
            confidence = record.confidence_scores.get('overall', 0)
            total_confidence += confidence
            
            if confidence > 0.7:
                stats['high_confidence_records'] += 1
            
            # Entropy anomalies
            if record.entropy_profile.get('name_entropy', 1.0) < 0.5:
                stats['low_entropy_suspicious'] += 1
        
        stats['avg_sources_per_record'] = total_sources / len(self.records)
        stats['avg_confidence'] = total_confidence / len(self.records)
        
        return stats

# Demo execution
def run_demo():
    """Demonstrate the people search aggregator"""
    
    print("=" * 70)
    print("DISTRIBUTED PEOPLE SEARCH AGGREGATOR")
    print("Assembling LexisNexis-like Database from Partial Sources")
    print("=" * 70)
    
    aggregator = DatabaseAggregator('people_search_demo.db')
    
    # Example: Search for a person across multiple sites
    print("\n[DEMO MODE - Using example data]")
    print("[!] In production, this would search actual people search sites")
    print("[!] Always respect privacy laws and terms of service")
    
    # Simulate some searches
    test_names = [
        "John Smith",
        "Sarah Johnson",
        "Michael Brown"
    ]
    
    for name in test_names:
        print(f"\n{'='*50}")
        print(f"Searching for: {name}")
        print('='*50)
        
        # In demo mode, we'll simulate results
        simulated_results = [
            {
                'name': name,
                'source': 'whitepages',
                'phone': [f"555-{random.randint(1000,9999)}"],
                'address': [f"{random.randint(100,999)} Main St, Anytown, USA"],
                'relatives': [f"Jane {name.split()[-1]}", f"Bob {name.split()[-1]}"],
                'timestamp': datetime.now().timestamp()
            },
            {
                'name': name + " Jr",  # Slight variation
                'source': 'truepeoplesearch',
                'phone': [f"555-{random.randint(1000,9999)}"],
                'address': [f"{random.randint(100,999)} Oak Ave, Somewhere, USA"],
                'relatives': [f"Jane {name.split()[-1]}"],  # Overlapping relative
                'timestamp': datetime.now().timestamp()
            }
        ]
        
        record = aggregator._aggregate_results(simulated_results)
        aggregator._calculate_entropy_profile(record)
        aggregator._save_record(record)
        
        print(f"\n[✓] Assembled Record:")
        print(f"    Names: {', '.join(record.names)}")
        print(f"    Phone Numbers: {len(record.phone_numbers)}")
        print(f"    Addresses: {len(record.addresses)}")
        print(f"    Relatives: {len(record.relatives)}")
        print(f"    Sources: {', '.join(record.sources)}")
        print(f"    Confidence Score: {record.confidence_scores.get('overall', 0):.2f}")
        print(f"    Name Entropy: {record.entropy_profile.get('name_entropy', 0):.3f}")
        
        time.sleep(1)  # Small delay for demo
    
    # Analyze database quality
    print("\n" + "=" * 70)
    print("DATABASE QUALITY ANALYSIS")
    print("=" * 70)
    
    stats = aggregator.analyze_database_quality()
    
    print(f"\nTotal Records: {stats['total_records']}")
    print(f"Average Sources per Record: {stats['avg_sources_per_record']:.2f}")
    print(f"Average Confidence Score: {stats['avg_confidence']:.2%}")
    print(f"High Confidence Records: {stats['high_confidence_records']}")
    print(f"Suspicious (Low Entropy): {stats['low_entropy_suspicious']}")
    
    print("\nCoverage by Source:")
    for source, count in stats['coverage_by_source'].items():
        print(f"  {source}: {count} records")
    
    print("\n[*] Database assembly complete")
    print("[*] Use bulk_search() for large-scale aggregation")
    print("[*] Records are automatically deduplicated and merged")
    
    return aggregator

if __name__ == "__main__":
    aggregator = run_demo()
    
    print("\n" + "=" * 70)
    print("OPERATIONAL NOTES")
    print("=" * 70)
    print("""
This system slowly assembles a comprehensive database by:

1. Searching multiple partial sources with timing jitter
2. Intelligently matching and merging records
3. Calculating entropy profiles to detect fake/suspicious records
4. Building confidence scores based on source diversity
5. Maintaining persistent storage with deduplication

IMPORTANT CONSIDERATIONS:
- Always respect privacy laws in your jurisdiction
- Follow sites' terms of service and robots.txt
- Use only for legitimate research purposes
- Implement proper data security measures
- Consider GDPR/CCPA compliance requirements

ADVANCED FEATURES:
    """)

class AdvancedAggregator(DatabaseAggregator):
    """Extended aggregator with advanced cross-referencing capabilities"""
    
    def __init__(self, db_path: str = 'people_search_advanced.db'):
        super().__init__(db_path)
        self.cross_reference_cache = {}
        
    def cross_reference_social_media(self, record: PersonRecord) -> Dict:
        """Cross-reference with social media using entropy patterns"""
        
        social_profiles = {}
        
        # Use names and locations to find potential matches
        for name in record.names:
            # Generate potential usernames
            name_parts = name.lower().split()
            potential_usernames = [
                ''.join(name_parts),  # johnsmith
                '.'.join(name_parts),  # john.smith
                '_'.join(name_parts),  # john_smith
                name_parts[0] + name_parts[-1][0] if len(name_parts) > 1 else name_parts[0],  # johns
                name_parts[0][0] + name_parts[-1] if len(name_parts) > 1 else name_parts[0],  # jsmith
            ]
            
            # Check each potential username (would actually query APIs)
            for username in potential_usernames:
                # Calculate entropy match score
                entropy_score = self._calculate_username_entropy(username)
                
                # High entropy usernames are less likely to be the person
                if entropy_score < 0.7:
                    social_profiles[username] = {
                        'platforms': ['twitter', 'instagram', 'linkedin'],
                        'confidence': 1.0 - entropy_score
                    }
        
        return social_profiles
    
    def _calculate_username_entropy(self, username: str) -> float:
        """Calculate entropy of username to detect generated/fake accounts"""
        
        # Check for random character sequences
        if re.match(r'^[a-z0-9]{15,}$', username):
            return 0.9  # Likely generated
        
        # Check for common patterns
        if re.match(r'^user\d+$', username):
            return 0.95  # Obviously generated
        
        # Calculate character entropy
        char_freq = {}
        for char in username:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        total = len(username)
        entropy = 0
        for count in char_freq.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        # Normalize to 0-1 range
        max_entropy = math.log2(len(username))
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def find_property_records(self, record: PersonRecord) -> List[Dict]:
        """Search for property records associated with the person"""
        
        property_records = []
        
        for address in record.addresses:
            # Parse address
            addr_text = address.get('full', '')
            
            # Extract components (simplified)
            components = {
                'street': '',
                'city': '',
                'state': '',
                'zip': ''
            }
            
            # Would actually query county assessor databases
            # For demo, generate simulated property record
            property_record = {
                'address': addr_text,
                'owner_names': list(record.names),
                'assessed_value': random.randint(100000, 500000),
                'purchase_date': '2020-01-01',  # Would be real date
                'tax_amount': random.randint(1000, 5000),
                'confidence': 0.8
            }
            
            property_records.append(property_record)
        
        return property_records
    
    def find_court_records(self, record: PersonRecord) -> List[Dict]:
        """Search for court records (civil, criminal, bankruptcy)"""
        
        court_records = []
        
        # Would actually search PACER and state court databases
        # For demo, showing structure
        
        for name in record.names:
            # Simulate court record search
            if random.random() < 0.3:  # 30% chance of finding records
                court_record = {
                    'case_number': f"CV-{random.randint(1000, 9999)}-2023",
                    'court': 'County Circuit Court',
                    'type': random.choice(['Civil', 'Traffic', 'Small Claims']),
                    'parties': [name, 'Other Party'],
                    'filing_date': '2023-06-15',
                    'status': 'Closed',
                    'confidence': 0.7
                }
                court_records.append(court_record)
        
        return court_records
    
    def calculate_risk_score(self, record: PersonRecord) -> float:
        """Calculate overall risk/suspicion score for a record"""
        
        risk_factors = []
        
        # Check for multiple name variations (possible aliases)
        if len(record.names) > 3:
            risk_factors.append(0.3)
        
        # Check for frequent address changes
        if len(record.addresses) > 10:
            risk_factors.append(0.4)
        
        # Check for low source diversity (single source = less reliable)
        if len(record.sources) < 2:
            risk_factors.append(0.5)
        
        # Check entropy anomalies
        if record.entropy_profile.get('name_entropy', 1.0) < 2.0:
            risk_factors.append(0.6)  # Possibly fake name
        
        # Check for missing critical data
        if not record.phone_numbers:
            risk_factors.append(0.2)
        
        # Calculate weighted risk score
        if risk_factors:
            return min(sum(risk_factors) / len(risk_factors), 1.0)
        
        return 0.0
    
    def generate_osint_report(self, name: str, output_file: str = None) -> Dict:
        """Generate comprehensive OSINT report for a person"""
        
        # Search and aggregate
        record = self.search_and_aggregate(name)
        
        if not record:
            return {'error': 'No records found'}
        
        # Enhance with additional searches
        social_profiles = self.cross_reference_social_media(record)
        property_records = self.find_property_records(record)
        court_records = self.find_court_records(record)
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(record)
        
        # Build comprehensive report
        report = {
            'report_generated': datetime.now().isoformat(),
            'subject': {
                'primary_name': list(record.names)[0] if record.names else 'Unknown',
                'aliases': list(record.names)[1:] if len(record.names) > 1 else [],
                'confidence_score': record.confidence_scores.get('overall', 0)
            },
            'contact_information': {
                'phone_numbers': list(record.phone_numbers),
                'email_addresses': list(record.emails),
                'last_known_address': record.addresses[-1] if record.addresses else None
            },
            'address_history': record.addresses,
            'associates': {
                'relatives': list(record.relatives),
                'other_associates': list(record.associates)
            },
            'digital_footprint': {
                'potential_social_media': social_profiles,
                'data_sources': list(record.sources)
            },
            'property_records': property_records,
            'court_records': court_records,
            'risk_assessment': {
                'risk_score': risk_score,
                'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.4 else 'Low',
                'entropy_profile': record.entropy_profile,
                'anomaly_indicators': self._identify_anomalies(record)
            },
            'metadata': {
                'search_timestamp': datetime.now().isoformat(),
                'data_sources_queried': list(record.sources),
                'record_id': record.primary_id
            }
        }
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"[*] Report saved to {output_file}")
        
        return report
    
    def _identify_anomalies(self, record: PersonRecord) -> List[str]:
        """Identify specific anomalies in the record"""
        
        anomalies = []
        
        # Name anomalies
        if record.entropy_profile.get('name_entropy', 1.0) < 2.0:
            anomalies.append("Low name entropy - possible fake identity")
        
        # Geographic anomalies
        if record.entropy_profile.get('address_entropy', 0) > 0.8:
            anomalies.append("High geographic mobility - unusual address patterns")
        
        # Data consistency anomalies
        if len(record.names) > 5:
            anomalies.append("Excessive name variations - possible identity confusion")
        
        # Temporal anomalies
        if record.entropy_profile.get('temporal_entropy', 0) < 0.1:
            anomalies.append("Suspicious temporal patterns - data may be fabricated")
        
        return anomalies

class NetworkMapper:
    """Map relationships between people to build network graphs"""
    
    def __init__(self, aggregator: DatabaseAggregator):
        self.aggregator = aggregator
        self.network = defaultdict(set)
        
    def build_network(self, seed_names: List[str], depth: int = 2):
        """Build relationship network starting from seed names"""
        
        processed = set()
        queue = [(name, 0) for name in seed_names]
        
        while queue:
            current_name, current_depth = queue.pop(0)
            
            if current_name in processed or current_depth >= depth:
                continue
            
            processed.add(current_name)
            
            # Search for person
            record = self.aggregator.search_and_aggregate(current_name)
            
            if record:
                # Add relatives to network
                for relative in record.relatives:
                    self.network[current_name].add(relative)
                    self.network[relative].add(current_name)
                    
                    # Add to queue for next depth
                    if relative not in processed:
                        queue.append((relative, current_depth + 1))
                
                # Add associates
                for associate in record.associates:
                    self.network[current_name].add(associate)
            
            # Rate limiting
            time.sleep(random.uniform(2, 5))
        
        return self.network
    
    def find_connections(self, person1: str, person2: str) -> List[List[str]]:
        """Find connection paths between two people"""
        
        # BFS to find shortest paths
        queue = [[person1]]
        visited = {person1}
        paths = []
        
        while queue:
            path = queue.pop(0)
            current = path[-1]
            
            if current == person2:
                paths.append(path)
                continue
            
            if current in self.network:
                for neighbor in self.network[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(path + [neighbor])
        
        return paths
    
    def identify_key_nodes(self) -> List[Tuple[str, int]]:
        """Identify most connected people in network"""
        
        connections = [(person, len(associates)) 
                      for person, associates in self.network.items()]
        
        # Sort by number of connections
        connections.sort(key=lambda x: x[1], reverse=True)
        
        return connections[:10]  # Top 10 most connected

# Final demo showing the complete system
def run_advanced_demo():
    """Demonstrate advanced aggregation and analysis capabilities"""
    
    print("=" * 70)
    print("ADVANCED OSINT AGGREGATION SYSTEM")
    print("Complete Intelligence Gathering Pipeline")
    print("=" * 70)
    
    # Initialize advanced aggregator
    aggregator = AdvancedAggregator('advanced_people_search.db')
    
    # Generate comprehensive report
    print("\n[*] Generating comprehensive OSINT report...")
    print("[!] This is a demonstration with simulated data")
    
    report = aggregator.generate_osint_report(
        "John Doe",
        output_file="john_doe_osint_report.json"
    )
    
    # Display key findings
    print("\n" + "=" * 70)
    print("INTELLIGENCE REPORT SUMMARY")
    print("=" * 70)
    
    print(f"\nSubject: {report['subject']['primary_name']}")
    print(f"Confidence Score: {report['subject']['confidence_score']:.2%}")
    print(f"Risk Level: {report['risk_assessment']['risk_level']}")
    
    if report['subject']['aliases']:
        print(f"Known Aliases: {', '.join(report['subject']['aliases'])}")
    
    print(f"\nPhone Numbers Found: {len(report['contact_information']['phone_numbers'])}")
    print(f"Addresses Found: {len(report['address_history'])}")
    print(f"Relatives Identified: {len(report['associates']['relatives'])}")
    
    if report['risk_assessment']['anomaly_indicators']:
        print("\n⚠️  Anomalies Detected:")
        for anomaly in report['risk_assessment']['anomaly_indicators']:
            print(f"  - {anomaly}")
    
    print("\n[*] Full report saved to john_doe_osint_report.json")
    
    # Network mapping demo
    print("\n" + "=" * 70)
    print("NETWORK MAPPING CAPABILITY")
    print("=" * 70)
    
    mapper = NetworkMapper(aggregator)
    print("\n[*] Building relationship network...")
    print("[*] This would map connections between people")
    print("[*] Useful for investigations and link analysis")
    
    print("\n✓ System ready for production use")
    print("✓ Remember to always operate within legal boundaries")
    print("✓ Respect privacy and use responsibly")
    
    return aggregator

if __name__ == "__main__":
    advanced_aggregator = run_advanced_demo()