# =====================================
# aggregation/aggregator.py - Main Aggregator
# =====================================
"""
Main aggregation engine that combines all components
"""

import hashlib
from typing import Dict, List, Optional
from core.models import PersonRecord, Address, PhoneNumber
from core.database import Database
from scrapers import get_scraper
from aggregation.matcher import RecordMatcher
from analysis.risk_assessment import RiskAssessor
from utils.rate_limiter import RateLimiter
from config import Config

class OSINTAggregator:
    """Main aggregation engine"""
    
    def __init__(self, db_path: str = None):
        self.db = Database(db_path or Config.DATABASE_PATH)
        self.matcher = RecordMatcher()
        self.risk_assessor = RiskAssessor()
        self.rate_limiter = RateLimiter()
        
        # Initialize scrapers
        self.scrapers = self._initialize_scrapers()
    
    def _initialize_scrapers(self) -> Dict:
        """Initialize all available scrapers"""
        scrapers = {}
        
        for site in ['truepeoplesearch', 'whitepages', 'spokeo', 'beenverified']:
            try:
                scrapers[site] = get_scraper(site)
            except Exception as e:
                print(f"Failed to initialize {site} scraper: {e}")
        
        return scrapers
    
    def search_person(self, first_name: str, last_name: str, 
                     location: str = None, sites: List[str] = None) -> PersonRecord:
        """Search for a person across multiple sites"""
        
        if sites is None:
            sites = list(self.scrapers.keys())
        
        all_results = []
        
        for site in sites:
            if site not in self.scrapers:
                continue
            
            # Rate limiting
            self.rate_limiter.wait_if_needed(site)
            
            # Perform search
            scraper = self.scrapers[site]
            results = scraper.search(
                first_name=first_name,
                last_name=last_name,
                location=location
            )
            
            all_results.extend(results)
        
        # Aggregate results
        if all_results:
            record = self._aggregate_results(all_results)
            
            # Calculate risk score
            record.risk_indicators = self.risk_assessor.assess(record)
            
            # Save to database
            self.db.save_record(record)
            
            return record
        
        return None
    
    def _aggregate_results(self, results: List[Dict]) -> PersonRecord:
        """Aggregate search results into unified record"""
        
        # Find or create record
        record = self._find_or_create_record(results)
        
        # Merge all results
        for result in results:
            self._merge_result_into_record(result, record)
        
        return record
    
    def _find_or_create_record(self, results: List[Dict]) -> PersonRecord:
        """Find existing record or create new one"""
        
        # Check for existing match in database
        for result in results:
            existing = self.db.find_similar_records(result)
            if existing:
                best_match = self.matcher.find_best_match(result, existing)
                if best_match:
                    return best_match
        
        # Create new record
        primary_id = self._generate_record_id(results[0])
        return PersonRecord(primary_id=primary_id)
    
    def _generate_record_id(self, data: Dict) -> str:
        """Generate unique ID for record"""
        components = []
        
        if 'name' in data:
            components.append(data['name'])
        if 'phone' in data:
            components.append(str(data['phone']))
        
        id_string = '|'.join(components)
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]
    
    def _merge_result_into_record(self, result: Dict, record: PersonRecord):
        """Merge a single result into the record"""
        
        # Add name
        if 'name' in result:
            record.names.add(result['name'])
        
        # Add address
        if 'address' in result:
            addr = Address(
                full_address=result['address'],
                sources={result.get('source', 'unknown')}
            )
            record.addresses.append(addr)
        
        # Add phone
        if 'phone' in result:
            phone = PhoneNumber(
                number=result['phone'],
                sources={result.get('source', 'unknown')}
            )
            record.phone_numbers.append(phone)
        
        # Track source
        record.sources.add(result.get('source', 'unknown'))