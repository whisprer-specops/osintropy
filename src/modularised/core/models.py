# =====================================
# core/models.py - Data Models
# =====================================
"""
Data models for the OSINT aggregator
"""

from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional
from datetime import datetime

@dataclass
class Address:
    """Structured address information"""
    full_address: str
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: str = "USA"
    first_seen: float = field(default_factory=lambda: datetime.now().timestamp())
    last_seen: float = field(default_factory=lambda: datetime.now().timestamp())
    sources: Set[str] = field(default_factory=set)

@dataclass
class PhoneNumber:
    """Structured phone number information"""
    number: str
    formatted: Optional[str] = None
    type: Optional[str] = None  # mobile, landline, voip
    carrier: Optional[str] = None
    first_seen: float = field(default_factory=lambda: datetime.now().timestamp())
    sources: Set[str] = field(default_factory=set)

@dataclass
class PersonRecord:
    """Unified person record assembled from multiple sources"""
    primary_id: str  # Hash of core identifiers
    
    # Identity information
    names: Set[str] = field(default_factory=set)
    birth_dates: Set[str] = field(default_factory=set)
    
    # Contact information
    addresses: List[Address] = field(default_factory=list)
    phone_numbers: List[PhoneNumber] = field(default_factory=list)
    emails: Set[str] = field(default_factory=set)
    
    # Relationships
    relatives: Set[str] = field(default_factory=set)
    associates: Set[str] = field(default_factory=set)
    
    # Property and legal
    property_records: List[Dict] = field(default_factory=list)
    court_records: List[Dict] = field(default_factory=list)
    
    # Digital footprint
    social_profiles: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    sources: Set[str] = field(default_factory=set)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    entropy_profile: Dict[str, float] = field(default_factory=dict)
    risk_indicators: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    last_updated: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def merge(self, other: 'PersonRecord'):
        """Merge another record into this one"""
        self.names.update(other.names)
        self.birth_dates.update(other.birth_dates)
        self.emails.update(other.emails)
        self.relatives.update(other.relatives)
        self.associates.update(other.associates)
        self.sources.update(other.sources)
        
        # Merge addresses with deduplication
        existing_addresses = {addr.full_address for addr in self.addresses}
        for addr in other.addresses:
            if addr.full_address not in existing_addresses:
                self.addresses.append(addr)
        
        # Merge phone numbers with deduplication
        existing_phones = {phone.number for phone in self.phone_numbers}
        for phone in other.phone_numbers:
            if phone.number not in existing_phones:
                self.phone_numbers.append(phone)
        
        self.last_updated = datetime.now().timestamp()