# =====================================
# core/database.py
# =====================================
"""
Database operations for persistent storage
"""

import sqlite3
import json
from typing import List, Optional, Dict
from core.models import PersonRecord, Address, PhoneNumber
from datetime import datetime

class Database:
    """SQLite database handler"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Person records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS person_records (
                    primary_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_updated REAL NOT NULL
                )
            ''')
            
            # Search log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_term TEXT NOT NULL,
                    location TEXT,
                    timestamp REAL NOT NULL,
                    results_count INTEGER,
                    sources TEXT
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_person_updated 
                ON person_records(last_updated)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_search_timestamp 
                ON search_log(timestamp)
            ''')
            
            conn.commit()
    
    def save_record(self, record: PersonRecord):
        """Save or update a person record"""
        
        # Convert record to JSON-serializable format
        data = self._record_to_dict(record)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO person_records 
                (primary_id, data, created_at, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (
                record.primary_id,
                json.dumps(data),
                record.created_at,
                record.last_updated
            ))
            
            conn.commit()
    
    def get_record(self, primary_id: str) -> Optional[PersonRecord]:
        """Retrieve a record by ID"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT data FROM person_records 
                WHERE primary_id = ?
            ''', (primary_id,))
            
            row = cursor.fetchone()
            
            if row:
                data = json.loads(row[0])
                return self._dict_to_record(data, primary_id)
        
        return None
    
    def find_similar_records(self, search_data: Dict) -> List[PersonRecord]:
        """Find potentially matching records"""
        
        records = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all records (in production, would filter more intelligently)
            cursor.execute('''
                SELECT primary_id, data 
                FROM person_records 
                ORDER BY last_updated DESC 
                LIMIT 100
            ''')
            
            for row in cursor.fetchall():
                primary_id, data_json = row
                data = json.loads(data_json)
                
                # Basic name matching (would be more sophisticated)
                if 'name' in search_data and 'names' in data:
                    search_name = search_data['name'].lower()
                    for name in data['names']:
                        if search_name in name.lower() or name.lower() in search_name:
                            record = self._dict_to_record(data, primary_id)
                            records.append(record)
                            break
        
        return records
    
    def log_search(self, search_term: str, location: str = None,
                   results_count: int = 0, sources: List[str] = None):
        """Log a search operation"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO search_log 
                (search_term, location, timestamp, results_count, sources)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                search_term,
                location,
                datetime.now().timestamp(),
                results_count,
                ','.join(sources) if sources else ''
            ))
            
            conn.commit()
    
    def _record_to_dict(self, record: PersonRecord) -> Dict:
        """Convert PersonRecord to dictionary"""
        
        return {
            'names': list(record.names),
            'birth_dates': list(record.birth_dates),
            'addresses': [
                {
                    'full_address': addr.full_address,
                    'city': addr.city,
                    'state': addr.state,
                    'sources': list(addr.sources)
                }
                for addr in record.addresses
            ],
            'phone_numbers': [
                {
                    'number': phone.number,
                    'type': phone.type,
                    'sources': list(phone.sources)
                }
                for phone in record.phone_numbers
            ],
            'emails': list(record.emails),
            'relatives': list(record.relatives),
            'associates': list(record.associates),
            'sources': list(record.sources),
            'confidence_scores': record.confidence_scores,
            'entropy_profile': record.entropy_profile,
            'risk_indicators': record.risk_indicators
        }
    
    def _dict_to_record(self, data: Dict, primary_id: str) -> PersonRecord:
        """Convert dictionary to PersonRecord"""
        
        record = PersonRecord(primary_id=primary_id)
        
        record.names = set(data.get('names', []))
        record.birth_dates = set(data.get('birth_dates', []))
        record.emails = set(data.get('emails', []))
        record.relatives = set(data.get('relatives', []))
        record.associates = set(data.get('associates', []))
        record.sources = set(data.get('sources', []))
        
        # Reconstruct addresses
        for addr_data in data.get('addresses', []):
            addr = Address(
                full_address=addr_data['full_address'],
                city=addr_data.get('city'),
                state=addr_data.get('state'),
                sources=set(addr_data.get('sources', []))
            )
            record.addresses.append(addr)
        
        # Reconstruct phone numbers
        for phone_data in data.get('phone_numbers', []):
            phone = PhoneNumber(
                number=phone_data['number'],
                type=phone_data.get('type'),
                sources=set(phone_data.get('sources', []))
            )
            record.phone_numbers.append(phone)
        
        record.confidence_scores = data.get('confidence_scores', {})
        record.entropy_profile = data.get('entropy_profile', {})
        record.risk_indicators = data.get('risk_indicators', [])
        
        return record