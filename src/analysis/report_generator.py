# =====================================
# analysis/report_generator.py
# =====================================
"""
Generate comprehensive OSINT reports
"""

import json
from datetime import datetime
from typing import Dict, Optional
from core.models import PersonRecord

class ReportGenerator:
    """Generate formatted OSINT reports"""
    
    def generate(self, record: PersonRecord) -> Dict:
        """Generate comprehensive report from person record"""
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'record_id': record.primary_id,
                'confidence_score': record.confidence_scores.get('overall', 0),
                'sources_count': len(record.sources)
            },
            
            'subject_information': {
                'names': list(record.names),
                'primary_name': list(record.names)[0] if record.names else 'Unknown',
                'aliases': list(record.names)[1:] if len(record.names) > 1 else []
            },
            
            'contact_information': {
                'phone_numbers': [
                    {
                        'number': phone.number,
                        'type': phone.type,
                        'sources': list(phone.sources)
                    }
                    for phone in record.phone_numbers
                ],
                'email_addresses': list(record.emails),
                'address_history': [
                    {
                        'address': addr.full_address,
                        'city': addr.city,
                        'state': addr.state,
                        'sources': list(addr.sources)
                    }
                    for addr in record.addresses
                ]
            },
            
            'relationships': {
                'relatives': list(record.relatives),
                'associates': list(record.associates)
            },
            
            'risk_assessment': {
                'indicators': record.risk_indicators,
                'entropy_profile': record.entropy_profile,
                'confidence_scores': record.confidence_scores
            },
            
            'data_sources': {
                'sources_used': list(record.sources),
                'created_at': datetime.fromtimestamp(record.created_at).isoformat(),
                'last_updated': datetime.fromtimestamp(record.last_updated).isoformat()
            }
        }
        
        return report
    
    def save_report(self, report: Dict, filename: str):
        """Save report to file"""
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[✓] Report saved to {filename}")
    
    def generate_summary(self, report: Dict) -> str:
        """Generate text summary of report"""
        
        lines = []
        lines.append("=" * 60)
        lines.append("OSINT INTELLIGENCE REPORT")
        lines.append("=" * 60)
        
        # Subject
        lines.append(f"\nSubject: {report['subject_information']['primary_name']}")
        
        if report['subject_information']['aliases']:
            lines.append(f"Aliases: {', '.join(report['subject_information']['aliases'])}")
        
        # Confidence
        confidence = report['report_metadata']['confidence_score']
        lines.append(f"Confidence Score: {confidence:.2%}")
        
        # Contact info
        phone_count = len(report['contact_information']['phone_numbers'])
        addr_count = len(report['contact_information']['address_history'])
        lines.append(f"\nPhone Numbers: {phone_count}")
        lines.append(f"Known Addresses: {addr_count}")
        
        # Relationships
        rel_count = len(report['relationships']['relatives'])
        assoc_count = len(report['relationships']['associates'])
        if rel_count > 0:
            lines.append(f"Known Relatives: {rel_count}")
        if assoc_count > 0:
            lines.append(f"Known Associates: {assoc_count}")
        
        # Risk indicators
        if report['risk_assessment']['indicators']:
            lines.append("\n⚠️ Risk Indicators:")
            for indicator in report['risk_assessment']['indicators']:
                lines.append(f"  - {indicator}")
        
        # Sources
        lines.append(f"\nData Sources: {', '.join(report['data_sources']['sources_used'])}")
        lines.append(f"Last Updated: {report['data_sources']['last_updated']}")
        
        return '\n'.join(lines)