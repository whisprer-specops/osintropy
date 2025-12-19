"""
CSV exporter for OSINT data with flattening of nested structures.
"""

import csv
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class CSVExporter:
    """
    Exports OSINT data to CSV format with automatic field flattening.
    """
    
    def __init__(self, delimiter: str = ',', quoting: int = csv.QUOTE_MINIMAL):
        """
        Initialize CSV exporter.
        
        Args:
            delimiter: Field delimiter character
            quoting: CSV quoting style
        """
        self.delimiter = delimiter
        self.quoting = quoting
    
    def export(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export aggregated OSINT data to CSV.
        
        Args:
            data: Aggregated data dictionary
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        # Extract records from all sources
        all_records = self._extract_all_records(data)
        
        if not all_records:
            logger.warning("No records to export")
            return ""
        
        # Flatten records
        flattened = [self._flatten_record(record) for record in all_records]
        
        # Get all unique field names
        fieldnames = self._get_all_fields(flattened)
        
        # Write CSV
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    delimiter=self.delimiter,
                    quoting=self.quoting
                )
                
                writer.writeheader()
                writer.writerows(flattened)
            
            logger.info(f"Exported {len(flattened)} records to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise
    
    def export_by_source(self, data: Dict[str, Any], 
                        output_dir: str) -> Dict[str, str]:
        """
        Export each source to separate CSV file.
        
        Args:
            data: Aggregated data dictionary
            output_dir: Output directory
            
        Returns:
            Dictionary mapping source names to output paths
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        sources = data.get('sources', {})
        
        for source_name, source_data in sources.items():
            records = source_data.get('records', [])
            
            if records:
                # Flatten records
                flattened = [self._flatten_record(record) for record in records]
                fieldnames = self._get_all_fields(flattened)
                
                # Write CSV for this source
                filename = output_dir_path / f"{source_name}.csv"
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=fieldnames,
                        delimiter=self.delimiter,
                        quoting=self.quoting
                    )
                    
                    writer.writeheader()
                    writer.writerows(flattened)
                
                results[source_name] = str(filename)
                logger.info(f"Exported {source_name}: {len(flattened)} records")
        
        return results
    
    def _extract_all_records(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all records from aggregated data structure.
        
        Args:
            data: Aggregated data
            
        Returns:
            List of all records with source tags
        """
        all_records = []
        sources = data.get('sources', {})
        
        for source_name, source_data in sources.items():
            records = source_data.get('records', [])
            
            for record in records:
                # Add source tag
                record_copy = record.copy()
                record_copy['_source'] = source_name
                all_records.append(record_copy)
        
        return all_records
    
    def _flatten_record(self, record: Dict[str, Any], 
                       parent_key: str = '', 
                       sep: str = '_') -> Dict[str, str]:
        """
        Flatten nested dictionary structure for CSV export.
        
        Args:
            record: Record dictionary
            parent_key: Parent key for nested items
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary with string values
        """
        items = []
        
        for key, value in record.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dicts
                items.extend(self._flatten_record(value, new_key, sep).items())
            elif isinstance(value, list):
                # Convert lists to semicolon-separated strings
                if value:
                    if isinstance(value[0], dict):
                        # List of dicts - convert to JSON-like string
                        items.append((new_key, str(value)))
                    else:
                        # Simple list - join with semicolons
                        items.append((new_key, '; '.join(str(v) for v in value)))
                else:
                    items.append((new_key, ''))
            else:
                # Convert to string
                items.append((new_key, str(value) if value is not None else ''))
        
        return dict(items)
    
    def _get_all_fields(self, records: List[Dict[str, str]]) -> List[str]:
        """
        Get all unique field names from records.
        
        Args:
            records: List of flattened records
            
        Returns:
            Sorted list of field names
        """
        all_fields = set()
        
        for record in records:
            all_fields.update(record.keys())
        
        # Sort fields, but put _source first if present
        sorted_fields = sorted(all_fields)
        
        if '_source' in sorted_fields:
            sorted_fields.remove('_source')
            sorted_fields.insert(0, '_source')
        
        return sorted_fields
    
    def export_summary(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export summary statistics to CSV.
        
        Args:
            data: Aggregated data
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        summary = data.get('summary', {})
        sources = data.get('sources', {})
        
        # Create summary rows
        rows = []
        
        # Overall summary
        rows.append({
            'Metric': 'Total Sources',
            'Value': summary.get('sources_queried', 0)
        })
        rows.append({
            'Metric': 'Total Records',
            'Value': summary.get('total_records', 0)
        })
        rows.append({
            'Metric': 'Average Entropy',
            'Value': f"{summary.get('average_entropy', 0):.3f}"
        })
        
        # Per-source statistics
        for source_name, source_data in sources.items():
            rows.append({
                'Metric': f'{source_name} - Records',
                'Value': len(source_data.get('records', []))
            })
            rows.append({
                'Metric': f'{source_name} - Entropy',
                'Value': f"{source_data.get('entropy_score', 0):.3f}"
            })
        
        # Write summary CSV
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Metric', 'Value'])
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Exported summary to {output_path}")
        return str(output_path)
