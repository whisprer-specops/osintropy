"""
JSON exporter for OSINT data with pretty formatting and compression options.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class JSONExporter:
    """
    Exports OSINT data to JSON format with various formatting options.
    """
    
    def __init__(self, indent: int = 2, sort_keys: bool = True):
        """
        Initialize JSON exporter.
        
        Args:
            indent: Number of spaces for indentation (None for compact)
            sort_keys: Whether to sort dictionary keys
        """
        self.indent = indent
        self.sort_keys = sort_keys
    
    def export(self, data: Dict[str, Any], filename: str, 
               add_metadata: bool = True) -> str:
        """
        Export data to JSON file.
        
        Args:
            data: Data dictionary to export
            filename: Output filename
            add_metadata: Whether to add export metadata
            
        Returns:
            Path to exported file
        """
        # Add metadata if requested
        if add_metadata:
            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'exporter': 'OSINTropy JSONExporter',
                    'version': '2.0.0'
                },
                'data': data
            }
        else:
            export_data = data
        
        # Ensure output directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    export_data,
                    f,
                    indent=self.indent,
                    sort_keys=self.sort_keys,
                    ensure_ascii=False,
                    default=str  # Convert non-serializable objects to strings
                )
            
            logger.info(f"Exported data to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            raise
    
    def export_compact(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data in compact JSON format (no indentation).
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        original_indent = self.indent
        self.indent = None
        
        try:
            result = self.export(data, filename, add_metadata=False)
            return result
        finally:
            self.indent = original_indent
    
    def export_pretty(self, data: Dict[str, Any], filename: str, 
                     indent: int = 4) -> str:
        """
        Export with extra-pretty formatting (large indentation).
        
        Args:
            data: Data to export
            filename: Output filename
            indent: Indentation level
            
        Returns:
            Path to exported file
        """
        original_indent = self.indent
        self.indent = indent
        
        try:
            result = self.export(data, filename)
            return result
        finally:
            self.indent = original_indent
    
    def to_json_string(self, data: Dict[str, Any], 
                      pretty: bool = True) -> str:
        """
        Convert data to JSON string without writing to file.
        
        Args:
            data: Data to convert
            pretty: Whether to use pretty formatting
            
        Returns:
            JSON string
        """
        return json.dumps(
            data,
            indent=self.indent if pretty else None,
            sort_keys=self.sort_keys,
            ensure_ascii=False,
            default=str
        )
    
    def export_multiple(self, data_dict: Dict[str, Dict[str, Any]], 
                       output_dir: str) -> Dict[str, str]:
        """
        Export multiple datasets to separate JSON files.
        
        Args:
            data_dict: Dictionary mapping filenames to data
            output_dir: Output directory
            
        Returns:
            Dictionary mapping filenames to output paths
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for name, data in data_dict.items():
            # Ensure .json extension
            if not name.endswith('.json'):
                name = f"{name}.json"
            
            filepath = output_dir_path / name
            output_path = self.export(data, str(filepath))
            results[name] = output_path
        
        logger.info(f"Exported {len(results)} files to {output_dir}")
        return results
    
    @staticmethod
    def load(filename: str) -> Dict[str, Any]:
        """
        Load JSON data from file.
        
        Args:
            filename: File to load
            
        Returns:
            Loaded data dictionary
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded data from {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            raise
