"""
Quick diagnostic script to check project structure and missing files.
"""

import os
from pathlib import Path

# Files that should exist
REQUIRED_FILES = {
    'scrapers': [
        '__init__.py',
        'base_scraper.py',
        'truepeoplesearch.py',
        'whitepages.py',
        'spokeo.py',
        'beenverified.py'
    ],
    'aggregation': [
        '__init__.py',
        'aggregator.py',
        'network_mapper.py'
    ],
    'analysis': [
        '__init__.py',
        'entropy_calculator.py',
        'anomaly_detection.py'
    ],
    'utils': [
        '__init__.py',
        'logger.py',
        'proxy_manager.py'
    ],
    'export': [
        '__init__.py',
        'json_exporter.py',
        'csv_exporter.py'
    ],
    'tests': [
        '__init__.py',
        'test_scrapers.py',
        'test_aggregation.py',
        'test_analysis.py',
        'test_utils.py',
        'run_tests.py'
    ]
}

def check_files():
    """Check which required files exist."""
    print("=" * 80)
    print("OSINTropy Structure Check")
    print("=" * 80)
    
    missing_files = []
    existing_files = []
    
    for directory, files in REQUIRED_FILES.items():
        print(f"\n{directory}/")
        for filename in files:
            filepath = Path(directory) / filename
            exists = filepath.exists()
            status = "✓" if exists else "✗"
            
            if exists:
                existing_files.append(str(filepath))
                print(f"  {status} {filename}")
            else:
                missing_files.append(str(filepath))
                print(f"  {status} {filename} (MISSING)")
    
    print("\n" + "=" * 80)
    print(f"Summary: {len(existing_files)} files found, {len(missing_files)} missing")
    print("=" * 80)
    
    if missing_files:
        print("\nMissing files:")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("\n✓ All required files present!")
    
    return len(missing_files) == 0

if __name__ == '__main__':
    all_good = check_files()
    exit(0 if all_good else 1)
