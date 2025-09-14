# =====================================
# Example usage script
# =====================================
"""
Example of how to use the modular OSINT aggregator
"""

if __name__ == "__main__":
    from aggregation.aggregator import OSINTAggregator
    from analysis.report_generator import ReportGenerator
    
    # Initialize the aggregator
    print("[*] Initializing OSINT Aggregator...")
    aggregator = OSINTAggregator(db_path="osint_data.db")
    
    # Perform a search
    print("[*] Searching for person across multiple sites...")
    record = aggregator.search_person(
        first_name="John",
        last_name="Smith",
        location="New York, NY",
        sites=['truepeoplesearch']  # Start with one site for testing
    )
    
    if record:
        print(f"[✓] Found record with {len(record.sources)} sources")
        
        # Generate report
        report_gen = ReportGenerator()
        report = report_gen.generate(record)
        
        # Print summary
        summary = report_gen.generate_summary(report)
        print("\n" + summary)
        
        # Save full report
        report_gen.save_report(report, "john_smith_report.json")
    else:
        print("[×] No records found")