# =====================================
# main.py - Main Entry Point
# =====================================

Main entry point for the OSINT Aggregator


from aggregation.aggregator import OSINTAggregator
from analysis.report_generator import ReportGenerator

def main()
    Main function
    
    # Initialize aggregator
    aggregator = OSINTAggregator()
    
    # Example search
    record = aggregator.search_person(
        first_name=John,
        last_name=Smith,
        location=New York
    )
    
    if record
        # Generate report
        report_gen = ReportGenerator()
        report = report_gen.generate(record)
        
        # Save report
        report_gen.save_report(report, outputjohn_smith_report.json)
        
        print(fFound record with {len(record.sources)} sources)
        print(fConfidence {record.confidence_scores.get('overall', 0).2%})
    else
        print(No records found)

if __name__ == __main__
    main()