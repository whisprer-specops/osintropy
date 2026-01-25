"""
Enhanced example usage script for OSINTropy.
Demonstrates all features including new scrapers, network mapping, and anomaly detection.
"""

import json
import traceback
from aggregation.aggregator import OSINTAggregator
from aggregation.network_mapper import NetworkMapper
from analysis.anomaly_detection import AnomalyDetector
from export.json_exporter import JSONExporter
from export.csv_exporter import CSVExporter
from utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_logger(__name__)


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("OSINTropy - Entropy-Based OSINT Tool")
    logger.info("Enhanced with network mapping and anomaly detection")
    logger.info("=" * 80)
    
    # Target information
    target_first_name = "Raechel"
    target_last_name = "Rukstela"
    target_location = "San Francisco, CA"
    
    # Initialize aggregator
    aggregator = OSINTAggregator(db_path='osint_data.db')
    
    logger.info(f"\nSearching for: {target_first_name} {target_last_name} in {target_location}")
    
    # Perform search
    logger.info("\n" + "-" * 80)
    logger.info("Phase 1: Data Collection")
    logger.info("-" * 80)
    
    try:
        # Your aggregator uses search_person method
        person_record = aggregator.search_person(
            first_name=target_first_name,
            last_name=target_last_name,
            location=target_location
        )
        
        logger.info(f"Search completed successfully")
        logger.info(f"Person Record Type: {type(person_record)}")
        logger.info(f"Person Record: {person_record}")
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Continue with demo using mock data
        logger.info("\nContinuing with demo using mock data...")
        person_record = None
    
    # Build aggregated data structure for analysis
    logger.info("\n" + "-" * 80)
    logger.info("Phase 2: Building Aggregated Data Structure")
    logger.info("-" * 80)
    
    # Create a mock aggregated data structure for demo purposes
    aggregated_data = {
        'sources': {
            'truepeoplesearch': {
                'records': [
                    {
                        'name': 'Daniel Asplund',
                        'age': 43,
                        'location': 'Knivsta, SWE',
                        'phones': ['305-555-1234'],
                        'relatives': ['Sister Aslplund', 'Mom Asplund']
                    }
                ],
                'entropy_score': 0.75,
                'timestamp': 1234567890.0
            },
            'whitepages': {
                'records': [
                       {'name': 'Daniel Asplund',
                        'age': 42,
                        'addresses': ['123 Main St, Knivsta SWE'],
                        'phones': ['305-555-1234', '305-555-5678']
                    }
                ],
                'entropy_score': 0.68,
                'timestamp': 1234567891.0
            }
        },
        'summary': {
            'sources_queried': 4,
            'total_records': 2,
            'average_entropy': 0.715
        }
    }
    
    logger.info(f"Using aggregated data with {len(aggregated_data['sources'])} sources")
    
    # Network mapping
    logger.info("\n" + "-" * 80)
    logger.info("Phase 3: Network Mapping")
    logger.info("-" * 80)
    
    network_mapper = NetworkMapper()
    network_graph = network_mapper.map_relationships(aggregated_data)
    
    logger.info(f"\nNetwork Graph Statistics:")
    logger.info(f"  Nodes: {network_graph['stats']['node_count']}")
    logger.info(f"  Edges: {network_graph['stats']['edge_count']}")
    logger.info(f"  Node types: {network_graph['stats']['node_types']}")
    logger.info(f"  Relationship types: {network_graph['stats']['relationship_types']}")
    
    # Find clusters
    clusters = network_mapper.find_clusters(min_connections=2)
    logger.info(f"  Detected clusters: {len(clusters)}")
    
    # Anomaly detection
    logger.info("\n" + "-" * 80)
    logger.info("Phase 4: Anomaly Detection")
    logger.info("-" * 80)
    
    detector = AnomalyDetector(sensitivity=0.7)
    anomaly_report = detector.analyze(aggregated_data)
    
    logger.info(f"\nAnomaly Detection Results:")
    logger.info(f"  Total anomalies: {anomaly_report['total_anomalies']}")
    logger.info(f"  Overall score: {anomaly_report['overall_anomaly_score']:.3f}")
    logger.info(f"  By severity: {anomaly_report['anomalies_by_severity']}")
    logger.info(f"  By type: {anomaly_report['anomalies_by_type']}")
    
    if anomaly_report['anomalies']:
        logger.info("\n  Top anomalies:")
        for anomaly in anomaly_report['anomalies'][:5]:
            logger.info(f"    [{anomaly['type']}] {anomaly['description']} (severity: {anomaly['severity']:.2f})")
    
    logger.info("\n  Recommendations:")
    for rec in anomaly_report['recommendations']:
        logger.info(f"    • {rec}")
    
    # Risk assessment
    logger.info("\n" + "-" * 80)
    logger.info("Phase 5: Risk Assessment")
    logger.info("-" * 80)
    
    risk_report = aggregator.risk_assessor.assess(aggregated_data)
    
    logger.info(f"\nRisk Assessment:")
    logger.info(f"  Risk Level: {risk_report['risk_level']}")
    logger.info(f"  Overall Confidence: {risk_report['overall_confidence']:.3f}")
    logger.info(f"  Source Confidence: {risk_report['source_confidence']:.3f}")
    logger.info(f"  Data Consistency: {risk_report['data_consistency']:.3f}")
    logger.info(f"  Coverage Score: {risk_report['coverage_score']:.3f}")
    logger.info(f"  Recency Score: {risk_report['recency_score']:.3f}")
    
    if risk_report['risk_factors']:
        logger.info(f"\n  Risk Factors:")
        for factor in risk_report['risk_factors']:
            logger.info(f"    • {factor}")
    
    logger.info(f"\n  Recommendations:")
    for rec in risk_report['recommendations']:
        logger.info(f"    • {rec}")
    
    # Export results
    logger.info("\n" + "-" * 80)
    logger.info("Phase 6: Data Export")
    logger.info("-" * 80)
    
    # JSON export
    json_exporter = JSONExporter()
    
    # Export aggregated data
    json_file = json_exporter.export(
        aggregated_data,
        filename='osint_results.json'
    )
    logger.info(f"  Exported aggregated data: {json_file}")
    
    # Export network graph
    with open('network_graph.json', 'w') as f:
        json.dump(network_graph, f, indent=2)
    logger.info(f"  Exported network graph: network_graph.json")
    
    # Export anomaly report
    with open('anomaly_report.json', 'w') as f:
        json.dump(anomaly_report, f, indent=2)
    logger.info(f"  Exported anomaly report: anomaly_report.json")
    
    # Export risk assessment
    with open('risk_assessment.json', 'w') as f:
        json.dump(risk_report, f, indent=2)
    logger.info(f"  Exported risk assessment: risk_assessment.json")
    
    # CSV export
    csv_exporter = CSVExporter()
    csv_file = csv_exporter.export(
        aggregated_data,
        filename='osint_results.csv'
    )
    logger.info(f"  Exported to CSV: {csv_file}")
    
    # Export network in Cytoscape format for visualization
    cytoscape_json = network_mapper.export_graph(format='cytoscape')
    with open('network_cytoscape.json', 'w') as f:
        f.write(cytoscape_json)
    logger.info(f"  Exported Cytoscape format: network_cytoscape.json")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("OSINTropy Analysis Complete!")
    logger.info("=" * 80)
    logger.info("\nFiles generated:")
    logger.info("  • osint_results.json - Aggregated OSINT data")
    logger.info("  • osint_results.csv - CSV format data")
    logger.info("  • network_graph.json - Network relationship graph")
    logger.info("  • network_cytoscape.json - Cytoscape visualization format")
    logger.info("  • anomaly_report.json - Anomaly detection report")
    logger.info("  • risk_assessment.json - Risk assessment report")
    logger.info("  • osint_data.db - SQLite database with all records")
    logger.info("\nAnalysis Summary:")
    logger.info(f"  • Network nodes: {network_graph['stats']['node_count']}")
    logger.info(f"  • Detected clusters: {len(clusters)}")
    logger.info(f"  • Anomalies found: {anomaly_report['total_anomalies']}")
    logger.info(f"  • Risk level: {risk_report['risk_level']}")
    logger.info(f"  • Confidence: {risk_report['overall_confidence']:.1%}")
    logger.info("\nNote: Sites returned 403 errors (anti-bot protection).")
    logger.info("Consider using proxies or API access for production use.")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"\nError occurred: {e}", exc_info=True)
