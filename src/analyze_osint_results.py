"""
OSINT Results Analyzer
Analyzes and visualizes OSINTropy output files
"""

import json
from datetime import datetime, timedelta


def analyze_results(results_dir='.'):
    """Analyze all OSINT output files."""
    
    print("=" * 80)
    print("OSINTropy Results Analysis")
    print("=" * 80)
    
    # 1. Risk Assessment
    print("\n📊 RISK ASSESSMENT")
    print("-" * 80)
    with open(f'{results_dir}/risk_assessment.json', 'r') as f:
        risk = json.load(f)
    
    print(f"Risk Level: {risk['risk_level']}")
    print(f"Overall Confidence: {risk['overall_confidence']:.1%}")
    print(f"\nBreakdown:")
    print(f"  • Source Confidence: {risk['source_confidence']:.1%}")
    print(f"  • Data Consistency: {risk['data_consistency']:.1%}")
    print(f"  • Coverage Score: {risk['coverage_score']:.1%}")
    print(f"  • Recency Score: {risk['recency_score']:.1%}")
    
    if risk['risk_factors']:
        print(f"\n⚠️  Risk Factors:")
        for factor in risk['risk_factors']:
            print(f"  • {factor}")
    
    # 2. Anomaly Report
    print("\n\n🔍 ANOMALY DETECTION")
    print("-" * 80)
    with open(f'{results_dir}/anomaly_report.json', 'r') as f:
        anomalies = json.load(f)
    
    print(f"Total Anomalies: {anomalies['total_anomalies']}")
    print(f"Anomaly Score: {anomalies['overall_anomaly_score']:.2f}")
    print(f"\nBy Severity:")
    for severity, count in anomalies['anomalies_by_severity'].items():
        if count > 0:
            print(f"  • {severity.upper()}: {count}")
    
    print(f"\nBy Type:")
    for atype, count in anomalies['anomalies_by_type'].items():
        print(f"  • {atype}: {count}")
    
    if anomalies['anomalies']:
        print(f"\n🚨 Top Anomalies:")
        for i, anomaly in enumerate(anomalies['anomalies'][:5], 1):
            print(f"\n  {i}. [{anomaly['type']}] {anomaly['description']}")
            print(f"     Severity: {anomaly['severity']:.2f}")
            if 'age_days' in anomaly['metadata']:
                days = anomaly['metadata']['age_days']
                years = days / 365.25
                print(f"     Age: {years:.1f} years old")
    
    # 3. Network Graph
    print("\n\n🕸️  NETWORK GRAPH")
    print("-" * 80)
    with open(f'{results_dir}/network_graph.json', 'r') as f:
        network = json.load(f)
    
    print(f"Nodes: {network['stats']['node_count']}")
    print(f"Edges: {network['stats']['edge_count']}")
    print(f"\nNode Types:")
    for ntype, count in network['stats']['node_types'].items():
        print(f"  • {ntype}: {count}")
    
    print(f"\nRelationship Types:")
    for rtype, count in network['stats']['relationship_types'].items():
        print(f"  • {rtype}: {count}")
    
    # Find central person
    print(f"\n👤 Central Entity:")
    for node_id, node in network['nodes'].items():
        if node['type'] == 'person' and len(node.get('sources', [])) > 1:
            print(f"  Name: {node['data']['name']}")
            print(f"  Sources: {', '.join(node['sources'])}")
            if 'age' in node['data']:
                print(f"  Age: {node['data']['age']}")
            if 'phones' in node['data']:
                print(f"  Phones: {', '.join(node['data']['phones'])}")
            if 'relatives' in node['data']:
                print(f"  Relatives: {', '.join(node['data']['relatives'])}")
            break
    
    # 4. Source Comparison
    print("\n\n📋 SOURCE COMPARISON")
    print("-" * 80)
    with open(f'{results_dir}/osint_results.json', 'r') as f:
        results = json.load(f)
    
    sources_data = results['data']['sources']
    print(f"Sources with data: {len(sources_data)}")
    print(f"Total records: {results['data']['summary']['total_records']}")
    print(f"Average entropy: {results['data']['summary']['average_entropy']:.3f}")
    
    print(f"\nPer-Source Analysis:")
    for source, data in sources_data.items():
        print(f"\n  {source.upper()}:")
        print(f"    Entropy: {data['entropy_score']:.3f}")
        print(f"    Records: {len(data['records'])}")
        
        # Show first record details
        if data['records']:
            rec = data['records'][0]
            if 'age' in rec:
                print(f"    Age: {rec['age']}")
            if 'phones' in rec:
                print(f"    Phones: {len(rec['phones'])} found")
            if 'relatives' in rec:
                print(f"    Relatives: {len(rec['relatives'])} found")
    
    # 5. Data Quality Summary
    print("\n\n✅ DATA QUALITY SUMMARY")
    print("-" * 80)
    
    # Calculate quality score
    quality_score = (
        risk['overall_confidence'] * 0.4 +
        (1.0 - anomalies['overall_anomaly_score']) * 0.3 +
        risk['data_consistency'] * 0.3
    )
    
    print(f"Overall Quality Score: {quality_score:.1%}")
    
    if quality_score >= 0.8:
        quality = "EXCELLENT"
        emoji = "🟢"
    elif quality_score >= 0.6:
        quality = "GOOD"
        emoji = "🟡"
    elif quality_score >= 0.4:
        quality = "FAIR"
        emoji = "🟠"
    else:
        quality = "POOR"
        emoji = "🔴"
    
    print(f"Quality Rating: {emoji} {quality}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if risk['recency_score'] < 0.5:
        print("  • ⚠️  Data is outdated - refresh from sources")
    
    if risk['source_confidence'] < 0.7:
        print("  • ⚠️  Low source coverage - query additional sources")
    
    if anomalies['total_anomalies'] > 5:
        print("  • ⚠️  High anomaly count - manual verification needed")
    
    cross_ref_count = sum(1 for node in network['nodes'].values() 
                         if len(node.get('sources', [])) > 1)
    if cross_ref_count < 2:
        print("  • ⚠️  Limited cross-source verification")
    
    if quality_score < 0.6:
        print("  • ⚠️  Overall quality below acceptable threshold")
    else:
        print("  • ✅ Data quality acceptable for further analysis")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    analyze_results()
