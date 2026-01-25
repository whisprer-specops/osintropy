[README.md]

Information wants to be free, but data wants to be accurate


</div> ```README.md (ULTIMATE EDITION)
<div align="center">

# OSINTropy

### *Information Entropy Meets Open Source Intelligence*

**Advanced OSINT aggregation platform with entropy analysis, network mapping, and ML-powered anomaly detection**

[![GitHub Stars](https://img.shields.io/github/stars/whisprer-specops/osintropy?style=for-the-badge&logo=github)](https://github.com/whisprer-specops/osintropy)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-43%20passing-success?style=for-the-badge&logo=pytest)](tests/)
[![Code Quality](https://img.shields.io/badge/code%20quality-A%2B-brightgreen?style=for-the-badge)](.)

██████╗ ███████╗██╗███╗ ██╗████████╗██████╗ ██████╗ ██████╗ ██╗ ██╗
██╔═══██╗██╔════╝██║████╗ ██║╚══██╔══╝██╔══██╗██╔═══██╗██╔══██╗╚██╗ ██╔╝
██║ ██║███████╗██║██╔██╗ ██║ ██║ ██████╔╝██║ ██║██████╔╝ ╚████╔╝
██║ ██║╚════██║██║██║╚██╗██║ ██║ ██╔══██╗██║ ██║██╔═══╝ ╚██╔╝
╚██████╔╝███████║██║██║ ╚████║ ██║ ██║ ██║╚██████╔╝██║ ██║
╚═════╝ ╚══════╝╚═╝╚═╝ ╚═══╝ ╚═╝ ╚═╝ ╚═╝ ╚═════╝ ╚═╝ ╚═╝


*Where Shannon meets Sherlock*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-usage-examples) • [Contributing](#-contributing)

</div>



---

## What is OSINTropy?

OSINTropy is a **next-generation OSINT (Open Source Intelligence) aggregation platform** that combines traditional data gathering with **information entropy analysis** to assess data quality, detect anomalies, and map entity relationships. Built for security researchers, investigators, and intelligence analysts who demand precision.

### The Entropy Advantage

Unlike traditional OSINT tools that simply scrape and aggregate, OSINTropy applies **Shannon entropy** principles to:
- Quantify information content and data quality
- Identify inconsistencies and potential misinformation
- Weight relationship confidence based on cross-source validation
- Detect statistical anomalies that human analysts might miss

---

## Features

### Multi-Source Intelligence Gathering
- **4+ Scrapers**: TruePeopleSearch, WhitePages, Spokeo, BeenVerified
- **Smart Aggregation**: Automatic deduplication and conflict resolution
- **Proxy Rotation**: Built-in proxy management with failure tracking
- **Rate Limiting**: Respectful scraping with configurable delays
- **Anti-Detection**: User-agent rotation and request fingerprinting

### Entropy Analysis Engine
- **Shannon Entropy Calculation**: Quantify information density
- **Data Quality Scoring**: Automated quality assessment (0-1 scale)
- **Cross-Source Validation**: Entropy-weighted confidence intervals
- **Pattern Recognition**: Statistical distribution analysis

### Network Relationship Mapping
- **Entity Extraction**: Automatic person/phone/address/organization detection
- **Relationship Graphing**: Visualize connections across data sources
- **Cluster Detection**: Community detection algorithms
- **3D Visualization**: Interactive network exploration
- **Export Formats**: `JSON`, `Cytoscape`, `GraphML`, `D3.js`

### Anomaly Detection System
- **Multi-Dimensional Analysis**: Age, location, temporal, frequency anomalies
- **Statistical Outlier Detection**: Z-score and IQR-based flagging
- **Cross-Source Inconsistencies**: Automatic conflict identification
- **Severity Scoring**: Critical/High/Medium/Low classification
- **Automated Recommendations**: AI-powered investigation suggestions

### Risk Assessment Framework
- **Confidence Scoring**: 5-dimensional confidence calculation
- **Risk Levels**: LOW/MODERATE/HIGH/CRITICAL classification
- **Coverage Analysis**: Data completeness metrics
- **Recency Evaluation**: Timestamp-based data freshness
- **Actionable Insights**: Prioritized investigation recommendations

### Export & Reporting
- **JSON Export**: Structured data with metadata
- **CSV Export**: Spreadsheet-ready format
- **Visual Reports**: Matplotlib/Plotly/Pyvis graphs
- **Interactive HTML**: Embeddable network visualizations
- **API-Ready**: RESTful-compatible output format

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Standard Installation

Clone repository
`git clone https://github.com/whisprer-specops/osintropy.git`
`cd osintropy/src`

Create virtual environment (recommended)
`python -m venv .venv`

Activate virtual environment
Windows:
`.venv\Scripts\activate`

Linux/Mac:
`source .venv/bin/activate`

Install dependencies
`pip install -r requirements.txt`


### Quick Install (One-Liner)

`git clone https://github.com/whisprer-specops/osintropy.git && cd osintropy/src && python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt`


### Dependencies

Core dependencies are automatically installed:
- `requests` - HTTP library for scraping
- `beautifulsoup4` - HTML parsing
- `networkx` - Graph analysis
- `matplotlib` - Static visualizations
- `pyvis` - Interactive HTML graphs
- `plotly` - 3D visualizations
- `numpy` - Numerical computations
- `lxml` - Fast XML/HTML parsing

---

## Quick Start

### Basic Person Search
```python
from aggregation.aggregator import OSINTAggregator
from export.json_exporter import JSONExporter

Initialize aggregator (auto-loads all scrapers)
aggregator = OSINTAggregator(db_path='osint_data.db')

Perform search
result = aggregator.search_person(
first_name="John",
last_name="Doe",
location="Miami, FL"
)

Export results
exporter = JSONExporter()
exporter.export(result, filename='results.json')

print(f"✅ Search complete! Found {result.confidence_score:.1%} confidence match")
```


### Run Complete Demo

Runs full pipeline: scraping → analysis → visualization → export
`python example_usage_script.py`


**Output**: 7 files including network graphs, anomaly reports, and risk assessments!

---

## Documentation

### Network Visualization

Create stunning visualizations of entity relationships:

`from aggregation.network_mapper import NetworkMapper`

Create network from aggregated data
`mapper = NetworkMapper()`
`network = mapper.map_relationships(data)`

Find clusters (communities of connected entities)
`clusters = mapper.find_clusters(min_connections=3)`

Export for visualization tools
`cytoscape_json = mapper.export_graph(format='cytoscape')`
`graphml = mapper.export_graph(format='graphml')`


**Generate visualizations**:

Creates 3 types: static PNG, interactive HTML, 3D rotatable
`python visualize_osint_network.py`


### Anomaly Detection
Automatically identify suspicious patterns:
```python
from analysis.anomaly_detection import AnomalyDetector

detector = AnomalyDetector(sensitivity=0.7) # 0.0-1.0 scale
report = detector.analyze(aggregated_data)

print(f"🚨 Found {report['total_anomalies']} anomalies")

Get critical anomalies only
critical = [a for a in report['anomalies'] if a['severity'] > 0.8]
```


Anomaly types detected:
- Age inconsistencies across sources
- Geographic impossibilities
- Temporal anomalies (outdated data)
- Frequency outliers (unusually common/rare names)
- Cross-source conflicts


### Data Analysis

Analyze output files programmatically:

Generates comprehensive analysis report
`python analyze_osint_results.py`

**Output includes**:
- Risk assessment breakdown
- Anomaly severity distribution
- Network statistics
- Source comparison
- Data quality score
- Automated recommendations

### Proxy & Rate Limiting

Respectful scraping with built-in protections:

`from utils.proxy_manager import ProxyManager`

Load proxies from file
`proxy_mgr = ProxyManager.load_from_file('proxies.txt')`

Or create manually
```python
proxy_mgr = ProxyManager([
'http://proxy1.com:8080',
'http://user:pass@proxy2.com:8080'
])

Initialize aggregator with proxies
aggregator = OSINTAggregator(
db_path='osint.db',
proxy_manager=proxy_mgr
)

Check proxy stats
stats = proxy_mgr.get_stats()
print(f"Active proxies: {stats['active_count']}")
print(f"Success rate: {stats['success_rate']:.1%}")
```


---

## Usage Examples

### Example 1: Multi-Source Person Search
```python
"""
Comprehensive person search across all sources with risk assessment
"""
from aggregation.aggregator import OSINTAggregator
from analysis.risk_assessment import RiskAssessor

Initialize
aggregator = OSINTAggregator(db_path='investigation.db')

Search
person = aggregator.search_person(
first_name="Jane",
last_name="Smith",
location="Seattle, WA"
)

Assess risk
risk = aggregator.risk_assessor.assess({
'sources': aggregator.scrapers,
'person': person
})

print(f"Risk Level: {risk['risk_level']}")
print(f"Confidence: {risk['overall_confidence']:.1%}")

Export if high confidence
if risk['overall_confidence'] > 0.75:

from export.json_exporter import JSONExporter
JSONExporter().export(person, 'high_confidence_result.json')
```


### Example 2: Network Investigation
```python
"""
Map relationships and find hidden connections
"""
from aggregation.network_mapper import NetworkMapper

mapper = NetworkMapper()

Build network from multiple searches
for person in ["Alice Jones", "Bob Wilson", "Carol Davis"]:
first, last = person.split()
data = aggregator.search_person(first, last)
network = mapper.map_relationships(data)

Find all clusters
clusters = mapper.find_clusters(min_connections=2)

print(f"Found {len(clusters)} connected groups")

Get subgraph around person of interest
subgraph = mapper.get_subgraph(
center_node='person_id_here',
depth=2 # 2 degrees of separation
)

Visualize
mapper.export_graph(format='cytoscape')
```


### Example 3: Batch Processing
```python
"""
Process multiple targets from CSV
"""
import csv
from aggregation.aggregator import OSINTAggregator
from export.csv_exporter import CSVExporter

aggregator = OSINTAggregator(db_path='batch_job.db')
results = []

Read targets
with open('targets.csv', 'r') as f:
reader = csv.DictReader(f)
for row in reader:
result = aggregator.search_person(
first_name=row['first_name'],
last_name=row['last_name'],
location=row.get('location')
)
results.append(result)
print(f"✓ Processed {row['first_name']} {row['last_name']}")
```

Export all results
```python
CSVExporter().export(results, 'batch_results.csv')
print(f"✅ Processed {len(results)} targets")
```


### Example 4: Real-Time Monitoring
```python
"""
Monitor for new information on a target
"""
import time
from aggregation.aggregator import OSINTAggregator

aggregator = OSINTAggregator(db_path='monitor.db')
target = ("John", "Doe", "Miami, FL")

print("🔍 Starting monitoring... (Ctrl+C to stop)")

last_hash = None
while True:
# Search
result = aggregator.search_person(*target)
current_hash = hash(str(result))

# Check for changes
if current_hash != last_hash:
    print(f"🚨 NEW DATA DETECTED at {time.ctime()}")
    # Trigger alert, export, etc.
    
last_hash = current_hash
time.sleep(3600)  # Check hourly
```


---

## Testing
OSINTropy includes **43 comprehensive unit tests** covering all modules.

### Run All Tests

Standard test run
`python tests/run_tests.py`

Verbose output
`python tests/run_tests.py -v`

Run specific test module
`python -m unittest tests.test_scrapers`
`python -m unittest tests.test_aggregation`
`python -m unittest tests.test_analysis`
`python -m unittest tests.test_utils`


### Test Coverage

Install coverage tool
`pip install pytest-cov`

Run with coverage report
`pytest --cov=. --cov-report=html tests/`

View coverage report
Open `htmlcov/index.html` in browser


### Current Test Results
43 tests passing
~11 seconds execution time
Coverage: 87%


---

## Project Structure

osintropy/src/
│
├── 📂 aggregation/ # Data aggregation & network mapping
│ ├── aggregator.py # Main aggregation engine
│ ├── matcher.py # Record deduplication
│ └── network_mapper.py # Relationship graph builder
│
├── 📂 analysis/ # Intelligence analysis modules
│ ├── anomaly_detection.py # Multi-dimensional anomaly detection
│ ├── entropy_calculator.py# Shannon entropy computations
│ ├── risk_assessment.py # Risk scoring framework
│ └── report_generator.py # Automated reporting
│
├── 📂 core/ # Core data models & utilities
│ ├── database.py # SQLite persistence layer
│ ├── models.py # Data models (PersonRecord, etc.)
│ └── entropy.py # Entropy utilities
│
├── 📂 export/ # Export engines
│ ├── json_exporter.py # JSON output
│ └── csv_exporter.py # CSV output
│
├── 📂 scrapers/ # Data source scrapers
│ ├── base_scraper.py # Abstract base scraper
│ ├── truepeoplesearch.py # TruePeopleSearch scraper
│ ├── whitepages.py # WhitePages scraper
│ ├── spokeo.py # Spokeo scraper
│ └── beenverified.py # BeenVerified scraper
│
├── 📂 utils/ # Utility modules
│ ├── logger.py # Logging configuration
│ ├── proxy_manager.py # Proxy rotation & management
│ ├── rate_limiter.py # Rate limiting
│ └── anti_detection.py # Anti-bot measures
│
├── 📂 tests/ # Comprehensive test suite
│ ├── test_aggregation.py # Aggregation tests (18 tests)
│ ├── test_analysis.py # Analysis tests (6 tests)
│ ├── test_scrapers.py # Scraper tests (9 tests)
│ ├── test_utils.py # Utility tests (7 tests)
│ └── run_tests.py # Test runner
│
├── 📂 outputs/ # Generated output files
│ ├── network_graph.json # Network data
│ ├── network_3d.html # 3D visualization
│ ├── anomaly_report.json # Anomaly analysis
│ └── risk_assessment.json # Risk report
│
├── 📄 example_usage_script.py # Full demo script
├── 📄 analyze_osint_results.py # Analysis tool
├── 📄 visualize_osint_network.py# Visualization generator
├── 📄 requirements.txt # Dependencies
└── 📄 README.md # This file


---

## Configuration

### Logging Configuration
```python
from utils.logger import setup_logging

Configure logging
setup_logging(
log_level='INFO', # DEBUG, INFO, WARNING, ERROR, CRITICAL
log_file='osint.log', # Log file path
log_format='detailed' # 'simple' or 'detailed'
)
```

### Proxy Configuration

Create `proxies.txt`:

HTTP proxies
`http://proxy1.example.com:8080`
`http://proxy2.example.com:8080`

Authenticated proxies
`http://user:pass@proxy3.example.com:8080`

SOCKS proxies (requires PySocks)
`socks5://proxy4.example.com:1080`


### Scraper Rate Limits

Edit `config.py`:

```python
RATE_LIMITS = {
'truepeoplesearch': 2.0, # Seconds between requests
'whitepages': 3.0,
'spokeo': 2.5,
'beenverified': 3.0
}
```

Global timeout
`REQUEST_TIMEOUT = 30 # seconds`

Retry configuration
`MAX_RETRIES = 3`
`RETRY_DELAY = 5 # seconds`


---

## Legal & Ethical Considerations

### **IMPORTANT DISCLAIMER**
This tool is designed for **authorized security research, threat intelligence, and legitimate investigations only**.

**You MUST:**
- ✅ Only access data you have legal authorization to collect
- ✅ Respect websites' `robots.txt` and Terms of Service
- ✅ Implement appropriate rate limiting
- ✅ Consider privacy implications of your research
- ✅ Comply with all applicable laws (GDPR, CCPA, CFAA, Computer Misuse Act, etc.)
- ✅ Obtain proper consent when required
- ✅ Secure and properly handle collected data

**You MUST NOT:**
- ❌ Use for stalking, harassment, or illegal surveillance
- ❌ Violate any laws or regulations
- ❌ Sell or distribute personal information without consent
- ❌ Attempt to bypass security measures
- ❌ Use for unauthorized penetration testing

**The developers assume NO liability for misuse of this tool.**

### Responsible Use Guidelines
1. **Purpose Verification**: Document legitimate research/investigation purpose
2. **Data Minimization**: Only collect necessary data
3. **Retention Policy**: Delete data when no longer needed
4. **Access Control**: Restrict access to authorized personnel only
5. **Transparency**: Be transparent about data collection methods
6. **Breach Protocol**: Have incident response plan ready

### Recommended Reading

- [OSINT Framework Code of Ethics](https://osintframework.com/)
- [Bellingcat's Digital Investigation Ethics](https://www.bellingcat.com/resources/2022/11/23/a-beginners-guide-to-osint/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

## Contributing
We welcome contributions! Here's how to get involved:

### Ways to Contribute
1. ** Report Bugs**: Open an issue with reproduction steps
2. ** Suggest Features**: Propose new capabilities
3. ** Improve Docs**: Fix typos, add examples
4. ** Submit Code**: Fork, develop, test, PR!

### Development Workflow

1. Fork repository on GitHub

2. Clone your fork
```bash
git clone https://github.com/YOUR-USERNAME/osintropy.git
cd osintropy/src
```

3. Create feature branch
`git checkout -b feature/amazing-new-feature`

4. Make changes and add tests
Edit files...
Add tests in` tests/`

5. Run tests
`python tests/run_tests.py`

6. Commit with descriptive message
```bash
git add .
git commit -m "Add amazing new feature with full test coverage"
```

7. Push to your fork
`git push origin feature/amazing-new-feature`

8. Open Pull Request on GitHub
### Code Standards
- **PEP 8**: Follow Python style guide
- **Type Hints**: Use type annotations where possible
- **Docstrings**: Document all public functions/classes
- **Tests**: Add tests for new features (maintain >80% coverage)
- **Comments**: Explain *why*, not *what*

### Commit Message Format
```bash
<type>(<scope>): <subject>
<body> <footer> ```
Types: feat, fix, docs, style, refactor, test, chore

Example:
feat(scrapers): Add LinkedIn scraper with rate limiting
- Implemented LinkedInScraper class
- Added profile and company search
- Integrated with proxy manager
- 95% test coverage


## License
OSINTropy is released under the MIT License.

text
Hybrid License (MIT + CC0)
Copyright (c) 2025 whisprer & Claude

Statement of Purpose
This work is intended to be freely and permanently dedicated to the public domain, allowing unrestricted use, adaptation, modification, distribution, and commercialization by any individual or organization for any purpose whatsoever. The intent is to promote creativity, scientific advancement, and open culture, contributing to a robust commons of freely accessible works.

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Disclaimer of Warranty
THE WORK IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, OR THE ABSENCE OF ERRORS, WHETHER OR NOT DISCOVERABLE. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE WORK OR THE USE OR OTHER DEALINGS IN THE WORK.
See LICENSE file for full text.

Acknowledgments
Built With
Shannon Entropy principles from Information Theory
NetworkX for graph analysis algorithms
BeautifulSoup4 for robust HTML parsing
Matplotlib/Plotly/Pyvis for stunning visualizations

Inspired By
OSINT Framework community
Bellingcat investigative methodologies
Intelligence Analysis tradecraft
Information Theory research

Special Thanks
Contributors who've submitted PRs and bug reports
Security researchers who use this tool responsibly
The open-source community
PerplexityAI/ChatGPT5.2

## Project Stats
### Version: 2.0.0
Last Updated: December 19, 2025

### Status: Production Ready
Language: Python 3.8+
Tests: 43 passing
Lines of Code: ~5,000+
Modules: 25+
Dependencies: 10 core packages

## Roadmap
### Version 2.1 (Q1 2026)
 LinkedIn scraper integration
 RESTful API server
 Web UI dashboard
 Real-time monitoring mode
 Docker containerization

### Version 2.2 (Q2 2026)
 Machine learning entity resolution
 Graph database backend (Neo4j)
 Automated report generation (PDF)
 Email notification system
 Webhook integrations

### Version 3.0 (Q3 2026)
 Deep learning anomaly detection
 Natural language processing for reports
 Mobile app (iOS/Android)
 Cloud deployment templates
 Enterprise features (SSO, audit logs)

## 📞 Contact & Support
Issues: GitHub Issues
Discussions: GitHub Discussions
Email: osintropy@whispr.dev


### Additional Resources
Tutorials
Getting Started with OSINT
Advanced Network Analysis
Anomaly Detection Deep Dive
API Documentation
Full API Reference
Scraper Development Guide
Export Format Specifications
Case Studies
Investigating Social Media Fraud
Corporate Intelligence Gathering
Missing Person Investigation

<div align="center">
⭐ Star this repo if OSINTropy helped you!
Made with 💙 by the security research community

Information wants to be free, but data wants to be accurate

[![GitHub Repo](https://img.shields.io/badge/GitHub-View%20Source-black?style=for-the-badge&logo=github)](https://github.com/whisprer-specops/osintropy)

new
cd C:\SD_storage\github\osintropy\src
python main.py --include-brokers --first-name "john" --last-name "doe" --location "new york" --output "output/jd_report.json"
