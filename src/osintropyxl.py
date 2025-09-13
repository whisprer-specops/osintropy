#!/usr/bin/env python3
"""
OSINT Web Scraping to Entropy Analysis Pipeline
Complete pipeline from Beautiful Soup scraping to anomaly detection
"""

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import time
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import math
from urllib.parse import urlparse, urljoin
import concurrent.futures
from fake_useragent import UserAgent

# For entropy analysis
from tensorflow.keras import Sequential, layers
from sklearn.preprocessing import StandardScaler

# Initialize user agent randomizer
ua = UserAgent()

@dataclass
class ScrapedProfile:
    """Container for scraped profile data"""
    url: str
    timestamp: float
    raw_html: str
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    entropy_features: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    risk_indicators: List[str] = field(default_factory=list)

class WebScrapingEngine:
    """Advanced web scraping with anti-detection measures"""
    
    def __init__(self, delay_range: Tuple[float, float] = (0.5, 2.0)):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.scraped_data = []
        
    def scrape_url(self, url: str, headers: Optional[Dict] = None) -> Optional[ScrapedProfile]:
        """Scrape a single URL with entropy-aware timing"""
        
        # Add timing jitter (entropy harvesting!)
        delay = np.random.uniform(*self.delay_range)
        time.sleep(delay)
        
        # Randomize headers if not provided
        if headers is None:
            headers = {
                'User-Agent': ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        
        try:
            # Capture request timing entropy
            start_time = time.perf_counter()
            response = self.session.get(url, headers=headers, timeout=10)
            response_time = time.perf_counter() - start_time
            
            if response.status_code == 200:
                profile = ScrapedProfile(
                    url=url,
                    timestamp=datetime.now().timestamp(),
                    raw_html=response.text
                )
                
                # Add timing entropy as a feature
                profile.entropy_features['response_time_jitter'] = response_time
                
                return profile
                
        except Exception as e:
            print(f"[!] Error scraping {url}: {str(e)}")
            return None
    
    def scrape_linkedin_style(self, profile_url: str) -> Optional[ScrapedProfile]:
        """Specialized scraper for LinkedIn-style profiles"""
        
        profile = self.scrape_url(profile_url)
        if not profile:
            return None
        
        soup = BeautifulSoup(profile.raw_html, 'html.parser')
        
        # Extract profile elements (generic approach for demo)
        profile.extracted_data = {
            'name': self._extract_text(soup, ['h1', '.profile-name', '#name']),
            'headline': self._extract_text(soup, ['h2', '.headline', '.title']),
            'location': self._extract_text(soup, ['.location', '.locality']),
            'connections': self._extract_count(soup, ['.connections', '.network-info']),
            'skills': self._extract_list(soup, ['.skill', '.skills-section li']),
            'experience': self._extract_list(soup, ['.experience', '.position']),
            'education': self._extract_list(soup, ['.education', '.school']),
            'posts': self._extract_posts(soup),
            'images': self._extract_images(soup),
            'links': self._extract_links(soup)
        }
        
        return profile
    
    def _extract_text(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Extract text from first matching selector"""
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return ""
    
    def _extract_count(self, soup: BeautifulSoup, selectors: List[str]) -> int:
        """Extract numeric count from text"""
        text = self._extract_text(soup, selectors)
        numbers = re.findall(r'\d+', text.replace(',', ''))
        return int(numbers[0]) if numbers else 0
    
    def _extract_list(self, soup: BeautifulSoup, selectors: List[str]) -> List[str]:
        """Extract list of items"""
        items = []
        for selector in selectors:
            elements = soup.select(selector)
            items.extend([e.get_text(strip=True) for e in elements])
        return items[:20]  # Limit for processing
    
    def _extract_posts(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract post/activity data"""
        posts = []
        for post in soup.select('.post, .activity, article')[:10]:
            post_data = {
                'text': post.get_text(strip=True)[:500],
                'timestamp': self._extract_timestamp(post),
                'engagement': self._extract_engagement(post)
            }
            posts.append(post_data)
        return posts
    
    def _extract_timestamp(self, element) -> float:
        """Extract and parse timestamp from element"""
        time_elements = element.select('time, .timestamp, .date')
        if time_elements:
            # Try to parse datetime attribute or text
            return datetime.now().timestamp()  # Simplified for demo
        return 0
    
    def _extract_engagement(self, element) -> int:
        """Extract engagement metrics"""
        engagement = 0
        for metric in element.select('.likes, .comments, .shares, .reactions'):
            text = metric.get_text()
            numbers = re.findall(r'\d+', text.replace(',', ''))
            if numbers:
                engagement += int(numbers[0])
        return engagement
    
    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Extract image URLs for analysis"""
        images = []
        for img in soup.select('img')[:20]:
            src = img.get('src', '')
            if src and not src.startswith('data:'):
                images.append(src)
        return images
    
    def _extract_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract external links"""
        links = []
        base_domain = urlparse(soup.get('url', '')).netloc
        
        for link in soup.select('a[href]')[:50]:
            href = link.get('href', '')
            if href and not href.startswith('#'):
                full_url = urljoin(soup.get('url', ''), href)
                if urlparse(full_url).netloc != base_domain:
                    links.append(full_url)
        return links

class ProfileEntropyAnalyzer:
    """Calculate entropy metrics from scraped profile data"""
    
    @staticmethod
    def calculate_text_entropy(text: str) -> float:
        """Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Character frequency
        freq = Counter(text.lower())
        total = len(text)
        
        entropy = 0
        for count in freq.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    @staticmethod
    def calculate_temporal_entropy(timestamps: List[float]) -> float:
        """Entropy of temporal patterns"""
        if len(timestamps) < 2:
            return 0.0
        
        # Calculate intervals
        sorted_times = sorted(timestamps)
        intervals = np.diff(sorted_times)
        
        if len(intervals) == 0:
            return 0.0
        
        # Normalize and calculate entropy
        normalized = intervals / np.sum(intervals)
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        
        return entropy
    
    @staticmethod
    def calculate_structural_entropy(profile_data: Dict) -> Dict[str, float]:
        """Extract structural entropy features"""
        features = {}
        
        # Name entropy (fake names often have patterns)
        name = profile_data.get('name', '')
        features['name_entropy'] = ProfileEntropyAnalyzer.calculate_text_entropy(name)
        features['name_length'] = len(name) / 50.0  # Normalized
        
        # Skills diversity
        skills = profile_data.get('skills', [])
        features['skills_count'] = len(skills) / 20.0  # Normalized
        features['skills_diversity'] = len(set(skills)) / (len(skills) + 1)
        
        # Experience coherence
        experience = profile_data.get('experience', [])
        features['experience_count'] = len(experience) / 10.0
        
        # Post activity patterns
        posts = profile_data.get('posts', [])
        if posts:
            post_lengths = [len(p.get('text', '')) for p in posts]
            features['post_length_variance'] = np.std(post_lengths) / 100.0 if post_lengths else 0
            
            # Engagement patterns
            engagements = [p.get('engagement', 0) for p in posts]
            features['engagement_mean'] = np.mean(engagements) / 100.0 if engagements else 0
            features['engagement_variance'] = np.std(engagements) / 50.0 if engagements else 0
        
        # Link patterns (suspicious profiles often have specific link patterns)
        links = profile_data.get('links', [])
        features['external_links'] = len(links) / 20.0
        features['unique_domains'] = len(set([urlparse(l).netloc for l in links])) / 10.0
        
        # Connection patterns
        connections = profile_data.get('connections', 0)
        features['connection_count'] = min(connections / 500.0, 2.0)  # Cap at 2.0
        
        return features
    
    @staticmethod
    def calculate_image_entropy(image_urls: List[str]) -> float:
        """Calculate entropy from image URL patterns"""
        if not image_urls:
            return 0.0
        
        # Analyze URL patterns (CDN usage, naming conventions)
        domains = [urlparse(url).netloc for url in image_urls]
        domain_entropy = len(set(domains)) / len(domains) if domains else 0
        
        # File naming patterns
        filenames = [urlparse(url).path.split('/')[-1] for url in image_urls]
        
        # Check for generated/random filenames
        random_pattern_score = 0
        for filename in filenames:
            if re.match(r'^[a-f0-9]{8,}', filename.lower()):
                random_pattern_score += 1
        
        randomness = random_pattern_score / (len(filenames) + 1)
        
        return (domain_entropy + randomness) / 2

class AnomalyDetectionEngine:
    """Autoencoder-based anomaly detection for profile data"""
    
    def __init__(self, encoding_dim: int = 4):
        self.encoding_dim = encoding_dim
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.feature_names = []
        
    def build_model(self, input_dim: int):
        """Build autoencoder for profile anomaly detection"""
        self.model = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.encoding_dim, activation='relu', name='encoding'),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def train_on_profiles(self, normal_profiles: List[Dict[str, float]], epochs: int = 100):
        """Train on normal profile patterns"""
        
        # Convert to array
        df = pd.DataFrame(normal_profiles)
        self.feature_names = df.columns.tolist()
        X = df.values
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Build and train
        if self.model is None:
            self.build_model(X.shape[1])
        
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        
        # Calculate threshold
        predictions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
        self.threshold = np.mean(mse) + 2.5 * np.std(mse)
        
        return history
    
    def detect_anomalies(self, profiles: List[Dict[str, float]]) -> List[Dict]:
        """Detect anomalies in profile data"""
        
        df = pd.DataFrame(profiles)
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction errors
        mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
        
        # Feature-level analysis
        feature_errors = np.abs(X_scaled - predictions)
        
        results = []
        for i, (profile, score) in enumerate(zip(profiles, mse)):
            # Identify top anomalous features
            top_features_idx = np.argsort(feature_errors[i])[-3:]
            suspicious_features = [self.feature_names[idx] for idx in top_features_idx]
            
            # Classify risk
            if score > self.threshold * 2:
                risk = 'CRITICAL'
            elif score > self.threshold:
                risk = 'HIGH'
            elif score > self.threshold * 0.7:
                risk = 'MEDIUM'
            else:
                risk = 'LOW'
            
            results.append({
                'anomaly_score': score,
                'risk_level': risk,
                'is_anomaly': score > self.threshold,
                'suspicious_features': suspicious_features,
                'profile_data': profile
            })
        
        return results

class OSINTPipeline:
    """Complete pipeline from scraping to anomaly detection"""
    
    def __init__(self):
        self.scraper = WebScrapingEngine()
        self.entropy_analyzer = ProfileEntropyAnalyzer()
        self.anomaly_detector = AnomalyDetectionEngine()
        self.profiles_database = []
        
    def analyze_profile(self, url: str) -> Dict:
        """Complete analysis pipeline for a single profile"""
        
        print(f"\n[*] Scraping: {url}")
        
        # Step 1: Scrape the profile
        profile = self.scraper.scrape_linkedin_style(url)
        if not profile:
            return {'error': 'Failed to scrape profile'}
        
        # Step 2: Extract entropy features
        print("[*] Calculating entropy features...")
        
        # Text entropy
        all_text = ' '.join([
            profile.extracted_data.get('name', ''),
            profile.extracted_data.get('headline', ''),
            ' '.join(profile.extracted_data.get('skills', []))
        ])
        profile.entropy_features['text_entropy'] = self.entropy_analyzer.calculate_text_entropy(all_text)
        
        # Temporal entropy from posts
        post_times = [p.get('timestamp', 0) for p in profile.extracted_data.get('posts', [])]
        if post_times:
            profile.entropy_features['temporal_entropy'] = self.entropy_analyzer.calculate_temporal_entropy(post_times)
        
        # Structural entropy
        structural_features = self.entropy_analyzer.calculate_structural_entropy(profile.extracted_data)
        profile.entropy_features.update(structural_features)
        
        # Image entropy
        profile.entropy_features['image_entropy'] = self.entropy_analyzer.calculate_image_entropy(
            profile.extracted_data.get('images', [])
        )
        
        # Step 3: Detect anomalies (if model is trained)
        if self.anomaly_detector.model:
            print("[*] Running anomaly detection...")
            results = self.anomaly_detector.detect_anomalies([profile.entropy_features])
            if results:
                result = results[0]
                profile.anomaly_score = result['anomaly_score']
                profile.risk_indicators = self._interpret_anomaly(result)
        
        # Store in database
        self.profiles_database.append(profile)
        
        return self._format_analysis_report(profile)
    
    def batch_analyze(self, urls: List[str], max_workers: int = 3) -> List[Dict]:
        """Analyze multiple profiles in parallel"""
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.analyze_profile, url): url for url in urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[!] Error analyzing {url}: {str(e)}")
                    results.append({'url': url, 'error': str(e)})
        
        return results
    
    def train_baseline(self, normal_profile_urls: List[str]):
        """Train the anomaly detector on normal profiles"""
        
        print("[*] Building baseline from normal profiles...")
        
        # Collect entropy features from normal profiles
        normal_features = []
        for url in normal_profile_urls:
            profile = self.scraper.scrape_linkedin_style(url)
            if profile:
                # Calculate entropy features
                all_text = ' '.join([
                    profile.extracted_data.get('name', ''),
                    profile.extracted_data.get('headline', ''),
                    ' '.join(profile.extracted_data.get('skills', []))
                ])
                
                features = {
                    'text_entropy': self.entropy_analyzer.calculate_text_entropy(all_text),
                    'response_time_jitter': profile.entropy_features.get('response_time_jitter', 0)
                }
                
                # Add structural features
                features.update(
                    self.entropy_analyzer.calculate_structural_entropy(profile.extracted_data)
                )
                
                # Add image entropy
                features['image_entropy'] = self.entropy_analyzer.calculate_image_entropy(
                    profile.extracted_data.get('images', [])
                )
                
                normal_features.append(features)
        
        if normal_features:
            print(f"[*] Training on {len(normal_features)} normal profiles...")
            self.anomaly_detector.train_on_profiles(normal_features)
            print("[*] Baseline established!")
    
    def _interpret_anomaly(self, anomaly_result: Dict) -> List[str]:
        """Interpret anomaly detection results"""
        
        indicators = []
        
        if anomaly_result['risk_level'] in ['HIGH', 'CRITICAL']:
            indicators.append(f"Risk Level: {anomaly_result['risk_level']}")
        
        # Interpret suspicious features
        for feature in anomaly_result['suspicious_features']:
            if 'entropy' in feature:
                indicators.append(f"Abnormal {feature.replace('_', ' ')}")
            elif 'connection' in feature:
                indicators.append("Suspicious connection patterns")
            elif 'engagement' in feature:
                indicators.append("Unusual engagement metrics")
            elif 'skills' in feature:
                indicators.append("Skill listing anomalies")
            elif 'links' in feature:
                indicators.append("Suspicious external link patterns")
        
        # Add specific warnings based on score
        score = anomaly_result['anomaly_score']
        if score > self.anomaly_detector.threshold * 3:
            indicators.append("CRITICAL: Likely fake or compromised profile")
        elif score > self.anomaly_detector.threshold * 2:
            indicators.append("WARNING: High probability of synthetic profile")
        elif score > self.anomaly_detector.threshold:
            indicators.append("NOTICE: Profile exhibits unusual patterns")
        
        return indicators
    
    def _format_analysis_report(self, profile: ScrapedProfile) -> Dict:
        """Format analysis results for reporting"""
        
        report = {
            'url': profile.url,
            'timestamp': datetime.fromtimestamp(profile.timestamp).isoformat(),
            'profile_data': {
                'name': profile.extracted_data.get('name', 'Unknown'),
                'headline': profile.extracted_data.get('headline', ''),
                'location': profile.extracted_data.get('location', ''),
                'connections': profile.extracted_data.get('connections', 0),
                'skills_count': len(profile.extracted_data.get('skills', [])),
                'experience_count': len(profile.extracted_data.get('experience', [])),
                'post_count': len(profile.extracted_data.get('posts', []))
            },
            'entropy_analysis': profile.entropy_features,
            'anomaly_detection': {
                'score': profile.anomaly_score,
                'risk_indicators': profile.risk_indicators
            }
        }
        
        return report
    
    def export_intelligence(self, filename: str = 'osint_analysis.json'):
        """Export collected intelligence to file"""
        
        intelligence = {
            'analysis_timestamp': datetime.now().isoformat(),
            'profiles_analyzed': len(self.profiles_database),
            'anomalies_detected': sum(1 for p in self.profiles_database if p.anomaly_score > 0),
            'profiles': [self._format_analysis_report(p) for p in self.profiles_database]
        }
        
        with open(filename, 'w') as f:
            json.dump(intelligence, f, indent=2)
        
        print(f"[*] Intelligence exported to {filename}")
        
        return intelligence

# Demo execution
def run_demo():
    """Demonstrate the complete OSINT pipeline"""
    
    print("=" * 70)
    print("OSINT WEB SCRAPING TO ENTROPY ANALYSIS PIPELINE")
    print("Beautiful Soup → Entropy Extraction → Anomaly Detection")
    print("=" * 70)
    
    pipeline = OSINTPipeline()
    
    # Example URLs (would be real LinkedIn profiles in production)
    # For demo, using example URLs that would need to be replaced
    demo_urls = [
        "https://example.com/profile1",  # Replace with actual URLs
        "https://example.com/profile2",
        "https://example.com/profile3"
    ]
    
    print("\n[DEMO MODE - Using simulated data]")
    print("[*] In production, replace demo_urls with actual profile URLs")
    
    # Simulate training on normal profiles
    print("\n" + "=" * 70)
    print("PHASE 1: ESTABLISHING BASELINE")
    print("=" * 70)
    
    # Generate simulated normal profiles for training
    normal_profiles = []
    for i in range(20):
        normal_profile = {
            'text_entropy': np.random.normal(4.5, 0.5),
            'response_time_jitter': np.random.normal(1.5, 0.3),
            'name_entropy': np.random.normal(3.5, 0.4),
            'skills_count': np.random.normal(0.5, 0.2),
            'connection_count': np.random.normal(0.6, 0.2),
            'engagement_mean': np.random.normal(0.3, 0.1),
            'external_links': np.random.normal(0.2, 0.1),
            'image_entropy': np.random.normal(0.5, 0.2)
        }
        normal_profiles.append(normal_profile)
    
    print(f"[*] Training baseline on {len(normal_profiles)} normal profiles...")
    pipeline.anomaly_detector.train_on_profiles(normal_profiles)
    print("[✓] Baseline established!")
    
    # Simulate analyzing suspicious profiles
    print("\n" + "=" * 70)
    print("PHASE 2: ANALYZING SUSPICIOUS PROFILES")
    print("=" * 70)
    
    suspicious_profiles = [
        {
            'name': 'Fake Profile 1',
            'text_entropy': 2.1,  # Too low - generated text
            'response_time_jitter': 0.1,  # Too consistent
            'name_entropy': 1.5,  # Fake name pattern
            'skills_count': 1.5,  # Too many skills
            'connection_count': 0.1,  # Too few connections
            'engagement_mean': 0.0,  # No engagement
            'external_links': 0.8,  # Many suspicious links
            'image_entropy': 0.1  # Stock photos
        },
        {
            'name': 'Compromised Account',
            'text_entropy': 6.2,  # Erratic text patterns
            'response_time_jitter': 3.5,  # Unusual access patterns
            'name_entropy': 3.5,  # Normal
            'skills_count': 0.5,  # Normal
            'connection_count': 0.6,  # Normal
            'engagement_mean': 1.5,  # Sudden spike
            'external_links': 1.2,  # Many new links
            'image_entropy': 0.9  # Changed images
        },
        {
            'name': 'Normal Profile',
            'text_entropy': 4.3,
            'response_time_jitter': 1.4,
            'name_entropy': 3.6,
            'skills_count': 0.4,
            'connection_count': 0.7,
            'engagement_mean': 0.3,
            'external_links': 0.2,
            'image_entropy': 0.5
        }
    ]
    
    print("\n[*] Analyzing profiles for anomalies...")
    results = pipeline.anomaly_detector.detect_anomalies(suspicious_profiles)
    
    print("\n" + "-" * 70)
    print("ANALYSIS RESULTS")
    print("-" * 70)
    
    for i, result in enumerate(results):
        profile_name = suspicious_profiles[i]['name']
        print(f"\nProfile: {profile_name}")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Is Anomaly: {result['is_anomaly']}")
        
        if result['is_anomaly']:
            print("Suspicious Features:")
            for feature in result['suspicious_features']:
                print(f"  - {feature}")
            
            # Provide interpretation
            if 'text_entropy' in result['suspicious_features']:
                print("  → Text patterns suggest generated/fake content")
            if 'response_time_jitter' in result['suspicious_features']:
                print("  → Access patterns indicate automated activity")
            if 'engagement_mean' in result['suspicious_features']:
                print("  → Engagement metrics are abnormal")
            if 'external_links' in result['suspicious_features']:
                print("  → Suspicious external link patterns detected")
    
    print("\n" + "=" * 70)
    print("INTELLIGENCE SUMMARY")
    print("=" * 70)
    
    anomaly_count = sum(1 for r in results if r['is_anomaly'])
    print(f"\nProfiles Analyzed: {len(results)}")
    print(f"Anomalies Detected: {anomaly_count}")
    print(f"Detection Rate: {(anomaly_count/len(results))*100:.1f}%")
    
    print("\n[*] Pipeline ready for real-world OSINT operations")
    print("[*] Feed in actual URLs to begin intelligence gathering")
    
    return pipeline

if __name__ == "__main__":
    pipeline = run_demo()
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    print("""
# To use with real URLs:
pipeline = OSINTPipeline()

# Train baseline on known good profiles
normal_urls = ['https://linkedin.com/in/legitimate1', ...]
pipeline.train_baseline(normal_urls)

# Analyze suspicious profiles
suspicious_urls = ['https://linkedin.com/in/suspicious1', ...]
results = pipeline.batch_analyze(suspicious_urls)

# Export intelligence report
pipeline.export_intelligence('osint_report.json')
    """)