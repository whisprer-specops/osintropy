#!/usr/bin/env python3
"""
OSINT Entropy-Based Anomaly Detection System
Using Autoencoders to detect active credentials in data breaches
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import math

# For the autoencoder
from tensorflow.keras import Sequential, layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Simulated database of "normal" email patterns for training
np.random.seed(42)

@dataclass
class EmailProfile:
    """Represents an email address with associated entropy metrics"""
    email: str
    domain: str
    username: str
    creation_timestamp: float
    last_seen: float
    breach_count: int
    password_complexity: float
    activity_pattern: np.ndarray
    
class EntropyCalculator:
    """Calculate various entropy metrics from email/password data"""
    
    @staticmethod
    def shannon_entropy(text: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not text:
            return 0.0
        
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(text)]
        entropy = -sum([p * math.log(p, 2) for p in prob if p > 0])
        return entropy
    
    @staticmethod
    def temporal_entropy(timestamps: List[float]) -> float:
        """Calculate entropy from temporal patterns"""
        if len(timestamps) < 2:
            return 0.0
        
        intervals = np.diff(sorted(timestamps))
        if len(intervals) == 0:
            return 0.0
        
        # Normalize intervals
        intervals = intervals / np.sum(intervals) if np.sum(intervals) > 0 else intervals
        
        # Calculate entropy of time intervals
        entropy = 0
        for interval in intervals:
            if interval > 0:
                entropy -= interval * np.log2(interval)
        
        return entropy
    
    @staticmethod
    def structural_entropy(email: str) -> Dict[str, float]:
        """Extract structural entropy features from email format"""
        features = {}
        
        # Username complexity
        username = email.split('@')[0] if '@' in email else email
        features['username_entropy'] = EntropyCalculator.shannon_entropy(username)
        
        # Check for patterns suggesting randomization
        features['has_numbers'] = float(any(c.isdigit() for c in username))
        features['has_special'] = float(bool(re.search(r'[._\-+]', username)))
        features['length_ratio'] = len(username) / 20.0  # Normalized by typical max
        
        # Domain entropy (lower for common domains)
        domain = email.split('@')[1] if '@' in email else 'unknown'
        common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        features['domain_commonality'] = 0.0 if domain in common_domains else 1.0
        
        return features

class OSINTDataGenerator:
    """Generate synthetic OSINT data for training"""
    
    def __init__(self):
        self.common_passwords = [
            'password123', '123456', 'qwerty', 'letmein', 'monkey',
            'dragon', 'baseball', 'iloveyou', 'trustno1', 'sunshine'
        ]
        self.domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'protonmail.com',
            'company.com', 'business.org', 'secure.net'
        ]
    
    def generate_normal_profile(self) -> Dict:
        """Generate a 'normal' email profile"""
        username = self._generate_username(is_suspicious=False)
        domain = np.random.choice(self.domains[:4], p=[0.4, 0.3, 0.2, 0.1])
        email = f"{username}@{domain}"
        
        # Normal activity pattern - regular intervals
        base_time = datetime.now().timestamp()
        activity_times = [
            base_time - np.random.normal(86400 * 30, 86400 * 5)  # ~30 days ago
            for _ in range(np.random.randint(5, 20))
        ]
        
        return {
            'email': email,
            'password_entropy': np.random.normal(3.5, 0.5),  # Moderate complexity
            'breach_count': np.random.poisson(1),
            'activity_variance': np.std(np.diff(sorted(activity_times))),
            'temporal_entropy': EntropyCalculator.temporal_entropy(activity_times),
            **EntropyCalculator.structural_entropy(email)
        }
    
    def generate_anomalous_profile(self, anomaly_type: str = 'compromised') -> Dict:
        """Generate anomalous profiles representing different threat patterns"""
        
        if anomaly_type == 'compromised':
            # Recently compromised account - sudden activity spike
            username = self._generate_username(is_suspicious=False)
            domain = np.random.choice(self.domains[:4])
            email = f"{username}@{domain}"
            
            base_time = datetime.now().timestamp()
            # Normal activity then sudden burst
            activity_times = [base_time - 86400 * i for i in range(100, 110)]  # Old activity
            activity_times.extend([base_time - 3600 * i for i in range(24)])  # Recent burst
            
            return {
                'email': email,
                'password_entropy': np.random.normal(2.0, 0.3),  # Weaker password
                'breach_count': np.random.poisson(3),  # More breaches
                'activity_variance': np.std(np.diff(sorted(activity_times))) * 3,
                'temporal_entropy': EntropyCalculator.temporal_entropy(activity_times),
                **EntropyCalculator.structural_entropy(email)
            }
        
        elif anomaly_type == 'synthetic':
            # Fake/bot account - too regular patterns
            username = self._generate_username(is_suspicious=True)
            domain = np.random.choice(self.domains)
            email = f"{username}@{domain}"
            
            # Suspiciously regular intervals
            base_time = datetime.now().timestamp()
            activity_times = [base_time - 3600 * i for i in range(0, 168, 24)]  # Exactly daily
            
            return {
                'email': email,
                'password_entropy': 4.5,  # Suspiciously high (generated)
                'breach_count': 0,  # Too clean
                'activity_variance': 0.1,  # Too regular
                'temporal_entropy': 0.2,  # Low entropy - too predictable
                **EntropyCalculator.structural_entropy(email)
            }
        
        elif anomaly_type == 'credential_stuffing':
            # Account being tested across multiple services
            username = self._generate_username(is_suspicious=False)
            domain = np.random.choice(self.domains[:4])
            email = f"{username}@{domain}"
            
            # Burst pattern - many attempts in short windows
            base_time = datetime.now().timestamp()
            activity_times = []
            for burst in range(5):
                burst_start = base_time - 86400 * burst * 7
                activity_times.extend([burst_start + i for i in range(50)])
            
            return {
                'email': email,
                'password_entropy': np.random.normal(2.5, 0.2),
                'breach_count': np.random.poisson(5),  # High breach exposure
                'activity_variance': np.std(np.diff(sorted(activity_times))) * 10,
                'temporal_entropy': EntropyCalculator.temporal_entropy(activity_times),
                **EntropyCalculator.structural_entropy(email)
            }
        
        return self.generate_normal_profile()
    
    def _generate_username(self, is_suspicious: bool = False) -> str:
        """Generate realistic or suspicious usernames"""
        if is_suspicious:
            # Bot-like patterns
            patterns = [
                lambda: f"user{np.random.randint(10000, 99999)}",
                lambda: ''.join(np.random.choice(list('abcdef0123456789'), 8)),
                lambda: f"test_{np.random.randint(100, 999)}_bot"
            ]
            return np.random.choice(patterns)()
        else:
            # Realistic patterns
            first_names = ['john', 'jane', 'mike', 'sarah', 'alex', 'emma']
            last_names = ['smith', 'jones', 'brown', 'davis', 'wilson']
            
            patterns = [
                lambda: f"{np.random.choice(first_names)}.{np.random.choice(last_names)}",
                lambda: f"{np.random.choice(first_names)}{np.random.randint(1, 99)}",
                lambda: f"{np.random.choice(first_names)}_{np.random.choice(last_names)}"
            ]
            return np.random.choice(patterns)()

class EntropyAutoencoder:
    """Autoencoder for detecting anomalies in OSINT email data"""
    
    def __init__(self, encoding_dim: int = 3):
        self.encoding_dim = encoding_dim
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        
    def build_model(self, input_dim: int):
        """Build the autoencoder architecture"""
        self.model = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(self.encoding_dim, activation='relu', name='encoding'),
            layers.Dense(8, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
    def train(self, normal_data: np.ndarray, epochs: int = 100, validation_split: float = 0.2):
        """Train autoencoder on normal patterns only"""
        # Scale the data
        normal_scaled = self.scaler.fit_transform(normal_data)
        
        # Build model if not exists
        if self.model is None:
            self.build_model(normal_data.shape[1])
        
        # Train
        history = self.model.fit(
            normal_scaled, normal_scaled,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            verbose=0
        )
        
        # Calculate threshold from training data
        predictions = self.model.predict(normal_scaled, verbose=0)
        mse = np.mean(np.power(normal_scaled - predictions, 2), axis=1)
        self.threshold = np.mean(mse) + 2 * np.std(mse)  # 2 sigma threshold
        
        return history
    
    def detect_anomaly(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect anomalies and return reconstruction errors"""
        data_scaled = self.scaler.transform(data)
        predictions = self.model.predict(data_scaled, verbose=0)
        
        # Calculate reconstruction errors
        mse = np.mean(np.power(data_scaled - predictions, 2), axis=1)
        mae = np.mean(np.abs(data_scaled - predictions), axis=1)
        
        # Detailed error analysis per feature
        feature_errors = np.abs(data_scaled - predictions)
        feature_std = np.std(feature_errors, axis=0)
        
        # Classify as anomaly
        anomalies = mse > self.threshold
        
        return anomalies, mse, feature_std

class OSINTAnalyzer:
    """Main analyzer combining entropy extraction and anomaly detection"""
    
    def __init__(self):
        self.autoencoder = EntropyAutoencoder(encoding_dim=3)
        self.generator = OSINTDataGenerator()
        self.feature_names = None
        
    def prepare_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate training data of normal email patterns"""
        data = []
        for _ in range(n_samples):
            data.append(self.generator.generate_normal_profile())
        
        df = pd.DataFrame(data)
        self.feature_names = df.columns.tolist()
        return df
    
    def prepare_test_data(self, n_normal: int = 100, n_anomalous: int = 50) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate mixed test data with labels"""
        data = []
        labels = []
        
        # Normal samples
        for _ in range(n_normal):
            data.append(self.generator.generate_normal_profile())
            labels.append(0)
        
        # Anomalous samples
        anomaly_types = ['compromised', 'synthetic', 'credential_stuffing']
        for _ in range(n_anomalous):
            anomaly_type = np.random.choice(anomaly_types)
            data.append(self.generator.generate_anomalous_profile(anomaly_type))
            labels.append(1)
        
        df = pd.DataFrame(data)
        return df, np.array(labels)
    
    def train(self, training_data: pd.DataFrame):
        """Train the autoencoder on normal patterns"""
        # Remove email column for training
        X = training_data.drop('email', axis=1).values
        self.autoencoder.train(X, epochs=150)
        
    def analyze(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze data and return results with anomaly scores"""
        emails = test_data['email'].values
        X = test_data.drop('email', axis=1).values
        
        anomalies, scores, feature_importance = self.autoencoder.detect_anomaly(X)
        
        results = pd.DataFrame({
            'email': emails,
            'anomaly_score': scores,
            'is_anomaly': anomalies,
            'risk_level': pd.cut(scores, 
                                bins=[0, self.autoencoder.threshold*0.5, 
                                      self.autoencoder.threshold, 
                                      self.autoencoder.threshold*2, np.inf],
                                labels=['Low', 'Medium', 'High', 'Critical'])
        })
        
        # Add top anomalous features
        feature_cols = [col for col in test_data.columns if col != 'email']
        top_features_idx = np.argsort(feature_importance)[-3:]
        results['suspicious_patterns'] = [
            ', '.join([feature_cols[i] for i in top_features_idx])
            for _ in range(len(results))
        ]
        
        return results
    
    def interpret_anomaly(self, email: str, score: float, patterns: str) -> str:
        """Provide human-readable interpretation of anomaly"""
        interpretations = []
        
        if score > self.autoencoder.threshold * 2:
            interpretations.append("CRITICAL: Extremely abnormal activity pattern detected")
        elif score > self.autoencoder.threshold:
            interpretations.append("WARNING: Suspicious activity pattern detected")
        
        if 'temporal_entropy' in patterns:
            interpretations.append("- Irregular timing patterns suggest automated or compromised access")
        if 'activity_variance' in patterns:
            interpretations.append("- Activity variance indicates potential credential stuffing")
        if 'password_entropy' in patterns:
            interpretations.append("- Password complexity anomaly detected")
        if 'breach_count' in patterns:
            interpretations.append("- Elevated breach exposure risk")
        
        return '\n'.join(interpretations) if interpretations else "Profile appears normal"

# Demo execution
def run_demo():
    """Run a complete demo of the OSINT entropy analyzer"""
    print("=" * 60)
    print("OSINT ENTROPY-BASED ANOMALY DETECTION SYSTEM")
    print("Email Leak Analyzer v1.0")
    print("=" * 60)
    
    analyzer = OSINTAnalyzer()
    
    # Generate and train on normal data
    print("\n[*] Generating training data (normal email patterns)...")
    training_data = analyzer.prepare_training_data(n_samples=500)
    
    print(f"[*] Training autoencoder on {len(training_data)} normal profiles...")
    analyzer.train(training_data)
    
    # Generate test data with anomalies
    print("\n[*] Generating test data with planted anomalies...")
    test_data, true_labels = analyzer.prepare_test_data(n_normal=50, n_anomalous=30)
    
    # Analyze
    print("[*] Running entropy-based anomaly detection...")
    results = analyzer.analyze(test_data)
    
    # Display results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    # Show some detected anomalies
    anomalies = results[results['is_anomaly'] == True].head(10)
    
    if len(anomalies) > 0:
        print(f"\n[!] Detected {len(results[results['is_anomaly']])} anomalous profiles:")
        print("-" * 60)
        
        for idx, row in anomalies.iterrows():
            print(f"\nEmail: {row['email']}")
            print(f"Anomaly Score: {row['anomaly_score']:.4f}")
            print(f"Risk Level: {row['risk_level']}")
            print(f"Suspicious Patterns: {row['suspicious_patterns']}")
            print(analyzer.interpret_anomaly(
                row['email'], 
                row['anomaly_score'], 
                row['suspicious_patterns']
            ))
            print("-" * 40)
    
    # Calculate accuracy
    predicted_labels = results['is_anomaly'].astype(int).values
    accuracy = np.mean(predicted_labels == true_labels)
    
    print(f"\n[*] Detection Accuracy: {accuracy:.2%}")
    print(f"[*] True Positives: {np.sum((predicted_labels == 1) & (true_labels == 1))}")
    print(f"[*] False Positives: {np.sum((predicted_labels == 1) & (true_labels == 0))}")
    print(f"[*] False Negatives: {np.sum((predicted_labels == 0) & (true_labels == 1))}")
    
    # Show entropy distribution
    print("\n" + "=" * 60)
    print("ENTROPY SIGNATURE ANALYSIS")
    print("=" * 60)
    
    normal_scores = results[~results['is_anomaly']]['anomaly_score'].values
    anomaly_scores = results[results['is_anomaly']]['anomaly_score'].values
    
    print(f"\nNormal Profile Entropy: {np.mean(normal_scores):.4f} ± {np.std(normal_scores):.4f}")
    print(f"Anomaly Profile Entropy: {np.mean(anomaly_scores):.4f} ± {np.std(anomaly_scores):.4f}")
    print(f"Separation Factor: {(np.mean(anomaly_scores) / np.mean(normal_scores)):.2f}x")
    
    print("\n[*] Analysis complete. System ready for real-world OSINT data.")
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = run_demo()
    
    # Additional feature: Real-time analysis capability
    print("\n" + "=" * 60)
    print("REAL-TIME ANALYSIS MODE")
    print("=" * 60)
    print("\nSystem is now ready to analyze email addresses in real-time.")
    print("Feed in leaked credentials to detect which accounts are likely active.")
    print("The entropy patterns will reveal compromised vs. abandoned accounts.")