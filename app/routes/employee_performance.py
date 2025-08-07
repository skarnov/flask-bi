from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from Laravel

# Configuration
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

class PerformanceAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(n_clusters=3, random_state=42)
        self.anomaly_model = IsolationForest(contamination=0.05, random_state=42)
        self.initialized = False

    def initialize_models(self, df):
        """Initialize models with sample data"""
        features = self._get_features(df)
        self.scaler.fit(features)
        self.cluster_model.fit(features)
        self.anomaly_model.fit(features)
        self.initialized = True
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
        joblib.dump(self.cluster_model, os.path.join(MODEL_DIR, 'kmeans.joblib'))
        joblib.dump(self.anomaly_model, os.path.join(MODEL_DIR, 'isolation_forest.joblib'))

    def load_models(self):
        """Load pre-trained models"""
        try:
            self.scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
            self.cluster_model = joblib.load(os.path.join(MODEL_DIR, 'kmeans.joblib'))
            self.anomaly_model = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.joblib'))
            self.initialized = True
            return True
        except:
            return False

    def _get_features(self, df):
        """Extract and prepare features for modeling"""
        features = df[['completed_tasks', 'total_tasks', 'avg_task_hours', 'completion_rate']]
        return self.scaler.transform(features) if self.initialized else features

    def analyze(self, data):
        """Main analysis workflow"""
        df = pd.DataFrame(data)
        
        # Data cleaning
        df['avg_task_hours'] = df['avg_task_hours'].fillna(0)
        df['completed_tasks'] = df['completed_tasks'].fillna(0)
        df['total_tasks'] = df['total_tasks'].fillna(0)
        
        # Calculate metrics
        df['completion_rate'] = df.apply(
            lambda x: x['completed_tasks'] / x['total_tasks'] if x['total_tasks'] > 0 else 0,
            axis=1
        )
        
        # Initialize models if not loaded
        if not self.initialized and not self.load_models():
            self.initialize_models(df)
        
        # Get features
        features = self._get_features(df)
        
        # Cluster analysis
        df['performance_cluster'] = self.cluster_model.predict(features)
        
        # Anomaly detection
        df['is_anomaly'] = self.anomaly_model.predict(features)
        df['is_anomaly'] = df['is_anomaly'].apply(lambda x: x == -1)
        
        # Performance scoring (0-100)
        df['performance_score'] = (
            0.4 * df['completion_rate'] +
            0.3 * (1 - df['avg_task_hours']) +
            0.3 * df['completed_tasks']
        )
        df['performance_score'] = MinMaxScaler().fit_transform(
            df[['performance_score']]
        ) * 100
        
        return df.to_dict('records')

analyzer = PerformanceAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_performance():
    try:
        # Get JSON data from Laravel
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({
                'status': 'error',
                'message': 'Invalid data format. Expected array of employee data.'
            }), 400
        
        # Perform analysis
        results = analyzer.analyze(data)
        
        return jsonify({
            'status': 'success',
            'data': results,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)