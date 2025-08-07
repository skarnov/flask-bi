from flask import Blueprint, jsonify, request
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib
import os
from flask_cors import cross_origin

product_analysis_bp = Blueprint('product_analysis', __name__)
load_dotenv()

# Model paths
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

@product_analysis_bp.route('/product-analysis', methods=['POST'])
@cross_origin()
def analyze_products():
    try:
        # 1. Load and validate data
        products = request.get_json()
        if not products or not isinstance(products, list):
            raise ValueError("Expected JSON array of products")
        
        df = pd.DataFrame(products)
        required_cols = ['product_id', 'name', 'price', 'units_sold', 'stock']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required fields: {required_cols}")

        # 2. Feature engineering
        df = df.dropna()
        df['revenue'] = df['price'] * df['units_sold']
        df['stock_ratio'] = df['units_sold'] / (df['stock'] + df['units_sold'])
        df['sales_velocity'] = df['units_sold'] / df['price']
        
        # 3. Performance scoring
        scaler = MinMaxScaler()
        features = ['revenue', 'stock_ratio', 'sales_velocity']
        df[features] = scaler.fit_transform(df[features])
        
        df['performance_score'] = (
            0.5 * df['revenue'] + 
            0.3 * df['stock_ratio'] + 
            0.2 * df['sales_velocity']
        ) * 100

        # 4. Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['performance_tier'] = kmeans.fit_predict(df[features])
        df['performance_tier'] = df['performance_tier'].map({
            0: 'Low', 
            1: 'Medium', 
            2: 'High'
        })

        # 5. Save models (first run only)
        if not os.path.exists(f'{MODEL_DIR}/scaler.joblib'):
            joblib.dump(scaler, f'{MODEL_DIR}/scaler.joblib')
            joblib.dump(kmeans, f'{MODEL_DIR}/kmeans.joblib')

        # 6. Prepare response
        results = df.to_dict('records')
        summary = {
            'avg_score': round(df['performance_score'].mean(), 1),
            'top_product': df.loc[df['performance_score'].idxmax(), 'name'],
            'total_revenue': round(df['revenue'].sum(), 2)
        }

        return jsonify({
            'status': 'success',
            'data': results,
            'summary': summary,
            'model_version': '1.0'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400