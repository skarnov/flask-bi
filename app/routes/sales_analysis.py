from flask import Blueprint, jsonify, request
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask_cors import cross_origin

sales_forecast_bp = Blueprint('sales_forecast', __name__)
load_dotenv()

# Configuration
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
LARAVEL_API_URL = os.getenv('LARAVEL_API_URL')
os.makedirs(MODEL_DIR, exist_ok=True)

@sales_forecast_bp.route('/sales-forecast', methods=['GET'])
@cross_origin()
def sales_forecast():
    try:
        # 1. Get date range
        current_year = datetime.now().year
        start_date = datetime(current_year, 1, 1).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 2. Fetch sales data from Laravel API
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'status': 'completed'
        }
        response = requests.get(
            f"{LARAVEL_API_URL}/sales",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        sales_data = response.json().get('data', [])
        
        if not sales_data:
            raise ValueError("No sales data available from API")
        
        # 3. Process data
        df = pd.DataFrame(sales_data)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Aggregate monthly sales
        monthly_sales = df.resample('M', on='created_at').agg({
            'grand_total': 'sum',
            'id': 'count'
        }).rename(columns={
            'grand_total': 'total_sales',
            'id': 'transaction_count'
        })
        
        monthly_sales['month'] = monthly_sales.index.strftime('%b')
        monthly_sales['period'] = monthly_sales.index.strftime('%Y-%m')
        monthly_sales = monthly_sales.reset_index(drop=True)
        
        # 4. Forecasting
        forecast_result = calculate_forecast(monthly_sales)
        
        # 5. Prepare response
        return jsonify({
            'status': 'success',
            'data': {
                'actual_sales': monthly_sales.to_dict('records'),
                'forecast': forecast_result
            },
            'model_version': '1.0'
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 'error',
            'message': f"Failed to fetch sales data: {str(e)}"
        }), 502
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

def calculate_forecast(sales_data: pd.DataFrame) -> dict:
    """Calculate sales forecast with train-test validation"""
    if len(sales_data) < 3:
        return {
            'next_month_forecast': 0,
            'confidence_level': 0,
            'expected_range_low': 0,
            'expected_range_high': 0,
            'growth_rate': 0,
            'model_accuracy': 0
        }
    
    # Prepare features
    X = np.arange(len(sales_data)).reshape(-1, 1)
    y = sales_data['total_sales'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = max(0, 100 - (mae / np.mean(y_test)) * 100)
    
    # Predict next month
    next_month = np.array([[len(sales_data)]])
    next_month_scaled = scaler.transform(next_month)
    forecast = model.predict(next_month_scaled)[0]
    
    # Calculate confidence and range
    std_dev = np.std(y_test - y_pred)
    confidence = min(90 + (len(sales_data) * 2), 95)
    range_low = max(0, forecast - 1.5 * std_dev)
    range_high = forecast + 1.5 * std_dev
    
    # Growth rate calculation
    if len(sales_data) >= 2:
        current = sales_data.iloc[-1]['total_sales']
        previous = sales_data.iloc[-2]['total_sales']
        growth_rate = ((current - previous) / previous) * 100 if previous > 0 else 0
    else:
        growth_rate = 0
    
    # Save model if it doesn't exist
    model_path = f"{MODEL_DIR}/sales_forecast.joblib"
    if not os.path.exists(model_path):
        joblib.dump({
            'model': model,
            'scaler': scaler
        }, model_path)
    
    return {
        'next_month_forecast': round(float(forecast), 2),
        'confidence_level': round(float(confidence), 2),
        'expected_range_low': round(float(range_low), 2),
        'expected_range_high': round(float(range_high), 2),
        'growth_rate': round(float(growth_rate), 2),
        'model_accuracy': round(float(accuracy), 2)
    }