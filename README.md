# Flask Business Intelligence Suite

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/flask-2.0%2B-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive business intelligence solution built with Flask that provides:
- Employee performance analytics
- Product performance analysis
- Recommendation engine
- Sales forecasting

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Models](#models)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

### 1. Employee Performance Analytics
- Performance clustering (KMeans)
- Anomaly detection (Isolation Forest)
- Composite scoring (0-100 scale)
- Multi-metric analysis:
  - Task completion rates
  - Average task duration
  - Total throughput

### 2. Product Performance Analysis
- Revenue/sales analytics
- Inventory efficiency metrics
- Performance tiering (Low/Medium/High)
- Historical trend analysis

### 3. Recommendation Engine
- **6 Recommendation Strategies**:
  - Popular products
  - Trending items
  - Content-based (KNN)
  - Collaborative filtering
  - Similar user preferences
  - Personalized recommendations
- Hybrid approach combining:
  - Explicit ratings
  - Implicit purchase signals

### 4. Sales Forecasting
- Linear regression predictions
- Confidence intervals
- Growth rate calculations
- Model accuracy reporting
- Monthly aggregation

## Architecture

```mermaid
graph TD
    A[Frontend] -->|API Calls| B[Flask BI Suite]
    B --> C[Employee Analysis]
    B --> D[Product Analysis]
    B --> E[Recommendation Engine]
    B --> F[Sales Forecast]
    C --> G[(Database)]
    D --> G
    E --> G
    F --> H[External APIs]
```

### Tech Stack
- **Core**: Python 3.8+, Flask 2.0+
- **Machine Learning**:
  - scikit-learn
  - Surprise (recommendations)
- **Data Processing**: Pandas, NumPy
- **Persistence**: Joblib (model storage)
- **APIs**: REST, CORS-enabled

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL (recommended)
- Redis (for caching)

### Setup
1. Clone repository:
   ```bash
   git clone https://github.com/yourrepo/flask-bi-suite.git
   cd flask-bi-suite
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Configuration

### Environment Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_DIR` | Path to store trained models | `./models` |
| `DATABASE_URI` | Database connection string | `postgresql://user:pass@localhost/db` |
| `LARAVEL_API_URL` | External API base URL | `https://api.yourdomain.com` |
| `FLASK_ENV` | Runtime environment | `production` |

### Model Training
To train all models:
```bash
python train_models.py
```

## API Documentation

### Base URL
`https://erp.obydullah.com/api/v1`

### Endpoints

#### 1. Employee Performance
- **POST** `/analyze`
  - Input: Array of employee performance records
  - Output: Analysis with scores and clusters

#### 2. Product Analysis
- **POST** `/product-analysis`
  - Input: Product data array
  - Output: Performance metrics and tiering

#### 3. Recommendations
- **POST** `/recommendations/products`
  - Parameters:
    - `strategy`: Recommendation approach
    - `user_id`: For personalized strategies
    - `product_id`: For content-based
  - Output: Recommended products with scores

#### 4. Sales Forecast
- **GET** `/sales-forecast`
  - Parameters: `start_date`, `end_date`
  - Output: Forecast with confidence metrics

## Models

### 1. Employee Analysis
- **Clustering**: 3-tier KMeans
- **Anomaly Detection**: Isolation Forest
- **Scoring Formula**:
  ```
  0.4 * completion_rate + 
  0.3 * (1 - avg_task_hours) + 
  0.3 * completed_tasks
  ```

### 2. Product Analysis
- **Tiering**: KMeans clustering
- **Scoring**:
  ```
  0.5 * revenue + 
  0.3 * stock_ratio + 
  0.2 * sales_velocity
  ```

### 3. Recommendation System
- **Content-Based**: KNN on product features
- **Collaborative**: Item-based KNNBasic
- **Hybrid Scoring**: Combines explicit and implicit signals

### 4. Sales Forecasting
- **Algorithm**: Linear Regression
- **Features**: Time-indexed sales data
- **Validation**: 80/20 train-test split

## Deployment

### Production Setup
1. Configure WSGI server (Gunicorn recommended):
   ```bash
   gunicorn -w 4 -b :5000 wsgi:app
   ```

2. Set up NGINX proxy:
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;

       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
       }
   }
   ```

3. Implement caching for recommendation models

### Docker
```bash
docker build -t flask-bi .
docker run -p 5000:5000 flask-bi
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Note**: This is a template README. Replace placeholder values with your actual project details, and expand sections as needed for your specific implementation. Consider adding:
- Screenshots of sample outputs
- Detailed API examples with curl commands
- Performance benchmarks
- Roadmap for future features