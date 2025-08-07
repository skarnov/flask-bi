from flask import Blueprint, jsonify, request
from sqlalchemy import func, and_, or_
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from ..models import db, Product, Sale, SaleDetail, Stock, Review, User

recommendation_bp = Blueprint('product_recommendation_system', __name__, url_prefix='/recommendations')

# Model persistence
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Model paths
KNN_MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.joblib')
COLLAB_FILTERING_MODEL_PATH = os.path.join(MODEL_DIR, 'collab_filtering_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

def load_or_train_models():
    """Load or train recommendation models"""
    try:
        # Try to load pre-trained models
        knn_model = joblib.load(KNN_MODEL_PATH)
        collab_model = joblib.load(COLLAB_FILTERING_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except:
        # Train models if they don't exist
        knn_model, collab_model, scaler = train_recommendation_models()
        
    return knn_model, collab_model, scaler

def train_recommendation_models():
    """Train and save recommendation models"""
    print("Training recommendation models...")
    
    # 1. Prepare data for KNN model (content-based)
    products_data = prepare_product_features()
    
    # Split into train/test
    train_data, test_data = train_test_split(products_data, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.drop('product_id', axis=1))
    test_scaled = scaler.transform(test_data.drop('product_id', axis=1))
    
    # Train KNN model
    knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine')
    knn_model.fit(train_scaled)
    
    # Evaluate on test data
    test_distances, test_indices = knn_model.kneighbors(test_scaled)
    print(f"KNN Test distances mean: {np.mean(test_distances)}")
    
    # 2. Prepare data for collaborative filtering
    ratings_data = prepare_ratings_data()
    
    # Define reader and data for Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_data[['user_id', 'product_id', 'rating']], reader)
    
    # Train collaborative filtering model
    collab_model = KNNBasic(sim_options={
        'name': 'cosine',
        'user_based': False  # item-based
    })
    
    # Cross-validate
    cv_results = cross_validate(collab_model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    print(f"Collaborative Filtering CV results: {cv_results}")
    
    # Full training
    trainset = data.build_full_trainset()
    collab_model.fit(trainset)
    
    # Save models
    joblib.dump(knn_model, KNN_MODEL_PATH)
    joblib.dump(collab_model, COLLAB_FILTERING_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return knn_model, collab_model, scaler

def prepare_product_features():
    """Prepare product features dataframe for KNN model"""
    query = db.session.query(
        Product.id.label('product_id'),
        Product.name,
        Product.price,
        func.coalesce(func.avg(Review.rating), 0).label('avg_rating'),
        func.coalesce(func.count(Review.id), 0).label('review_count'),
        func.coalesce(func.sum(SaleDetail.quantity), 0).label('total_sales')
    ).outerjoin(
        Review, Product.id == Review.product_id
    ).outerjoin(
        Stock, Product.id == Stock.product_id
    ).outerjoin(
        SaleDetail, Stock.id == SaleDetail.stock_id
    ).outerjoin(
        Sale, SaleDetail.sale_id == Sale.id
    ).group_by(
        Product.id, Product.name, Product.price
    )
    
    df = pd.read_sql(query.statement, db.session.bind)
    
    # Feature engineering
    df['price'] = df['price'].astype(float)
    df['price_log'] = np.log1p(df['price'])
    df['popularity'] = df['review_count'] * df['avg_rating']
    
    return df

def prepare_ratings_data():
    """Prepare ratings data for collaborative filtering"""
    # Explicit ratings from reviews
    reviews_query = db.session.query(
        Review.user_id,
        Review.product_id,
        Review.rating
    ).filter(
        Review.rating.isnot(None)
    )
    
    reviews_df = pd.read_sql(reviews_query.statement, db.session.bind)
    
    # Implicit ratings from purchases (weighted lower than explicit ratings)
    purchases_query = db.session.query(
        Sale.user_id,
        Stock.product_id,
        func.lit(4.0).label('rating')  # Assume purchase implies positive rating
    ).join(
        SaleDetail, Sale.id == SaleDetail.sale_id
    ).join(
        Stock, SaleDetail.stock_id == Stock.id
    ).filter(
        Sale.status == 'completed'
    )
    
    purchases_df = pd.read_sql(purchases_query.statement, db.session.bind)
    
    # Combine both sources
    ratings_df = pd.concat([reviews_df, purchases_df], ignore_index=True)
    
    # Average if user rated same product multiple ways
    ratings_df = ratings_df.groupby(['user_id', 'product_id'])['rating'].mean().reset_index()
    
    return ratings_df

@recommendation_bp.route('/train', methods=['POST'])
def train_models():
    """Endpoint to trigger model retraining"""
    try:
        train_recommendation_models()
        return jsonify({"success": True, "message": "Models trained successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@recommendation_bp.route('/products', methods=['POST'])
def product_recommendations():
    # Input validation
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400
    
    strategy = data.get('strategy', 'popular')
    min_rating = float(data.get('min_rating', 0))
    min_purchases = int(data.get('min_purchases', 0))
    email = data.get('email')
    limit = int(data.get('limit', 50))

    # Load models
    knn_model, collab_model, scaler = load_or_train_models()

    # Strategy routing
    try:
        if strategy == 'popular':
            result = get_popular_products(min_rating, min_purchases, limit)
        elif strategy == 'trending':
            result = get_trending_products(min_rating, min_purchases, limit)
        elif strategy == 'content_based':
            product_id = data.get('product_id')
            if not product_id:
                return jsonify({"success": False, "message": "product_id required for content_based strategy"}), 400
            result = get_content_based_recommendations(product_id, knn_model, scaler, limit)
        elif strategy in ['similar_users', 'personalized', 'collaborative']:
            if not email:
                return jsonify({
                    "success": False,
                    "message": f"Email is required for {strategy} strategy"
                }), 400
            user = User.query.filter_by(email=email).first()
            if not user:
                return jsonify({"success": False, "message": "User not found"}), 404
            
            if strategy == 'similar_users':
                result = get_similar_users_recommendations(user.id, min_rating, min_purchases, limit)
            elif strategy == 'collaborative':
                result = get_collaborative_recommendations(user.id, collab_model, limit)
            else:
                result = get_personalized_recommendations(user.id, min_rating, min_purchases, limit)
        else:
            return jsonify({"success": False, "message": "Invalid strategy"}), 400

        return jsonify({
            "success": True,
            "strategy": strategy,
            "count": len(result),
            "recommendations": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

def get_content_based_recommendations(product_id, knn_model, scaler, limit=50):
    """Get recommendations based on product features using KNN"""
    products_df = prepare_product_features()
    
    if product_id not in products_df['product_id'].values:
        return []
    
    # Get features for the target product
    target_idx = products_df[products_df['product_id'] == product_id].index[0]
    target_features = products_df.drop('product_id', axis=1).iloc[target_idx]
    
    # Scale features
    target_scaled = scaler.transform([target_features])
    
    # Find similar products
    distances, indices = knn_model.kneighbors(target_scaled, n_neighbors=limit+1)
    
    # Exclude the product itself and get recommendations
    recommendations = []
    for i, idx in enumerate(indices[0]):
        if products_df.iloc[idx]['product_id'] != product_id:
            product = products_df.iloc[idx]
            recommendations.append({
                'product_id': product['product_id'],
                'product_name': product['name'],
                'similarity_score': float(1 - distances[0][i]),  # Convert to similarity
                'price': float(product['price']),
                'average_rating': float(product['avg_rating'])
            })
            if len(recommendations) >= limit:
                break
                
    return recommendations

def get_collaborative_recommendations(user_id, collab_model, limit=50):
    """Get recommendations using collaborative filtering"""
    # Get all product IDs
    product_ids = [p.id for p in Product.query.all()]
    
    # Get products the user has already purchased/rated
    user_products = db.session.query(
        Stock.product_id
    ).join(
        SaleDetail, Stock.id == SaleDetail.stock_id
    ).join(
        Sale, and_(
            SaleDetail.sale_id == Sale.id,
            Sale.status == 'completed',
            Sale.user_id == user_id
        )
    ).distinct().all()
    
    rated_products = db.session.query(
        Review.product_id
    ).filter(
        Review.user_id == user_id
    ).distinct().all()
    
    known_products = set([p.product_id for p in user_products] + [p.product_id for p in rated_products])
    
    # Predict ratings for unknown products
    predictions = []
    for product_id in product_ids:
        if product_id not in known_products:
            pred = collab_model.predict(user_id, product_id)
            predictions.append((product_id, pred.est))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get product details for top recommendations
    recommendations = []
    for product_id, rating in predictions[:limit]:
        product = Product.query.get(product_id)
        if product:
            recommendations.append({
                'product_id': product.id,
                'product_name': product.name,
                'predicted_rating': float(rating),
                'price': float(product.price)
            })
    
    return recommendations

def get_popular_products(min_rating, min_purchases, limit=50):
    query = db.session.query(
        Product.id.label('product_id'),
        Product.name.label('product_name'),
        func.coalesce(func.sum(SaleDetail.quantity), 0).label('total_purchases'),
        func.coalesce(func.avg(Review.rating), 0).label('average_rating')
    ).outerjoin(
        Stock, Product.id == Stock.product_id
    ).outerjoin(
        SaleDetail, Stock.id == SaleDetail.stock_id
    ).outerjoin(
        Sale, and_(
            SaleDetail.sale_id == Sale.id,
            Sale.status == 'completed'
        )
    ).outerjoin(
        Review, Product.id == Review.product_id
    ).group_by(
        Product.id, Product.name
    ).having(
        func.coalesce(func.avg(Review.rating), 0) >= min_rating
    ).having(
        func.coalesce(func.sum(SaleDetail.quantity), 0) >= min_purchases
    ).order_by(
        func.coalesce(func.sum(SaleDetail.quantity), 0).desc()
    ).limit(limit)

    return [{
        'product_id': item.product_id,
        'product_name': item.product_name,
        'total_purchases': int(item.total_purchases),
        'average_rating': float(item.average_rating)
    } for item in query.all()]

def get_trending_products(min_rating, min_purchases, limit=50):
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    query = db.session.query(
        Product.id.label('product_id'),
        Product.name.label('product_name'),
        func.coalesce(func.sum(SaleDetail.quantity), 0).label('total_purchases'),
        func.coalesce(func.avg(Review.rating), 0).label('average_rating'),
        (func.coalesce(func.count(Sale.id), 0) * func.coalesce(func.avg(Review.rating), 0)).label('trend_score')
    ).outerjoin(
        Stock, Product.id == Stock.product_id
    ).outerjoin(
        SaleDetail, Stock.id == SaleDetail.stock_id
    ).outerjoin(
        Sale, and_(
            SaleDetail.sale_id == Sale.id,
            Sale.status == 'completed',
            Sale.created_at >= thirty_days_ago
        )
    ).outerjoin(
        Review, Product.id == Review.product_id
    ).group_by(
        Product.id, Product.name
    ).having(
        func.coalesce(func.avg(Review.rating), 0) >= min_rating
    ).having(
        func.coalesce(func.sum(SaleDetail.quantity), 0) >= min_purchases
    ).order_by(
        func.coalesce(func.count(Sale.id), 0) * func.coalesce(func.avg(Review.rating), 0).desc()
    ).limit(limit)

    return [{
        'product_id': item.product_id,
        'product_name': item.product_name,
        'total_purchases': int(item.total_purchases),
        'average_rating': float(item.average_rating),
        'trend_score': float(item.trend_score)
    } for item in query.all()]

def get_similar_users_recommendations(user_id, min_rating, min_purchases, limit=50):
    # Get user's purchased products
    user_products = db.session.query(
        Stock.product_id
    ).join(
        SaleDetail, Stock.id == SaleDetail.stock_id
    ).join(
        Sale, and_(
            SaleDetail.sale_id == Sale.id,
            Sale.status == 'completed',
            Sale.user_id == user_id
        )
    ).distinct().all()
    
    user_product_ids = [p.product_id for p in user_products]
    
    if not user_product_ids:
        return []

    # Find similar users
    similar_users = db.session.query(
        Sale.user_id
    ).join(
        SaleDetail, Sale.id == SaleDetail.sale_id
    ).join(
        Stock, SaleDetail.stock_id == Stock.id
    ).filter(
        Stock.product_id.in_(user_product_ids),
        Sale.user_id != user_id,
        Sale.status == 'completed'
    ).distinct().all()
    
    similar_user_ids = [u.user_id for u in similar_users]
    
    if not similar_user_ids:
        return []

    # Get recommendations from similar users
    recommendations = db.session.query(
        Product.id.label('product_id'),
        Product.name.label('product_name'),
        func.count(Sale.id).label('purchased_by_similar_users'),
        func.coalesce(func.avg(Review.rating), 0).label('average_rating')
    ).join(
        Stock, Product.id == Stock.product_id
    ).join(
        SaleDetail, Stock.id == SaleDetail.stock_id
    ).join(
        Sale, and_(
            SaleDetail.sale_id == Sale.id,
            Sale.status == 'completed',
            Sale.user_id.in_(similar_user_ids)
        )
    ).outerjoin(
        Review, Product.id == Review.product_id
    ).filter(
        ~Product.id.in_(user_product_ids)
    ).group_by(
        Product.id, Product.name
    ).having(
        func.coalesce(func.avg(Review.rating), 0) >= min_rating
    ).having(
        func.count(Sale.id) >= min_purchases
    ).order_by(
        func.count(Sale.id).desc()
    ).limit(limit)

    return [{
        'product_id': item.product_id,
        'product_name': item.product_name,
        'purchased_by_similar_users': int(item.purchased_by_similar_users),
        'average_rating': float(item.average_rating)
    } for item in recommendations.all()]

def get_personalized_recommendations(user_id, min_rating, min_purchases, limit=50):
    # Get user's average rating
    user_avg_rating = db.session.query(
        func.avg(Review.rating)
    ).filter(
        Review.user_id == user_id
    ).scalar() or 3.0

    # Get user's purchased products
    user_products = db.session.query(
        Stock.product_id
    ).join(
        SaleDetail, Stock.id == SaleDetail.stock_id
    ).join(
        Sale, and_(
            SaleDetail.sale_id == Sale.id,
            Sale.status == 'completed',
            Sale.user_id == user_id
        )
    ).distinct().all()
    
    user_product_ids = [p.product_id for p in user_products]

    # Personalized scoring
    recommendations = db.session.query(
        Product.id.label('product_id'),
        Product.name.label('product_name'),
        func.coalesce(func.sum(SaleDetail.quantity), 0).label('total_purchases'),
        func.coalesce(func.avg(Review.rating), 0).label('average_rating'),
        (func.coalesce(func.sum(SaleDetail.quantity), 0) * 
         (5 - func.abs(func.coalesce(func.avg(Review.rating), 0) - user_avg_rating))).label('score')
    ).outerjoin(
        Stock, Product.id == Stock.product_id
    ).outerjoin(
        SaleDetail, Stock.id == SaleDetail.stock_id
    ).outerjoin(
        Sale, and_(
            SaleDetail.sale_id == Sale.id,
            Sale.status == 'completed'
        )
    ).outerjoin(
        Review, Product.id == Review.product_id
    ).filter(
        ~Product.id.in_(user_product_ids)
    ).group_by(
        Product.id, Product.name
    ).having(
        func.coalesce(func.avg(Review.rating), 0) >= min_rating
    ).having(
        func.coalesce(func.sum(SaleDetail.quantity), 0) >= min_purchases
    ).order_by(
        (func.coalesce(func.sum(SaleDetail.quantity), 0) * 
         (5 - func.abs(func.coalesce(func.avg(Review.rating), 0) - user_avg_rating))).desc()
    ).limit(limit)

    return [{
        'product_id': item.product_id,
        'product_name': item.product_name,
        'total_purchases': int(item.total_purchases),
        'average_rating': float(item.average_rating),
        'score': float(item.score)
    } for item in recommendations.all()]