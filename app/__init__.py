from flask import Flask
from dotenv import load_dotenv

def create_app():
    load_dotenv()
    app = Flask(__name__)

    from .routes.employee_performance import employee_bp
    from .routes.product_analysis import product_analysis_bp
    from .routes.product_recommendation_system import recommendation_bp
    from .routes.sales_analysis import sales_bp

    app.register_blueprint(employee_bp, url_prefix='/api')
    app.register_blueprint(product_analysis_bp, url_prefix='/api')
    app.register_blueprint(recommendation_bp, url_prefix='/api')
    app.register_blueprint(sales_bp, url_prefix='/api')

    return app
