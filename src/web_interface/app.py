"""Flask application factory and main app configuration."""

from flask import Flask
from flask_cors import CORS
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, origins=["*"])
    
    # App configuration
    app.config['SECRET_KEY'] = 'your-secret-key-here'  # In production, use environment variable
    app.config['DEBUG'] = config.web_config['debug']
    
    # Register blueprints
    from .api import api_bp
    from .dashboard import dashboard_bp
    
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(dashboard_bp, url_prefix='/')
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'stock-sentiment-analyzer'}
    
    logger.info("Flask application created and configured")
    return app