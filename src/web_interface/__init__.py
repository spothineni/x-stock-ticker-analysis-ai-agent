"""Web interface modules for the stock sentiment analyzer."""

from .app import create_app
from .api import api_bp
from .dashboard import dashboard_bp

__all__ = ['create_app', 'api_bp', 'dashboard_bp']