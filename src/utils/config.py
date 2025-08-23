"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    """Configuration manager for the stock sentiment analyzer."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration."""
        # Load environment variables
        load_dotenv()
        
        # Load YAML configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_env(self, key: str, default: str = None) -> str:
        """Get environment variable."""
        return os.getenv(key, default)
    
    @property
    def twitter_config(self) -> Dict[str, Any]:
        """Get Twitter API configuration."""
        return {
            'bearer_token': self.get_env('TWITTER_BEARER_TOKEN'),
            'api_key': self.get_env('TWITTER_API_KEY'),
            'api_secret': self.get_env('TWITTER_API_SECRET'),
            'access_token': self.get_env('TWITTER_ACCESS_TOKEN'),
            'access_token_secret': self.get_env('TWITTER_ACCESS_TOKEN_SECRET'),
        }
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            'url': self.get_env('DATABASE_URL', 'sqlite:///data/stock_sentiment.db'),
            'type': self.get('database.type', 'sqlite'),
            'path': self.get('database.path', 'data/stock_sentiment.db'),
        }
    
    @property
    def stock_tickers(self) -> list:
        """Get list of stock tickers to analyze."""
        return self.get('data_collection.stock_tickers', [])
    
    @property
    def web_config(self) -> Dict[str, Any]:
        """Get web interface configuration."""
        return {
            'host': self.get_env('FLASK_HOST', self.get('web_interface.host', '0.0.0.0')),
            'port': int(self.get_env('FLASK_PORT', self.get('web_interface.port', 12000))),
            'debug': self.get_env('FLASK_DEBUG', 'True').lower() == 'true',
        }

# Global configuration instance
config = Config()