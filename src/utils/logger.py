"""Logging configuration and utilities."""

import sys
from pathlib import Path
from loguru import logger
from .config import config

def setup_logger(name: str = "stock_sentiment") -> logger:
    """Set up logger with configuration from config file."""
    
    # Remove default handler
    logger.remove()
    
    # Get logging configuration
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', 
                           "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # Add file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / f"{name}.log",
        format=log_format,
        level=log_level,
        rotation=config.get('logging.rotation', '1 day'),
        retention=config.get('logging.retention', '30 days'),
        compression="zip"
    )
    
    return logger