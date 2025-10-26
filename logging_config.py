"""
Production logging configuration for LLM Lab
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure comprehensive logging for production"""
    logger = logging.getLogger("llm_lab")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    logger.handlers = []
    
    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler for production logs
    file_handler = RotatingFileHandler(
        'llm_lab.log',
        maxBytes=10485760,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
