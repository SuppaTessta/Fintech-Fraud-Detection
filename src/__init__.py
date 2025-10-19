"""
Fintech Fraud Detection - Machine Learning package

This package contains modules for:
- Data loading and validation
- Data preprocessing and feature engineering
- Model training and evaluation
- Configuration management
- Logging utilities
"""

__version__ = '1.0.0'

from .data_loader import load_dataset
from .model import train_model, evaluate_model, load_model
from .preprocess import preprocess_data
from .config import get_config, update_config

__all__ = [
    'load_dataset',
    'train_model',
    'evaluate_model',
    'load_model',
    'preprocess_data',
    'get_config',
    'update_config',
]
