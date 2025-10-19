"""Default configuration settings for the fraud detection system."""
from typing import Dict, Any, Optional
import os
import yaml
import logging

# Base configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    'model': {
        'contamination': 0.0017,
        'random_state': 42,
        'n_estimators': 100,
        'max_samples': 'auto',
        'n_jobs': -1
    },
    'preprocessing': {
        'undersample_ratio': 0.5,
        'random_state': 42
    },
    'paths': {
        'data_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
        'models_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
        'logs_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'),
        'plots_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file, falling back to defaults.

    Args:
        config_path: Optional path to a YAML configuration file

    Returns:
        Dict containing the configuration
    """
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                if custom_config:
                    _deep_update(config, custom_config)
        except Exception as e:
            logging.warning(f"Error loading config from {config_path}: {str(e)}")
            logging.warning("Using default configuration")

    # Ensure directories exist
    for dir_path in config['paths'].values():
        os.makedirs(dir_path, exist_ok=True)

    return config

def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """
    Recursively update a dictionary.

    Args:
        base_dict: Dictionary to update
        update_dict: Dictionary with updates
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

# Global configuration instance
config = load_config()

def get_config() -> Dict[str, Any]:
    """Get the current configuration."""
    return config

def update_config(new_config: Dict[str, Any]) -> None:
    """
    Update the current configuration.

    Args:
        new_config: Dictionary with configuration updates
    """
    global config
    _deep_update(config, new_config)
