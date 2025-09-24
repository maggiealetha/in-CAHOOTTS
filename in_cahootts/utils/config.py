"""
Configuration handling utilities.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Handle legacy config path format
    if not config_path.startswith('/') and not config_path.startswith('./'):
        # Legacy behavior: add mag_nODE_model/ prefix if not present
        if not config_path.startswith('mag_nODE_model/'):
            config_path = os.path.join("mag_nODE_model/", config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
        # Convert boolean strings to actual booleans (legacy compatibility)
        if 'use_prior' in config:
            config['use_prior'] = bool(config['use_prior'])
        if 'shuffle_prior' in config:
            config['shuffle_prior'] = bool(config['shuffle_prior'])
        if 'decay' in config:
            config['decay'] = bool(config['decay'])
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'neurons': 158,
            'dropout': 0.2,
            'use_prior': False,
            'prior_weight': 0.1
        },
        'training': {
            'epochs': 500,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'patience': 30,
            'batch_size': 20
        },
        'data': {
            'sequence_length': 60,
            'time_min': 0,
            'time_max': 60,
            'time_step': 1,
            'n_dataloaders': 1
        },
        'experiment': {
            'cross_validation_folds': 5,
            'random_seed': 257,
            'output_dir': 'results'
        }
    }