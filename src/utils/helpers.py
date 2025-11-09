"""
Utility functions for phishing research project
"""

import os
import yaml
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = 'logs', log_file: str = 'research.log'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/{log_file}'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_directories():
    """Create project directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'data/synthetic',
        'models/saved',
        'results',
        'logs',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Project directories created successfully")


def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(data: dict, filename: str, results_dir: str = 'results'):
    """Save results to JSON file"""
    import json
    
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    create_directories()
    logger = setup_logging()
    logger.info("Utility functions initialized")