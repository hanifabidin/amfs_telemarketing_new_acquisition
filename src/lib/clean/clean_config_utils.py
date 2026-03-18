# src/lib/clean/clean_config_utils.py
# src/lib/clean/clean_config_utils.py
import yaml
import logging
import os

def load_config(yaml_file_path: str) -> dict | None:
    """Loads configuration from a YAML file."""
    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded config from {yaml_file_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading config {yaml_file_path}: {e}")
        return None

def get_path(base: str, *path_parts: str) -> str:
    """Constructs a file path safely, ensuring no double slashes."""
    # Works for both /Volumes/... and dbfs:/...
    clean_parts = [p.strip('/') for p in path_parts if p]
    return os.path.join(base, *clean_parts)