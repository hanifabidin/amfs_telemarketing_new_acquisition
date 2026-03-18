# amfs_tm/src/lib/clean/clean_config_utils.py
import yaml
import logging
import os

def load_config(yaml_file_path: str) -> dict | None:
    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading {yaml_file_path}: {e}")
        return None

def get_path(base: str, *path_parts: str) -> str:
    """Constructs a file path safely by filtering out None/empty values."""
    if base is None:
        raise ValueError("CRITICAL: 'base_path' is missing from your config.")
    
    # Filter: Only keep parts that are strings and not empty
    # This specifically prevents the "NoneType" error in os.path.join
    valid_parts = [str(p).strip('/') for p in path_parts if p]
    
    return os.path.join(base.rstrip('/'), *valid_parts)