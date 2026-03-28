# amfs_tm/src/lib/filters/filter_apply.py
import yaml
import logging
from .rules import FilterRule

def filter_apply(main_config_path, filters_config_path, args):
    """
    Standardized entry point for the filtering pipeline.
    Corrected to handle nested 'PATHS' -> 'base_path' structure.
    """
    logging.info(f"--- Starting Filter Application for snapshot {args.snapshot} ---")

    # 1. Load Main Config
    with open(main_config_path, 'r') as f:
        main_cfg = yaml.safe_load(f)
    
    # FIX: Navigate the nested 'PATHS' key to get 'base_path'
    root_path = main_cfg.get('PATHS', {}).get('base_path')
    
    if not root_path:
        logging.error("CRITICAL ERROR: 'base_path' not found under 'PATHS' in main_config.yaml")
        raise KeyError("base_path missing in main_config.yaml")

    # 2. Load Filters Config
    with open(filters_config_path, 'r') as f:
        filters_cfg = yaml.safe_load(f)

    # 3. Initialize and Execute the Rules Engine
    try:
        engine = FilterRule(root_path, filters_cfg, args)
        output_file = engine.apply()
        logging.info(f"Filter Application successful. Output: {output_file}")
    except Exception as e:
        logging.error(f"Filter Application failed: {e}", exc_info=True)
        raise e