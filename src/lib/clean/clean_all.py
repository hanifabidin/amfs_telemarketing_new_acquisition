# amfs_tm/src/lib/clean/clean_all.py
import os
import logging
from . import clean_config_utils
from . import clean_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_cleaning_step(step_config: dict, paths_cfg: dict, snapshot: str):
    """Processes a single file from 01_raw to 02_processed."""
    base_path = paths_cfg.get('base_path')
    raw_prefix = paths_cfg.get('raw_dir_prefix')      
    proc_prefix = paths_cfg.get('processed_dir_prefix')

    # Ensure snapshot-specific filenames
    raw_fname = step_config.get('raw_data_filename_pattern').format(snapshot=snapshot)
    proc_fname = step_config.get('processed_data_filename_pattern').format(snapshot=snapshot)
    data_source = step_config.get('data_source', 'BM')
    cleaning_type = step_config.get('cleaning_type', 'standard')

    # Construct direct paths: base_path -> Volume -> Source -> Snapshot -> File
    input_fpath = clean_config_utils.get_path(base_path, raw_prefix, f'data_from_{data_source}', snapshot, raw_fname)
    output_fpath = clean_config_utils.get_path(base_path, proc_prefix, f'data_from_{data_source}', snapshot, proc_fname)

    if not os.path.exists(input_fpath):
        logging.error(f"Raw file not found: {input_fpath}")
        return

    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
    
    separator = '|' if raw_fname.endswith('.txt') else ','
    try:
        success = False
        if cleaning_type == 'standard':
            success = clean_utils.apply_standard_cleaning(input_fpath, output_fpath, separator=separator)
        elif cleaning_type == 'balance':
            success = clean_utils.apply_balance_cleaning(input_fpath, output_fpath, separator=separator)
        elif cleaning_type == 'custdemo':
            success = clean_utils.apply_custdemo_cleaning(input_fpath, output_fpath, separator=separator)
        elif cleaning_type == 'trxdebit_edc':
            success = clean_utils.apply_trxdebit_edc_cleaning(input_fpath, output_fpath, snapshot, separator=separator)
        elif cleaning_type == 'trxnet':
            success = clean_utils.apply_trxnet_cleaning(input_fpath, output_fpath, separator=separator)
        
        if success:
            logging.info(f"Cleaned: {proc_fname}")
    except Exception as e:
        logging.error(f"Failed to process {raw_fname}: {e}")

def clean_all(main_config_path: str, clean_config_path: str, args):
    """Orchestrator for the data cleaning pipeline."""
    main_cfg = clean_config_utils.load_config(main_config_path)
    clean_cfg = clean_config_utils.load_config(clean_config_path)

    if not main_cfg or not clean_cfg: return
    
    paths_cfg = main_cfg.get('PATHS', {})
    base_path = paths_cfg.get('base_path')
    snapshot = args.snapshot

    if base_path is None:
        logging.error("ERROR: 'base_path' not found in main_config.yaml")
        return

    # Pre-create directory structures in the Volumes
    for data_source in ['BM', 'AMFS']:
        for sub_prefix in [paths_cfg.get('raw_dir_prefix'), paths_cfg.get('processed_dir_prefix')]:
            if sub_prefix:
                target_dir = os.path.join(base_path, sub_prefix, f'data_from_{data_source}', snapshot)
                os.makedirs(target_dir, exist_ok=True)

    # Execute each step in the cleaning config
    for step in clean_cfg.get('cleaning_steps', []):
         run_cleaning_step(step, paths_cfg, snapshot)