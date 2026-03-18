# src/lib/clean/clean_all.py
# import sys
import boto3
import os
import logging
# import pandas as pd # Keep pandas for potential direct use


# Assuming lib is in the python path or using relative imports
# from .. import util
from .. import s3_util
from . import clean_config_utils
from . import clean_utils # Import specific data cleaning functions


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

s3_client = boto3.client('s3')

def run_cleaning_step(step_config: dict, paths_cfg: dict, snapshot: str):
    """Runs a single cleaning step defined in the config."""
    bucket = paths_cfg.get('bucket_name')
    root_prefix = paths_cfg.get('root_prefix')
    raw_prefix = paths_cfg.get('raw_dir_prefix')
    proc_prefix = paths_cfg.get('processed_dir_prefix')

    original_rel_path = step_config.get('original_data_relative_path')
    raw_pattern = step_config.get('raw_data_filename_pattern')
    proc_pattern = step_config.get('processed_data_filename_pattern')
    cleaning_type = step_config.get('cleaning_type', 'standard') # Default to standard
    data_source = step_config.get('data_source', 'BM') # Default to BM

    if not all([original_rel_path, raw_pattern, proc_pattern, bucket, root_prefix, raw_prefix, proc_prefix]):
        logging.warning(f"Skipping incomplete cleaning step config: {step_config}")
        return

    # Determine original data source path
    if data_source == 'BM':
        original_base = paths_cfg.get('original_data_dir_BM', '')
    elif data_source == 'AMFS':
        original_base = paths_cfg.get('original_data_dir_AMFS', '')
    else:
        logging.warning(f"Unknown data_source '{data_source}' for step: {step_config}")
        original_base = '' # Or skip

    # Construct S3 URIs
    original_data_key = os.path.join(original_base, original_rel_path).format(snapshot=snapshot)
    # Handle potential leading slash if original_base is just '/'
    original_data_key = original_data_key.lstrip('/')

    raw_fname = raw_pattern.format(snapshot=snapshot)
    proc_fname = proc_pattern.format(snapshot=snapshot)

    raw_s3_uri = clean_config_utils.get_s3_uri(bucket, root_prefix, raw_prefix, f'data_from_{data_source}', snapshot, raw_fname)
    processed_s3_uri = clean_config_utils.get_s3_uri(bucket, root_prefix, proc_prefix, f'data_from_{data_source}', snapshot, proc_fname)

    # --- Step 1: Copy original file to raw location ---
    copy_successful = False
    try:
        logging.info(f"Attempting to copy s3://{bucket}/{original_data_key} to {raw_s3_uri}")
        _, raw_key = clean_config_utils.get_s3_parts(raw_s3_uri)
        if raw_key:
            s3_client.copy_object(CopySource={'Bucket': bucket, 'Key': original_data_key},
                                  Bucket=bucket, Key=raw_key)
            logging.info(f"Successfully copied original data to raw location.")
            copy_successful = True
        else:
            logging.error(f"Could not determine raw key for {raw_s3_uri}")

    except s3_client.exceptions.NoSuchKey:
         logging.error(f"Original file not found: s3://{bucket}/{original_data_key}")
    except Exception as e:
        logging.error(f"Error copying {original_data_key} to {raw_key}: {e}")

    if not copy_successful and cleaning_type != 'skip_copy': # Allow skipping copy if raw file guaranteed to exist
         logging.warning(f"Skipping cleaning for {raw_s3_uri} due to copy failure.")
         return

    # --- Step 2: Apply cleaning based on type ---
    logging.info(f"Applying cleaning type '{cleaning_type}' to {raw_s3_uri}")
    success = False
    separator = '|' if raw_fname.endswith('.txt') else ',' # Infer separator (adjust if needed)

    try:
        if cleaning_type == 'standard':
            success = clean_utils.apply_standard_cleaning(raw_s3_uri, processed_s3_uri, separator=separator)
        elif cleaning_type == 'balance':
            success = clean_utils.apply_balance_cleaning(raw_s3_uri, processed_s3_uri, separator=separator)
        elif cleaning_type == 'custdemo':
            success = clean_utils.apply_custdemo_cleaning(raw_s3_uri, processed_s3_uri, separator=separator)
        elif cleaning_type == 'trxdebit_edc':
            success = clean_utils.apply_trxdebit_edc_cleaning(raw_s3_uri, processed_s3_uri, snapshot, separator=separator)
        elif cleaning_type == 'trxnet':
             # Standard cleaning might be needed first if awk/sed did more than just structure
             # temp_uri = raw_s3_uri + ".tmp"
             # if clean_utils.apply_standard_cleaning(raw_s3_uri, temp_uri, separator=separator):
             #      success = clean_utils.apply_trxnet_cleaning(temp_uri, processed_s3_uri, separator=separator)
             #      # Optionally delete temp file
             # else:
             #      logging.error("Standard cleaning failed before trxnet processing.")
             # For now, assume trxnet processing works directly on raw/copied file
             success = clean_utils.apply_trxnet_cleaning(raw_s3_uri, processed_s3_uri, separator=separator)
        # Add elif for other specific cleaning types if needed
        else:
            logging.warning(f"Unknown cleaning_type: {cleaning_type}. Applying standard cleaning.")
            success = clean_utils.apply_standard_cleaning(raw_s3_uri, processed_s3_uri, separator=separator)

        if success:
            logging.info(f"Successfully processed {raw_s3_uri} to {processed_s3_uri}")
        else:
            logging.error(f"Failed to process {raw_s3_uri}")

    except Exception as e:
         logging.error(f"Unhandled exception during cleaning step for {raw_s3_uri}: {e}", exc_info=True)


def clean_all(main_config_path: str, clean_config_path: str, args):
    """Main function to orchestrate data copying and cleaning."""

    # --- Load Configs ---
    main_cfg = clean_config_utils.load_config(main_config_path)
    clean_cfg = clean_config_utils.load_config(clean_config_path)

    if not main_cfg or not clean_cfg:
        logging.critical("Failed to load configuration files. Exiting.")
        return

    paths_cfg = main_cfg.get('PATHS', {})
    bucket = paths_cfg.get('bucket_name')
    root_prefix = paths_cfg.get('root_prefix')
    raw_prefix = paths_cfg.get('raw_dir_prefix')
    proc_prefix = paths_cfg.get('processed_dir_prefix')

    if not all([bucket, root_prefix, raw_prefix, proc_prefix]):
        logging.critical("Missing critical PATHS configuration in main_config.yaml. Exiting.")
        return

    snapshot = args.snapshot # YYYYMM format

    # --- Ensure Base S3 Directories Exist ---
    # Check/Create base raw and processed directories for BM and AMFS for the snapshot
    for data_source in ['BM', 'AMFS']:
        raw_path = os.path.join(root_prefix, raw_prefix, f'data_from_{data_source}', snapshot)
        proc_path = os.path.join(root_prefix, proc_prefix, f'data_from_{data_source}', snapshot)
        s3_util.check_and_create_s3_directory(bucket, raw_path)
        s3_util.check_and_create_s3_directory(bucket, proc_path)

    # --- Process BM Cleaning Steps ---
    logging.info("--- Starting BM Data Cleaning Steps ---")
    bm_cleaning_steps = clean_cfg.get('cleaning_steps', [])
    if not bm_cleaning_steps:
        logging.warning("No BM cleaning steps defined in clean_config.yaml.")
    else:
        for step in bm_cleaning_steps:
             run_cleaning_step(step, paths_cfg, snapshot)
    logging.info("--- Finished BM Data Cleaning Steps ---")

    # # --- Process AMFS Cleaning Steps (using dedicated functions) ---
    # logging.info("--- Starting AMFS Data Cleaning Steps ---")
    # # Assuming these functions handle their own file copying and processing logic
    # # Pass the loaded configs and args to them
    # clean_amfs_dataset.cslean_call_tracking(main_cfg, clean_cfg, args)
    # clean_amfs_dataset.clean_all_status(main_cfg, clean_cfg, args)
    # clean_amfs_dataset.clean_future_leads(main_cfg, clean_cfg, args)
    # # The DNC final creation depends on features, maybe call it later?
    # # clean_amfs_dataset.create_do_not_call_final(main_cfg, clean_cfg, args)
    # logging.info("--- Finished AMFS Data Cleaning Steps (Note: DNC final creation might need feature dependencies) ---")

    # logging.info("--- Data Cleaning Process Completed ---")

# Example of how this might be called (e.g., from main.py or a SageMaker script)
# if __name__ == "__main__":
#     # Set up argparse similar to main.py
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--snapshot", required=True, help="Snapshot in YYYYMM format")
#     # Add other args if needed by cleaning steps
#     # parser.add_argument("--DIL", required=True)
#
#     # Define config paths (these could be passed as args or fixed)
#     main_config_path = '../../configs/main_config.yaml'
#     clean_config_path = '../../configs/clean_config.yaml'
#
#     args = parser.parse_args()
#
#     # Set up basic logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
#     clean_all(main_config_path, clean_config_path, args)