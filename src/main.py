
# src/main.py
import argparse
import logging
import os
import sys

try:
    from lib.clean.clean_all import clean_all
    from lib.feature.features_generation import features_generation
    from lib.matrix.merge_all_features import create_modeling_matrix_pandas_dynamic
    from lib.matrix.impute_dummy import impute_and_dummy_matrix
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure 'src' directory is in the Python path or adjust imports.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define relative paths to YAML config files (assuming they are in ../configs/)
# In SageMaker, these might be passed as hyperparameters or mounted at specific paths
CONFIG_DIR = '../configs'
MAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, 'main_config.yaml')
CLEAN_CONFIG_PATH = os.path.join(CONFIG_DIR, 'clean_config.yaml')
FEATURES_CONFIG_PATH = os.path.join(CONFIG_DIR, 'features_config.ini')
MATRIX_CONFIG_PATH = os.path.join(CONFIG_DIR, 'matrix_config.ini') 

def run_pipeline(args):
    """Runs the AMFS TM pipeline steps."""
    logging.info("=== Starting AMFS TM Pipeline ===")
    logging.info(f"Snapshot: {args.snapshot}, DIL: {args.DIL}, Product: {args.product}")

    config_files = [
        MAIN_CONFIG_PATH, CLEAN_CONFIG_PATH, FEATURES_CONFIG_PATH, MATRIX_CONFIG_PATH
    ]
    for config_file in config_files:
        if not os.path.exists(config_file):
            logging.warning(f"Configuration file not found: {config_file}. Pipeline step might fail.")
            # Consider exiting if critical configs are missing

    # Step 1: Data Cleaning
    logging.info("\n--- Running Step 1: Data Cleaning ---")
    try:
        clean_all(MAIN_CONFIG_PATH, CLEAN_CONFIG_PATH, args)
    except Exception as e:
        logging.error(f"Data Cleaning failed: {e}", exc_info=True)
        # Decide whether to stop pipeline or continue
        return

    # Step 2: Feature Engineering
    logging.info("\n--- Running Step 2: Feature Engineering ---")
    try:
        # Assuming features_generation now takes YAML paths
        features_generation(MAIN_CONFIG_PATH, FEATURES_CONFIG_PATH, args)
    except Exception as e:
        logging.error(f"Feature Engineering failed: {e}", exc_info=True)
        return

    # Step 3: Matrix Generation
    logging.info("\n--- Running Step 3a: Matrix Generation, merge features---")
    try:
        # Assuming matrix_gen now takes YAML paths
        create_modeling_matrix_pandas_dynamic(MAIN_CONFIG_PATH, args)
    except Exception as e:
        logging.error(f"Matrix Generation merge failed: {e}", exc_info=True)

    logging.info("\n--- Running Step 3b: Matrix Generation, impute and dummified ---")
    try:
        # Assuming matrix_gen now takes YAML paths
        impute_and_dummy_matrix(MAIN_CONFIG_PATH, args)
    except Exception as e:
        logging.error(f"Matrix Generation impute dummy failed: {e}", exc_info=True)
        # return
    # # Step 5: Predict
    # logging.info("\n--- Running Step 5: Prediction ---")
    # try:
    #     # Assuming predict_apply now takes YAML paths
    #     predict_apply(args, MAIN_CONFIG_PATH, PREDICT_CONFIG_PATH, FEATURE_PREPROCESS_CONFIG_PATH)
    # except Exception as e:
    #     logging.error(f"Prediction failed: {e}", exc_info=True)
    #     # return

    # # Step 6. Leads Generation (Uncomment and update when ready)
    # # logging.info("\n--- Running Step 6: Leads Generation ---")
    # # try:
    # #     # Assuming leads_gen now takes YAML path
    # #     # leads_gen(LEADS_CONFIG_PATH, args)
    # #     pass # Placeholder
    # # except Exception as e:
    # #     logging.error(f"Leads Generation failed: {e}", exc_info=True)
    # #     # return

    logging.info("\n=== AMFS TM Pipeline Finished ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AMFS TM pipeline.")
    parser.add_argument("--snapshot", required=True, help="Snapshot date in YYYYMM format.")
    parser.add_argument("--DIL", required=False, help="DIL value for the campaign.")
    # Make product optional as in original code, handle None if necessary in predict step
    parser.add_argument("--product", help="Product code (e.g., MSK, MPPT).")
    # Add argument for config directory if needed, e.g., to override ../configs
    # parser.add_argument("--config-dir", default="../configs", help="Path to the configuration directory.")

    args = parser.parse_args()

    # Update config paths if --config-dir argument is used
    # CONFIG_DIR = args.config_dir
    # MAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, 'main_config.yaml')
    # ... update other paths ...

    run_pipeline(args)