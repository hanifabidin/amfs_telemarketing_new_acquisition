# amfs_tm/src/lib/matrix/merge_all_features.py
import os
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from lib import util
from .impute_dummy import impute_and_dummy_matrix

def create_modeling_matrix(main_config_path, matrix_config_path, args):
    """
    Creates modeling matrix by merging features defined in matrix_config.yaml.
    """
    logging.info(f"--- Matrix Generation: {args.snapshot} ---")

    # 1. Load Configurations
    with open(main_config_path, 'r') as f:
        main_cfg = yaml.safe_load(f)
    with open(matrix_config_path, 'r') as f:
        matrix_cfg = yaml.safe_load(f)

    # Resolve core paths
    root = main_cfg['PATHS']['base_path']
    mode = matrix_cfg['settings']['running_mode']
    snapshot = args.snapshot
    dil = args.dil
    snap_short = snapshot[2:] if len(snapshot) == 6 else snapshot
    campaign = (datetime.strptime(snapshot, '%Y%m') + relativedelta(months=1)).strftime('%Y%m')

    # Resolve I/O paths
    input_path = os.path.join(root, matrix_cfg['settings']['input_file'].format(
        snapshot=snapshot, campaign=campaign, dil=dil))
    merge_path = os.path.join(root, matrix_cfg['settings']['merge_output'].format(
        snapshot=snapshot, campaign=campaign, dil=dil))

    # --- STAGE 1: MERGE ---
    if mode in ['all', 'merge']:
        logging.info("--- Stage: Merging Features ---")
        if not os.path.exists(input_path):
            logging.error(f"Filtered leads file not found: {input_path}")
            return None

        # Load the base population (Filtered Leads)
        df_matrix = pd.read_csv(input_path)
        df_matrix.columns = df_matrix.columns.str.lower().str.strip()
        util.to_numeric(df_matrix, np.int64, 'cifno')
        logging.info(f"Base leads loaded. Shape: {df_matrix.shape}")

        # Iterate through config-defined feature sets
        for name, conf in matrix_cfg['feature_sets'].items():
            if not conf.get('enabled'):
                continue

            f_path = os.path.join(root, conf['file'].format(snapshot=snapshot, snap_short=snap_short))
            if not os.path.exists(f_path):
                logging.warning(f"Feature file missing for {name}: {f_path}")
                continue

            logging.info(f"Merging: {name}")
            
            # Column selection logic
            usecols = conf.get('usecols', [])
            key = conf.get('key', 'cifno')
            
            # Load data (only specified columns to save memory)
            if usecols:
                if key not in usecols:
                    usecols.append(key)
                df_feat = pd.read_csv(f_path, usecols=usecols)
            else:
                df_feat = pd.read_csv(f_path)

            df_feat.columns = df_feat.columns.str.lower().str.strip()

            # Standardize join key
            join_key = key.lower()
            if join_key in df_feat.columns and join_key != 'cifno':
                df_feat = df_feat.rename(columns={join_key: 'cifno'})
            
            util.to_numeric(df_feat, np.int64, 'cifno')

            # Left join to preserve leads population
            df_matrix = df_matrix.merge(df_feat, on='cifno', how='left')

        # Save Raw Merged Matrix
        os.makedirs(os.path.dirname(merge_path), exist_ok=True)
        df_matrix.to_csv(merge_path, index=False)
        logging.info(f"Raw Merge complete. Saved to: {merge_path}")
    
    # --- STAGE 2: IMPUTE & DUMMY ---
    if mode in ['all', 'impute']:
        # This script uses the raw merge file as its input
        impute_and_dummy_matrix(main_config_path, matrix_config_path, args)

    return merge_path