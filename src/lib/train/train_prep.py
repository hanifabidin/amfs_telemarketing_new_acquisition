# amfs_tm/src/lib/train/train_prep.py
import os
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta

def prepare_training_data(main_config_path, train_config_path, args):
    """
    Prepares training/testing data by joining matrices with targets.
    Handles the 04_modelling / 05_campaigns directory structure.
    """
    with open(main_config_path, 'r') as f: main_cfg = yaml.safe_load(f)
    with open(train_config_path, 'r') as f: train_cfg = yaml.safe_load(f)

    root = main_cfg['PATHS']['base_path']
    mode = args.mode if hasattr(args, 'mode') else train_cfg['settings']['mode']
    target_col = train_cfg['target_logic']['target_col']
    dil = args.dil

    def process_snapshot(snap):
        """Helper to resolve the specific filename and folder for a snapshot."""
        snap_dt = datetime.strptime(snap, '%Y%m')
        campaign = (snap_dt + relativedelta(months=1)).strftime('%Y%m')
        
        # Resolved Path: 05_campaigns/{campaign}/02_matrix/matrix_{snap}_DIL_{dil}_campaign_{campaign}_final.csv
        matrix_filename = f"matrix_{snap}_DIL_{dil}_campaign_{campaign}_final.csv"
        m_path = os.path.join(root, '05_campaigns', campaign, '02_matrix', matrix_filename)
        
        if not os.path.exists(m_path):
            logging.warning(f"Matrix missing for {snap} at: {m_path}")
            return None
        
        # Load matrix with mixed-type hardening
        df = pd.read_csv(m_path, low_memory=False)
        df.columns = df.columns.str.lower().str.strip()
        
        if mode == "dummy":
            logging.info(f"Generating synthetic targets for {snap}")
            np.random.seed(42)
            rate = train_cfg['settings']['dummy_conversion_rate']
            df[target_col] = np.random.choice([0, 1], size=len(df), p=[1-rate, rate])
            return df
        else:
            # REAL Logic: Filter for the 'Called Population' only
            ct_path = os.path.join(root, train_cfg['settings']['call_tracking_file'].format(campaign=campaign))
            if not os.path.exists(ct_path):
                logging.warning(f"Call tracking file missing for {campaign}: {ct_path}")
                return None
            
            ct_df = pd.read_csv(ct_path, sep='\t', usecols=['cifno', 'callid'])
            ct_df.columns = ct_df.columns.str.lower()
            
            # Filter for customers who were contacted
            called_df = ct_df[ct_df['callid'] >= train_cfg['target_logic']['called_threshold']].copy()
            # Define conversion (e.g., 901+)
            conv_thresh = train_cfg['target_logic']['conversion_threshold']
            called_df[target_col] = (called_df['callid'] >= conv_thresh).astype(int)
            
            # Inner join ensures training data only includes those actually called
            return df.merge(called_df[['cifno', target_col]], on='cifno', how='inner')

    # --- Pipeline Execution ---
    if mode == "dummy":
        logging.info(f"🛠 Running Dummy Target Prep for Snapshot: {args.snapshot}")
        df_final = process_snapshot(args.snapshot)
        if df_final is None: return None
        
        out_path = os.path.join(root, train_cfg['paths']['train_output'].format(mode=mode, snapshot=args.snapshot))
        
    else:
        logging.info("📡 Compiling Real Multi-Snapshot Dataset...")
        # Stack history for training
        train_list = [process_snapshot(s) for s in train_cfg['settings']['train_snapshots']]
        train_df = pd.concat([d for d in train_list if d is not None], axis=0)
        
        out_path = os.path.join(root, train_cfg['paths']['train_output'].format(mode=mode, snapshot=args.snapshot))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        train_df.to_csv(out_path, index=False)
        
        # Prepare separate hold-out test set
        test_df = process_snapshot(train_cfg['settings']['test_snapshot'])
        if test_df is not None:
            test_path = os.path.join(root, train_cfg['paths']['test_output'].format(mode=mode, snapshot=args.snapshot))
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            test_df.to_csv(test_path, index=False)
            logging.info(f"Test set saved: {test_path}")

        df_final = train_df # For the final save call below

    # Save and return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_final.to_csv(out_path, index=False)
    logging.info(f"✅ Success: Data saved to {out_path}")
    
    return out_path