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
    Prepares training/testing data. 
    Dummy: One snapshot, synthetic targets.
    Real: Stacks train_snapshots, filters for 'called only', creates test set.
    """
    with open(main_config_path, 'r') as f: main_cfg = yaml.safe_load(f)
    with open(train_config_path, 'r') as f: train_cfg = yaml.safe_load(f)

    root = main_cfg['PATHS']['base_path']
    mode = train_cfg['settings']['mode']
    target_col = train_cfg['target_logic']['target_col']
    
    def process_snapshot(snap):
        """Helper to load matrix and join with call tracking results."""
        snap_dt = datetime.strptime(snap, '%Y%m')
        campaign = (snap_dt + relativedelta(months=1)).strftime('%Y%m')
        
        # Load final matrix for that month
        m_path = os.path.join(root, '04_campaigns', campaign, '02_matrix', f"matrix_final_{snap}.csv")
        if not os.path.exists(m_path):
            logging.warning(f"Matrix missing for {snap}: {m_path}")
            return None
        
        df = pd.read_csv(m_path)
        
        if mode == "dummy":
            # Synthetic targets for the whole population
            np.random.seed(42)
            rate = train_cfg['settings']['dummy_conversion_rate']
            df[target_col] = np.random.choice([0, 1], size=len(df), p=[1-rate, rate])
            return df
        else:
            # REAL Logic: Join with call tracking and filter
            ct_path = os.path.join(root, train_cfg['settings']['call_tracking_file'].format(campaign=campaign))
            if not os.path.exists(ct_path):
                logging.warning(f"Call tracking missing for {campaign}: {ct_path}")
                return None
            
            ct_df = pd.read_csv(ct_path, sep='\t', usecols=['cifno', 'callid'])
            ct_df.columns = ct_df.columns.str.lower()
            
            # Filter: Only take customers who were actually called
            called_thresh = train_cfg['target_logic']['called_threshold']
            called_df = ct_df[ct_df['callid'] >= called_thresh].copy()
            
            # Label: Converted (1) vs Not Converted (0)
            conv_thresh = train_cfg['target_logic']['conversion_threshold']
            called_df[target_col] = (called_df['callid'] >= conv_thresh).astype(int)
            
            # Join and return only the called population
            df_final = df.merge(called_df[['cifno', target_col]], on='cifno', how='inner')
            logging.info(f"Snapshot {snap}: {len(df_final)} customers were called.")
            return df_final

    # --- EXECUTION ---
    if mode == "dummy":
        logging.info("🛠 Generating Single-Snapshot Dummy Data...")
        df_dummy = process_snapshot(args.snapshot)
        out_path = os.path.join(root, train_cfg['paths']['train_output'].format(mode=mode))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_dummy.to_csv(out_path, index=False)
        return out_path

    else:
        logging.info("📡 Compiling Real Training Dataset...")
        train_snaps = train_cfg['settings']['train_snapshots']
        train_list = [process_snapshot(s) for s in train_snaps]
        train_df = pd.concat([d for d in train_list if d is not None], axis=0)
        
        train_path = os.path.join(root, train_cfg['paths']['train_output'].format(mode=mode))
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_df.to_csv(train_path, index=False)
        
        logging.info("📡 Compiling Real Testing Dataset...")
        test_snap = train_cfg['settings']['test_snapshot']
        test_df = process_snapshot(test_snap)
        
        test_path = os.path.join(root, train_cfg['paths']['test_output'].format(mode=mode))
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        test_df.to_csv(test_path, index=False)
        
        return train_path