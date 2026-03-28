# amfs_tm/src/lib/segmentation/apply_flag.py
import os
import numpy as np
import pandas as pd
from lib import obj, util

class ApplyFlag(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)

        # 1. Source Paths (02_processed Volume)
        # Using self.root to ensure we stay within the absolute Volume boundaries
        self.processed_root = os.path.join(self.root, '02_processed')
        
        self.cif_acct_path = os.path.join(self.processed_root, 'data_from_BM', 
                                          self.snapshot, f'cif_acct_mapping_{self.snapshot}.csv')
        self.trx_net_path = os.path.join(self.processed_root, 'data_from_BM', 
                                         self.snapshot, f'axa_tran_net_{self.snapshot}.csv')
        self.all_status_path = os.path.join(self.processed_root, 'data_from_AMFS', 
                                            self.snapshot, f'ALL_STATUS_{self.snapshot}.csv')
        self.call_tracking_path = os.path.join(self.processed_root, 'data_from_AMFS', 
                                               self.snapshot, f'call_tracking_savings_{self.snapshot}.csv')

        # 2. Corrected Output Path: Directly inside 03_features
        self.output_dir = os.path.join(self.root, '03_features', 'cif_flag_mapping')
        
        self.cif_acct_complete = os.path.join(self.output_dir, f'cif_acct_mapping_{self.snapshot}_Complete.csv')
        self.flag_path = os.path.join(self.output_dir, f'cif_with_flag_{self.snapshot}_active_complete.csv')

    def read_cif2acct(self, snapshot=None):
        """Reads pipe-separated CIF mapping with standardized keys."""
        snap = snapshot if snapshot else self.snapshot
        in_path = os.path.join(self.processed_root, 'data_from_BM', snap, f'cif_acct_mapping_{snap}.csv')
        
        if not os.path.exists(in_path):
            print(f"Warning: Mapping file not found at {in_path}")
            return pd.DataFrame()

        # Handle Pipe Separator
        df = pd.read_csv(in_path, sep='|')
        df.columns = df.columns.str.lower().str.strip()
        
        # Column Aliasing for cifno and rekening
        if 'cif' in df.columns and 'cifno' not in df.columns:
            df = df.rename(columns={'cif': 'cifno'})
        if 'rekening' not in df.columns and 'acct_no' in df.columns:
            df = df.rename(columns={'acct_no': 'rekening'})

        # Verify columns exist after normalization
        if 'rekening' not in df.columns or 'cifno' not in df.columns:
            print(f"Error: Required columns missing in {in_path}. Found: {df.columns.tolist()}")
            return pd.DataFrame()

        # Type Hardening
        util.to_numeric(df, np.int64, 'rekening', 'cifno')
        return df[['rekening', 'cifno']]

    def accumulate(self):
        """Accumulates CIF mappings. If previous month is missing, uses current only."""
        prev_month = util.last_month(self.snapshot, out_format='%Y%m')
        
        print(f'--- Starting Accumulation for {self.snapshot} ---')
        curr_cif_df = self.read_cif2acct(snapshot=self.snapshot)
        
        if curr_cif_df.empty:
            print("CRITICAL: Current snapshot mapping is empty.")
            return pd.DataFrame(), []

        prev_cif_df = self.read_cif2acct(snapshot=prev_month)
        
        if prev_cif_df.empty:
            print(f"Cold Start: Previous month {prev_month} missing. Skipping history join.")
            accumulate_df = curr_cif_df
        else:
            accumulate_df = pd.concat([prev_cif_df, curr_cif_df]).drop_duplicates()
        
        accumulate_df = accumulate_df.reset_index(drop=True)
        
        # Ensure the new /03_features/cif_flag_mapping folder exists
        self.safe_makedirs(self.output_dir)
        accumulate_df.to_csv(self.cif_acct_complete, index=False)
        
        return accumulate_df.rename(columns={'rekening': 'acct_no'}), curr_cif_df.cifno.unique()

    def get_axa_cif(self):
        """Flags accounts with active AXA policies."""
        path = self.all_status_path
        if not os.path.exists(path):
            return pd.DataFrame(columns=['acct_no', 'axa_pol_flag'])

        all_status = pd.read_csv(path, usecols=['acct_no']).drop_duplicates()
        util.to_numeric(all_status, np.int64, 'acct_no')
        all_status['axa_pol_flag'] = 1
        return all_status.astype({'axa_pol_flag': np.int8})

    def get_trx_cif(self):
        """Flags CIFs with active bank transactions."""
        path = self.trx_net_path
        if not os.path.exists(path):
            return pd.DataFrame(columns=['cifno', 'bank_active_flag'])

        # Pipe separated trx file
        trx_db = pd.read_csv(path, sep='|', usecols=['cifno']).drop_duplicates()
        util.to_numeric(trx_db, np.int64, 'cifno')
        trx_db['bank_active_flag'] = 1
        return trx_db.astype({'bank_active_flag': np.int8})
        
    def get_calltracking_cif(self):
        """Flags contacted and converted accounts from call tracking."""
        path = self.call_tracking_path
        if not os.path.exists(path):
            return pd.DataFrame(columns=['acct_no', 'contacted', 'converted'])

        df = pd.read_csv(path).rename(columns={'accnumber': 'acct_no', 'callid': 'status_id'})
        util.to_numeric(df, np.int64, 'acct_no')

        # Logic for disposition codes
        df['contacted'] = (df['status_id'] >= 201).astype(np.int8)
        df['converted'] = (df['status_id'] >= 901).astype(np.int8)
        
        return df[['acct_no', 'contacted', 'converted']]

    def create(self):
        """Main execution loop for Master Flag creation."""
        print(f"--- Starting ApplyFlag.create: {self.snapshot} ---")
        
        cif_acc_df, active_cif_list = self.accumulate()
        if cif_acc_df.empty:
            return

        # Perform Merges
        df = cif_acc_df.merge(self.get_trx_cif(), on='cifno', how='left')
        df = df.merge(self.get_axa_cif(), on='acct_no', how='left')
        df = df.merge(self.get_calltracking_cif(), on='acct_no', how='left')

        # Cleanup NAs
        fill_cols = ['bank_active_flag', 'axa_pol_flag', 'contacted', 'converted']
        df[fill_cols] = df[fill_cols].fillna(0).astype(np.int8)

        # Aggregate to CIF level (max flag value)
        cif_final = df.groupby('cifno', as_index=False).agg({
            'bank_active_flag': 'max',
            'axa_pol_flag': 'max',
            'contacted': 'max',
            'converted': 'max'
        })

        # Keep only the active population
        cif_final = cif_final[cif_final['cifno'].isin(active_cif_list)]
        
        # Save to Volume
        cif_final.to_csv(self.flag_path, index=False)
        print(f"Success: Master Flags saved to {self.flag_path}")