# amfs_tm/src/lib/segmentation/apply_flag.py
import os
import numpy as np
import pandas as pd
from lib import obj, util

class ApplyFlag(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        """
        Updated for Databricks Volumes.
        root: Absolute path to project root (e.g., /Volumes/catalog/schema/vol/amfs_tm)
        """
        super().__init__(root, data_path, out_path, sep, snapshot)

        self.snapshot = snapshot
        
        # Paths updated to use standard Volume structure
        self.cif_acct_path = os.path.join(self.root, data_path, '{snapshot}/cif_acct_mapping_{snapshot}.csv')
        self.cif_acct_complete = os.path.join(self.root, 'data/03_features/BM_features/cif_acct_mapping_{snapshot}_Complete.csv')
        
        self.trx_net_path = os.path.join(self.root, data_path, '{snapshot}/axa_tran_net_{snapshot}.csv')
        self.all_status_path = os.path.join(self.root, 'data/02_processed/data_from_AMFS/{snapshot}/ALL_STATUS_{snapshot}.csv')
        self.call_tracking_path = os.path.join(self.root, 'data/02_processed/data_from_AMFS/{snapshot}/call_tracking_savings_{snapshot}.csv')
        self.flag_path = os.path.join(self.root, 'data/03_features/BM_features/cif_with_flag_{snapshot}_active_complete.csv')

    def accumulate(self):
        """Merges current and previous month CIF mappings to ensure coverage."""
        prev_month = util.last_month(self.snapshot, out_format='%Y%m')
        curr_cif_path = self.cif_acct_complete.format(snapshot=self.snapshot)

        print(f'Processing accumulation for {self.snapshot} and {prev_month}')
        
        prev_cif_df = self.read_cif2acct(snapshot=prev_month)
        curr_cif_df = self.read_cif2acct(snapshot=self.snapshot)
        
        accumulate_df = pd.concat([prev_cif_df, curr_cif_df]).drop_duplicates().reset_index(drop=True)
        
        # Ensure output directory exists in Volume
        os.makedirs(os.path.dirname(curr_cif_path), exist_ok=True)
        accumulate_df.to_csv(curr_cif_path, index=False, sep=',')
        
        accumulate_df = accumulate_df.rename(columns={'rekening': 'acct_no'})
        return accumulate_df, curr_cif_df.cifno.unique()

    def read_cif2acct(self, snapshot):
        """Reads and standardizes CIF to Account mapping files."""
        in_path = self.cif_acct_path.format(snapshot=snapshot)
        print(f'Reading mapping: {in_path}')
        
        if not os.path.exists(in_path):
            print(f"Warning: Mapping file not found at {in_path}")
            return pd.DataFrame(columns=['rekening', 'cifno'])

        try:
            # Handle variable column naming in raw files
            cif2acct = pd.read_csv(in_path, sep=self.sep, usecols=['rekening', 'cifno'])
        except ValueError:
            cif2acct = pd.read_csv(in_path, sep=self.sep, usecols=['rekening', 'cif'])
            cif2acct = cif2acct.rename(columns={'cif': 'cifno'})

        util.to_numeric(cif2acct, np.int64, 'rekening', 'cifno')
        return cif2acct

    def get_axa_cif(self, cif2acct_df=None):
        """Creates a flag for active insurance policies."""
        all_status_path = self.all_status_path.format(snapshot=self.snapshot)
        
        if not os.path.exists(all_status_path):
            print(f"Warning: Status file {all_status_path} not found.")
            return pd.DataFrame(columns=['cifno', 'axa_pol_flag'])

        all_status_lm = pd.read_csv(all_status_path, sep=',', usecols=['acct_no'])
        all_status_lm = all_status_lm.drop_duplicates().reset_index(drop=True)
        util.to_numeric(all_status_lm, np.int64, 'acct_no')

        if cif2acct_df is None:
            all_status_lm['axa_pol_flag'] = 1
            return all_status_lm.astype({'axa_pol_flag': np.int8})
        else:
            all_status_lm = cif2acct_df.merge(all_status_lm, on='acct_no', how='inner')
            axa_cif = all_status_lm[['cifno']].drop_duplicates().reset_index(drop=True)
            axa_cif['axa_pol_flag'] = 1
            return axa_cif.astype({'axa_pol_flag': np.int8})

    def get_trx_cif(self):
        """Creates a flag for active bank transactions."""
        trx_path = self.trx_net_path.format(snapshot=self.snapshot)
        
        if not os.path.exists(trx_path):
            return pd.DataFrame(columns=['cifno', 'bank_active_flag'])

        trx_db = pd.read_csv(trx_path, sep='|', usecols=['cifno'])
        bank_cif = trx_db.drop_duplicates().reset_index(drop=True)
        bank_cif['bank_active_flag'] = 1
        return bank_cif.astype({'bank_active_flag': np.int8})
        
    def get_calltracking_cif(self, cif2acct_df=None):
        """Extracts contacted and converted flags from call tracking data."""
        path = self.call_tracking_path.format(snapshot=self.snapshot)
        
        if not os.path.exists(path):
            return pd.DataFrame(columns=['cifno', 'contacted', 'converted'])

        call_tracking = pd.read_csv(path).rename(columns={'accnumber': 'acct_no'})
        util.to_numeric(call_tracking, np.int64, 'acct_no')

        # Logic based on call disposition IDs
        call_tracking['contacted'] = np.where(call_tracking['callid'] >= 201, 1, 0)
        call_tracking['converted'] = np.where(call_tracking['callid'] >= 901, 1, 0)

        if cif2acct_df is not None:
            call_tracking = cif2acct_df.merge(call_tracking, on='acct_no', how='inner')   
            call_tracking = call_tracking[['cifno', 'contacted', 'converted']]
        
        return call_tracking

    def create(self):
        """Main execution to create the CIF flag master table."""
        cif_acc_df, active_cif = self.accumulate()

        axa_cif = self.get_axa_cif()
        trx_cif = self.get_trx_cif()
        target_cif = self.get_calltracking_cif()

        # Join all flags
        cif_acc_df = cif_acc_df.merge(trx_cif, on='cifno', how='left')
        cif_acc_df = cif_acc_df.merge(axa_cif, on='acct_no', how='left')
        cif_acc_df = cif_acc_df.merge(target_cif, on='acct_no', how='left')

        # Fill NAs and finalize types
        fill_cols = ['bank_active_flag', 'axa_pol_flag', 'contacted', 'converted']
        cif_acc_df[fill_cols] = cif_acc_df[fill_cols].fillna(0).astype(np.int8)

        # Aggregate at CIF level
        cif_acc_df = cif_acc_df.groupby('cifno').agg('max').reset_index()

        # Filter for the specific snapshot's active CIFs
        cif_acc_df = cif_acc_df[cif_acc_df.cifno.isin(active_cif)]
        
        output_path = self.flag_path.format(snapshot=self.snapshot)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cif_acc_df.to_csv(output_path, sep=',', index=False)