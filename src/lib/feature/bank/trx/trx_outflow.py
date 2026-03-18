# amfs_tm/src/lib/feature/bank/trx/trx_outflow.py
import os
import numpy as np
import pandas as pd
from functools import reduce
from lib.feature.bank.trx.trx_lag import TrxLag, util

class TrxOutflow(TrxLag):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        # Initialize the updated TrxLag base class logic
        super().__init__(root, data_path, out_path, sep, snapshot)

    def create(self):
        """Processes monthly outflow transactions and calculates percentage distributions."""
        print(f'--- Starting TrxOutflow.create: {self.snapshot} ---')
        
        # Absolute Volume Paths
        input_path = os.path.join(self.abs_data_path, self.snapshot, f'trx_outflow_{self.snapshot}.csv')
        output_path = os.path.join(self.abs_out_path, f'trxout_{self.snapshot}_feat.csv')

        if not os.path.exists(input_path):
            print(f"Skipping: Source file not found at {input_path}")
            return

        trxout = pd.read_csv(input_path, sep=self.sep)
        trxout = trxout.fillna(0)

        # Standardizing CIF column
        print(f'Length before cleaning: {len(trxout)}')
        util.to_numeric(trxout, np.int64, 'cifno_pengirim')
        print(f'Length after cleaning: {len(trxout)}')

        # Aggregate by CIF
        # Drop non-numeric identifying columns before grouping
        cols_to_drop = [c for c in ['rekening_pengirim'] if c in trxout.columns]
        trxout = trxout.drop(cols_to_drop, axis=1, errors='ignore').groupby(['cifno_pengirim'], as_index=False).sum()

        # 1. Percentage of Total Amount
        trx_amt_cols = ['withdrawal', 'transfer_to_mandiri', 'transfer_to_others', 'bill_payment', 'trx_others']
        for col in trx_amt_cols:
            if col in trxout.columns:
                new_col = col + '_per'
                # FIXED: Initialize as float (0.0) to prevent FutureWarning
                trxout[new_col] = 0.0
                mask = trxout['trx_total'] > 0
                trxout.loc[mask, new_col] = trxout.loc[mask, col].astype(float) / trxout.loc[mask, 'trx_total']

        # 2. Percentage of Channel Amount
        trx_chnl_amt_cols = ['ib_amt', 'mb_amt', 'branch_amt', 'nm_atm_amt', 'm_atm_amt', 'mcm_amt', 'chnl_others_amt']
        existing_chnl_cols = [c for c in trx_chnl_amt_cols if c in trxout.columns]
        
        if existing_chnl_cols:
            trxout['chnl_amt_sum'] = trxout[existing_chnl_cols].sum(axis=1)
            for col in existing_chnl_cols:
                new_col = col + '_per'
                # FIXED: Initialize as float (0.0)
                trxout[new_col] = 0.0
                mask = trxout['chnl_amt_sum'] > 0
                trxout.loc[mask, new_col] = trxout.loc[mask, col].astype(float) / trxout.loc[mask, 'chnl_amt_sum']
            
            trxout = trxout.drop(['chnl_amt_sum'], axis=1)

        # Save to Volume
        self.safe_makedirs(os.path.dirname(output_path))
        trxout.to_csv(output_path, index=False)
        print(f'Finish TrxOutflow.create for {self.snapshot}')

    def create_lag(self):
        """Merges 6 months of outflow totals with Column Insurance."""
        print(f'Target Month: {self.target_month}, Snapshot: {self.snapshot}')

        def _rename(x):
            return {'trx_total': f'trxout_{x}'}

        usecols = ['cifno_pengirim', 'trx_total']
        trx_6m = []

        # --- COLD START GUARD ---
        for month in self.timewindow:
            # Look for feature files created in the previous 'create' step
            fpath = os.path.join(self.abs_out_path, f'trxout_{month}_feat.csv')
            if os.path.exists(fpath):
                ds = pd.read_csv(fpath, usecols=usecols)
                ds = ds.rename(columns=_rename(month))
                util.to_numeric(ds, np.int64, 'cifno_pengirim')
                trx_6m.append(ds)
            else:
                print(f"History missing for {month}. Skipping in outflow lag.")

        if not trx_6m:
            print("No data available for TrxOutflow lag calculation.")
            return

        # Merge available months
        df = reduce(lambda x, y: x.merge(y, on='cifno_pengirim', how='left'), trx_6m).fillna(0)

        # 3. Calculation Logic from TrxLag
        # These are now safe against missing columns due to the updated TrxLag base class
        self._diff_between(df, newcol_pref='trxout', usecol_pref='trxout', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='trxout', usecol_pref='trxout', start='lm3', end='lm6')
        
        # basic_stats (min/max/mean) across 3m and 6m
        self._basic_stats(df, x='trxout', sum_prefix=False)

        # Safe Save to Volume
        output_lag_path = os.path.join(self.abs_out_path, f'trxout_{self.snapshot}_lag_feat.csv')
        self.safe_makedirs(os.path.dirname(output_lag_path))
        df.to_csv(output_lag_path, index=False)
        print(f"Outflow Lag features saved: {output_lag_path}")