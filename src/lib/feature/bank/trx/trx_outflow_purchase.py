# amfs_tm/src/lib/feature/bank/trx/trx_outflow_purchase.py
import os
import numpy as np
import pandas as pd
from functools import reduce
from lib.feature.bank.trx.trx_lag import TrxLag, util

class TrxOutflowPayment(TrxLag):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        # Use updated TrxLag base class logic for absolute Volume paths
        super().__init__(root, data_path, out_path, sep, snapshot)

    def create(self):
        """Processes monthly purchase/payment transaction types."""
        print(f'--- Starting TrxOutflowPayment.create: {self.snapshot} ---')
        
        # Absolute Volume Paths
        input_path = os.path.join(self.abs_data_path, self.snapshot, f'trx_outflow_payment_purchase_{self.snapshot}.csv')
        output_path = os.path.join(self.abs_out_path, f'trxout_purchase_{self.snapshot}_feat.csv')

        if not os.path.exists(input_path):
            print(f"Skipping: Source file not found at {input_path}")
            return

        trxout_pur = pd.read_csv(input_path, sep=self.sep)
        util.to_numeric(trxout_pur, np.int64, 'customer_no')

        # Dynamically pivot based on transaction types found in data
        unique_purchase_types = np.unique(trxout_pur['transaction_type'].dropna())
        sum_df = trxout_pur[['customer_no']].drop_duplicates()

        print(f"Processing {len(unique_purchase_types)} purchase types for {len(sum_df)} customers.")

        for pur in unique_purchase_types:
            df = trxout_pur[trxout_pur['transaction_type'] == pur][['customer_no', 'frek', 'nominal']]
            df = df.rename(columns={'frek': f'{pur}_freq', 'nominal': f'{pur}_amt'})
            sum_df = sum_df.merge(df, on='customer_no', how='left')

        sum_df = sum_df.fillna(0)
        
        # Total frequency and amount aggregations
        freq_cols = [col for col in sum_df.columns if '_freq' in col]
        amt_cols = [col for col in sum_df.columns if '_amt' in col]
        
        sum_df['sum_purchase_freq'] = sum_df[freq_cols].sum(axis=1)
        sum_df['sum_purchase_amt'] = sum_df[amt_cols].sum(axis=1)

        # 1. Percentage distribution for each purchase type
        for col in amt_cols:
            new_col = f'{col}_per'
            # FIXED: Initialize as float (0.0) to prevent FutureWarning
            sum_df[new_col] = 0.0
            mask = sum_df['sum_purchase_amt'] > 0
            sum_df.loc[mask, new_col] = sum_df.loc[mask, col].astype(float) / sum_df.loc[mask, 'sum_purchase_amt']

        # 2. Safe Save to Volume
        self.safe_makedirs(os.path.dirname(output_path))
        sum_df.to_csv(output_path, index=False)
        print(f'Finish TrxOutflowPayment.create for {self.snapshot}')

    def create_lag(self):
        """Merges historical purchase amounts with Cold-Start Insurance."""
        print(f'Target Month: {self.target_month}, Snapshot: {self.snapshot}')

        def _rename(x):
            return {'sum_purchase_amt': f'purchase_amt_{x}'}

        usecols = ['customer_no', 'sum_purchase_amt']
        trx_6m = []

        # --- COLD START GUARD ---
        # Checks if previous months' feature files exist in the 03_features Volume
        for month in self.timewindow:
            fpath = os.path.join(self.abs_out_path, f'trxout_purchase_{month}_feat.csv')
            if os.path.exists(fpath):
                ds = pd.read_csv(fpath, usecols=usecols)
                ds = ds.rename(columns=_rename(month))
                util.to_numeric(ds, np.int64, 'customer_no')
                trx_6m.append(ds)
            else:
                print(f"History missing for {month}. Skipping in purchase lag.")

        if not trx_6m:
            print("No data available for TrxOutflowPayment lag calculation.")
            return

        # Merge available months based on customer_no
        df = reduce(lambda x, y: x.merge(y, on='customer_no', how='left'), trx_6m).fillna(0)

        # 3. Apply Lag calculations from TrxLag base class
        # These methods include 'Column Insurance' for missing month columns
        self._diff_between(df, newcol_pref='purchase_amt', usecol_pref='purchase_amt', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='purchase_amt', usecol_pref='purchase_amt', start='lm3', end='lm6')
        
        # basic_stats (min/max/mean) for purchase history
        self._basic_stats(df, x='purchase_amt', sum_prefix=False)

        # 4. Safe Save to Volume
        output_lag = os.path.join(self.abs_out_path, f'trxout_purchase_{self.snapshot}_lag_feat.csv')
        self.safe_makedirs(os.path.dirname(output_lag))
        df.to_csv(output_lag, index=False)
        print(f"Purchase Lag features saved: {output_lag}")