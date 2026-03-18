# amfs_tm/src/lib/feature/bank/trx/trx_net.py
import os
import numpy as np
import pandas as pd
from functools import reduce
from lib.feature.bank.trx.trx_lag import util, TrxLag

class TrxNet(TrxLag):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        # Initialize the updated TrxLag base class
        super().__init__(root, data_path, out_path, sep, snapshot)

    def create(self):
        """Processes current month transaction net flow."""
        print(f'--- Starting TrxNet.create: {self.snapshot} ---')
        
        # 1. Paths (Using absolute Volume paths from base class)
        input_path = os.path.join(self.abs_data_path, self.snapshot, f'axa_tran_net_{self.snapshot}.csv')
        output_path = os.path.join(self.abs_out_path, f'trxnet_{self.snapshot}_feat.csv')

        if not os.path.exists(input_path):
            print(f"Error: Source file not found at {input_path}")
            return

        # 2. Load and Clean
        trxnet = pd.read_csv(input_path, sep=self.sep)
        trxnet.columns = trxnet.columns.str.lower().str.strip()

        # Convert numerical columns
        num_cols = ['sum_db', 'sum_cd', 'num_db', 'num_cd', 'num_db_0', 'num_cd_0']
        util.to_numeric(trxnet, None, *[c for c in num_cols if c in trxnet.columns])
        trxnet = trxnet.fillna(0)

        # Cifno cleaning
        util.to_numeric(trxnet, np.int64, 'cifno')
        print(f'Processed {len(trxnet)} records.')

        # 3. Feature Logic
        trxnet['sum_cd_db_diff'] = trxnet['sum_cd'] - trxnet['sum_db']
        trxnet['num_cd_db_diff'] = trxnet['num_cd'] - trxnet['num_db']
        
        # Cleanup
        drop_cols = [c for c in ['cifno_01', 'cifno_15'] if c in trxnet.columns]
        if drop_cols:
            trxnet = trxnet.drop(drop_cols, axis=1)

        # 4. Safe Save
        self.safe_makedirs(os.path.dirname(output_path))
        trxnet.to_csv(output_path, index=False)
        print(f'Finish TrxNet.create for {self.snapshot}')

    def create_lag(self):
        """Joins historical data with 'Column Insurance' for missing months."""
        print(f'Target Month: {self.target_month}, Snapshot: {self.snapshot}')

        def _rename(x):
            return {'sum_db': f'sum_db_{x}', 'sum_cd': f'sum_cd_{x}'}

        usecols = ['cifno', 'sum_db', 'sum_cd']
        trx_6m = []

        # --- COLD START GUARD ---
        # Instead of list comprehension which crashes on missing files,
        # we loop and check existence.
        for month in self.timewindow:
            fpath = os.path.join(self.abs_out_path, f'trxnet_{month}_feat.csv')
            if os.path.exists(fpath):
                ds = pd.read_csv(fpath, usecols=usecols)
                ds = ds.rename(columns=_rename(month))
                util.to_numeric(ds, np.int64, 'cifno')
                trx_6m.append(ds)
            else:
                print(f"History missing for {month}. Skipping in lag merge.")

        if not trx_6m:
            print("No data available for TrxNet lag calculation.")
            return

        # Merge all available months
        df = reduce(lambda x, y: x.merge(y, on='cifno', how='left'), trx_6m).fillna(0)

        # 5. Calculation Logic (From TrxLag base class)
        # Note: These methods are now safe even if lm3 or lm6 are missing
        self._diff_between(df, newcol_pref='db', usecol_pref='sum_db', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='db', usecol_pref='sum_db', start='lm3', end='lm6')
        self._diff_between(df, newcol_pref='cd', usecol_pref='sum_cd', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='cd', usecol_pref='sum_cd', start='lm3', end='lm6')

        self._basic_stats(df, x='db')
        self._basic_stats(df, x='cd')

        # 6. Safe Save to Volume
        output_lag_path = os.path.join(self.abs_out_path, f'trxnet_{self.snapshot}_lag_feat.csv')
        self.safe_makedirs(os.path.dirname(output_lag_path))
        df.to_csv(output_lag_path, index=False)
        print(f"Lag features saved: {output_lag_path}")