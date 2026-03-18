# amfs_tm/src/lib/feature/bank/phsumm.py
import os
from functools import reduce
import numpy as np
import pandas as pd
from lib import obj, util

class BankPH(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)
        # max_date is used to clip future-dated account openings
        self.max_date = util.last_day(snapshot, out_format='%Y-%m-%d')

    def __handle_numerical(self, data, ph_cols):
        """Cleans and caps product holding counts without Numba."""
        for col in ph_cols:
            if col in data.columns:
                # Original logic: cap at 5, uint8 for memory efficiency
                data[col] = data[col].fillna(0).clip(upper=5).astype(np.uint8)

        # Standardizing loan and total columns
        data['loanmort'] = data.get('loanmort', 0).fillna(0)
        data['loanall'] = data.get('loanall', 0).fillna(0)
        data['totph'] = data.get('totph', 15).fillna(15).clip(upper=15)

        # TOT_SAVING: Sum of T1-T9, capped at 11
        s_cols = [f't{i}' for i in range(1, 10) if f't{i}' in data.columns]
        data['tot_saving'] = data[s_cols].sum(axis=1).clip(upper=11).astype(np.uint8)
        
        # TOT_LOAN: Sum of T12-T17, capped at 4
        l_cols = [f't{i}' for i in range(12, 18) if f't{i}' in data.columns]
        data['tot_loan'] = data[l_cols].sum(axis=1).clip(upper=4)

    def __handle_date(self, data, date_cols, date_format):
        """Validates dates and calculates account vintages."""
        for col in date_cols:
            if col not in data.columns: continue
            
            # Clean outliers (Original logic: 1917-1980 becomes 1980)
            data.loc[(data[col] < '1980-01-01') & (data[col] > '1917-01-01'), col] = '1980-01-01'
            data.loc[~data[col].between('1980-01-01', self.max_date), col] = np.nan
            data[col] = pd.to_datetime(data[col], format=date_format, errors='coerce')

        # Calculate First and Last opening dates across all account types
        f_cols = [c for c in ['sfstopn', 'dfstopn', 'gfstopn', 'cfstopn'] if c in data.columns]
        l_cols = [c for c in ['slstopn', 'dlstopn', 'glstopn', 'clstopn'] if c in data.columns]
        
        data['all_accts_first_date'] = data[f_cols].min(axis=1)
        data['all_accts_last_date'] = data[l_cols].max(axis=1)

        ref_date = pd.to_datetime(self.max_date)
        
        # Calculate Vintages (Months)
        if 'slstopn' in data.columns:
            data['saving_acct_vint_l'] = (ref_date - data['slstopn']).dt.days / 30.44
        if 'sfstopn' in data.columns:
            data['saving_acct_vint_f'] = (ref_date - data['sfstopn']).dt.days / 30.44

        data['all_acct_vint_l'] = (ref_date - data['all_accts_last_date']).dt.days / 30.44
        data['all_acct_vint_f'] = (ref_date - data['all_accts_first_date']).dt.days / 30.44

        # Cleanup: Remove intermediate date columns
        data.drop([c for c in date_cols + ['all_accts_first_date', 'all_accts_last_date'] if c in data.columns], 
                  axis=1, inplace=True)

    def create(self):
        """Monthly PHSUMM feature extraction."""
        input_path = os.path.join(self.abs_data_path, self.snapshot, f'PHSUMM_{self.snapshot}.csv')
        output_path = os.path.join(self.abs_out_path, f'PHSUMM_{self.snapshot}_feat.csv')

        if not os.path.exists(input_path):
            print(f"PHSUMM input missing: {input_path}")
            return

        data = pd.read_csv(input_path, sep=self.sep)
        data.columns = data.columns.str.lower().str.strip()
        
        # Mapping variants
        data = data.rename(columns={'loanmort_s': 'loanmort', 'loanall_s': 'loanall', 'cifno': 'cifno'})
        util.to_numeric(data, np.int64, 'cifno')

        ph_count_cols = [f't{i}' for i in range(1, 19)]
        time_cols = ['sfstopn', 'slstopn', 'dfstopn', 'dlstopn', 'gfstopn', 'glstopn', 'cfstopn', 'clstopn']

        # Ensure numeric types
        for col in [c for c in ph_count_cols + ['totph', 'loanmort', 'loanall'] if c in data.columns]:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # Filter out rows where only T12 (Insurance) exists
        if 'totph' in data.columns and 't12' in data.columns:
            data = data[data['totph'] != data['t12']].reset_index(drop=True)

        self.__handle_numerical(data, ph_count_cols)
        self.__handle_date(data, time_cols, '%Y-%m-%d')

        self.safe_makedirs(os.path.dirname(output_path))
        data.to_csv(output_path, index=False)
        print(f"Saved PHSUMM features for {self.snapshot}")

    def create_loan(self):
        """Temporal logic for Loan Increments with Column Insurance."""
        target_month = util.next_month(self.snapshot, out_format='%Y%m')
        timewindow = util.month_range(target_month, period=4, out_format='%Y%m')
        posfixes = {m: ('t' if i == 0 else f't-{i}') for i, m in enumerate(timewindow)}
        
        expected_flags = ['loan_recent_inc_flag', 'loan_recent_inc_flag2', 'loan_recent_inc_flag3']

        def __process(snapshot):
            path = os.path.join(self.abs_data_path, snapshot, f'PHSUMM_{snapshot}.csv')
            if not os.path.exists(path): return None
            
            ds = pd.read_csv(path, sep=self.sep)
            ds.columns = ds.columns.str.lower().str.strip()
            ds = ds.rename(columns={'loanall_s': 'loanall'})
            
            usecols = ['cifno', 'loanall', 't12', 't13', 't14', 't15', 't16', 't17', 'totph']
            ds = ds[[c for c in usecols if c in ds.columns]].copy()
            
            p = posfixes[snapshot]
            ds = ds.rename(columns={c: f"{c}_{p}" for c in ds.columns if c != 'cifno'})
            # CRITICAL: Ensure cifno is int64 for safe merging
            util.to_numeric(ds, np.int64, 'cifno')
            return ds

        datasets = [d for d in [__process(m) for m in timewindow] if d is not None]
        
        # --- COLUMN INSURANCE START ---
        if len(datasets) < 2:
            print("Cold Start: Missing history for Loan Lag. Creating empty structure.")
            current_path = os.path.join(self.abs_data_path, self.snapshot, f'PHSUMM_{self.snapshot}.csv')
            loan_diff = pd.read_csv(current_path, usecols=[0], names=['cifno'], header=0)
            util.to_numeric(loan_diff, np.int64, 'cifno') # Cast to int64
            for col in expected_flags + ['loan_recent_inc_count', 'loan_to_debit', 'loanall_t']:
                loan_diff[col] = 0.0 # Force float type
        else:
            loan_diff = reduce(lambda x, y: x.merge(y, on='cifno', how='left'), datasets).fillna(0)
            
            # Calculate flags where history exists
            for i in range(1, 4):
                curr, prev = f't-{i-1}', f't-{i}'
                flag_col = expected_flags[i-1]
                if f'loanall_{curr}' in loan_diff.columns and f'loanall_{prev}' in loan_diff.columns:
                    # increase occurred if prev < curr
                    loan_diff[flag_col] = (loan_diff[f'loanall_{prev}'] < loan_diff[f'loanall_{curr}']).astype(int)
                else:
                    loan_diff[flag_col] = 0
            
            loan_diff['loan_recent_inc_count'] = loan_diff[expected_flags].sum(axis=1)
            
            # Loan to Debit estimate (0.08 of current loan)
            loan_diff['loan_to_debit'] = 0.0
            if 'loanall_t' in loan_diff.columns:
                loan_diff.loc[loan_diff['loanall_t'] != 0, 'loan_to_debit'] = loan_diff['loanall_t'] * 0.08
        # --- COLUMN INSURANCE END ---

        output_path = os.path.join(self.abs_out_path, f'loan_diff_{self.snapshot}_feat.csv')
        self.safe_makedirs(os.path.dirname(output_path))
        loan_diff.to_csv(output_path, index=False)
        print(f"Saved Loan Lag features to {output_path}")

    def create_saving_account(self):
        """Calculates Undebit-to-Total Saving ratios."""
        input_path = os.path.join(self.abs_data_path, self.snapshot, f'PHSUMM_{self.snapshot}.csv')
        output_path = os.path.join(self.abs_out_path, f'Saving_Acct_{self.snapshot}_feat.csv')
        
        if not os.path.exists(input_path): return

        dataset = pd.read_csv(input_path, sep=self.sep)
        dataset.columns = dataset.columns.str.lower().str.strip()
        util.to_numeric(dataset, np.int64, 'cifno')

        s_cols = [f't{i}' for i in range(1, 10) if f't{i}' in dataset.columns]
        u_cols = [f't{i}' for i in [3, 4, 9] if f't{i}' in dataset.columns]
        
        dataset['tot_saving'] = dataset[s_cols].sum(axis=1)
        dataset['tot_undebit_saving'] = dataset[u_cols].sum(axis=1)
        # Ratio logic
        dataset['ratio_undebit_saving'] = (dataset['tot_undebit_saving'] / dataset['tot_saving'].replace(0, np.nan)).fillna(1)

        self.safe_makedirs(os.path.dirname(output_path))
        dataset.to_csv(output_path, index=False)
        print(f"Saved Saving Account features.")