# amfs_tm/src/lib/feature/bank/balance.py
import datetime
import os
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from functools import reduce
from lib import obj, util

class Balance(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)
        self.target_month = util.next_month(snapshot, out_format='%Y%m')
        self.snapshot_date = datetime.datetime.strptime(snapshot, '%Y%m')
        self.min_date = self.snapshot_date - relativedelta(days=1)
        self.max_date = self.snapshot_date + relativedelta(months=1)

    def create_clean(self):
        """Standardizes raw CIFSUMM and handles Databricks Volume paths."""
        input_file = os.path.join(self.abs_data_path, self.snapshot, f'cifsumm_{self.snapshot}.csv')
        output_file = os.path.join(self.abs_out_path, f'CIFSUMM_{self.snapshot}_cleaned.csv')

        if not os.path.exists(input_file):
            print(f"Error: Source file not found at {input_file}")
            return

        df = pd.read_csv(input_file, sep=self.sep)
        df.columns = df.columns.str.upper().str.strip()
        df['YYMM'] = self.snapshot_date.strftime('%Y-%m')
        
        for date_col in ['MXDT', 'MNDT']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
                # Outlier removal
                df.loc[(df[date_col] < self.min_date) | (df[date_col] > self.max_date), date_col] = np.nan
                df[f'weekday_{date_col}'] = df[date_col].dt.day_name()
                
                # Day range logic (Replaces Numba vectorize)
                days = df[date_col].dt.day.fillna(-1)
                df[f'{date_col}_range'] = np.select(
                    [days.between(1, 10), days.between(11, 20), days > 20],
                    [1, 2, 3], default=-1
                )

        if 'MXDT' in df.columns and 'MNDT' in df.columns:
            df['delta_days'] = (df['MXDT'] - df['MNDT']).dt.days
        if 'MX' in df.columns and 'MN' in df.columns:
            df['delta_max_min_bal'] = df['MX'] - df['MN']
        
        self.safe_makedirs(os.path.dirname(output_file))
        df.to_csv(output_file, index=False)
        print(f"Cleaned file saved: {output_file}")

    def create_raw(self):
        """Joins 6 months of data. Gracefully skips missing history."""
        base_usecols = [
            'CIFNO', 'AVG', 'SDEV', 'MED', 'MX', 'MN', 'QT1', 'QT3', 'D20', 'EOM',
            'NOD', 'delta_days', 'MXDT_range', 'MNDT_range', 'delta_max_min_bal',
            'weekday_MXDT', 'weekday_MNDT',
        ]

        def __process(index, month):
            path = os.path.join(self.abs_out_path, f'CIFSUMM_{month}_cleaned.csv')
            if not os.path.exists(path):
                print(f"History missing for {month}, skipping in join.")
                return None
            
            usecols = base_usecols if index < 1 else base_usecols[:-2]
            if index >= 3: usecols = ['CIFNO', 'AVG', 'MX', 'MN', 'EOM']

            # Check existing columns in file to avoid usecols errors
            cols_in_file = pd.read_csv(path, nrows=1).columns
            cifsumm = pd.read_csv(path, usecols=[c for c in usecols if c in cols_in_file], index_col='CIFNO')
            
            suffix = f'_{util.to_format(month)}'
            cifsumm = cifsumm.add_suffix(suffix)
            if index == 0:
                cifsumm = cifsumm.rename(columns={f'weekday_MXDT{suffix}': 'weekday_MXDT', 
                                                  f'weekday_MNDT{suffix}': 'weekday_MNDT'})
            return cifsumm

        timewindow = util.month_range(self.target_month, period=6, out_format='%Y%m')
        balance_db = [d for d in [__process(i, m) for i, m in enumerate(timewindow)] if d is not None]
        
        if not balance_db: return
        dataset = reduce(lambda x, y: x.join(y, how='left'), balance_db).reset_index()
        
        output_path = os.path.join(self.abs_out_path, f'bal_{util.to_format(self.snapshot)}_feat_raw.csv')
        self.safe_makedirs(os.path.dirname(output_path))
        dataset.to_csv(output_path, index=False)

    def create(self):
        """Main Feature Engineering with Column Insurance for Deltas."""
        snapshot_str = util.to_format(self.snapshot)
        input_path = os.path.join(self.abs_out_path, f'bal_{snapshot_str}_feat_raw.csv')
        if not os.path.exists(input_path): return

        dataset_raw = pd.read_csv(input_path)
        timewindow = util.month_range(self.target_month, period=6)

        # 1. Map columns that exist in the dataframe
        max_f = [f'MX_{m}' for m in timewindow if f'MX_{m}' in dataset_raw.columns]
        min_f = [f'MN_{m}' for m in timewindow if f'MN_{m}' in dataset_raw.columns]
        avg_f = [f'AVG_{m}' for m in timewindow if f'AVG_{m}' in dataset_raw.columns]
        eom_f = [f'EOM_{m}' for m in timewindow if f'EOM_{m}' in dataset_raw.columns]
        d20_3m = [f'D20_{m}' for m in timewindow[:3] if f'D20_{m}' in dataset_raw.columns]

        # 2. Basic Aggregations (Calculated even if history is partial)
        dataset_raw['max_bal_3mth'] = dataset_raw[max_f[:3]].max(axis=1) if max_f else 0
        dataset_raw['max_bal_6mth'] = dataset_raw[max_f].max(axis=1) if max_f else 0
        dataset_raw['min_bal_3mth'] = dataset_raw[min_f[:3]].min(axis=1) if min_f else 0
        dataset_raw['avg_bal_3mth'] = dataset_raw[avg_f[:3]].mean(axis=1) if avg_f else 0
        dataset_raw['d20_3mth'] = dataset_raw[d20_3m].mean(axis=1) if d20_3m else 0

        # 3. Vintage Logic
        if max_f:
            map_max = {f: i+1 for i, f in enumerate(max_f)}
            dataset_raw['max_vintage_6mth'] = dataset_raw[max_f].idxmax(axis=1).map(map_max)
        else:
            dataset_raw['max_vintage_6mth'] = 1

        # 4. --- COLUMN INSURANCE (Deltas & Ratios) ---
        # Ensures your matrix always has these columns for model consistency
        for i in range(3):
            col = f'delta_avg_bal_t{i+1}_t{i+2}'
            ratio_col = f'ratio_{col}'
            
            if len(avg_f) > i + 1:
                t1, t2 = avg_f[i], avg_f[i+1]
                dataset_raw[col] = dataset_raw[t1] - dataset_raw[t2]
                dataset_raw[ratio_col] = dataset_raw[col] / (dataset_raw[t2].replace(0, np.nan))
            else:
                dataset_raw[col] = np.nan
                dataset_raw[ratio_col] = np.nan

        dataset_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        output_path = os.path.join(self.abs_out_path, f'bal_{snapshot_str}_feat.csv')
        self.safe_makedirs(os.path.dirname(output_path))
        dataset_raw.to_csv(output_path, index=False)

    def create_real(self):
        """Final loan-deducted table. TYPE HARDENED to prevent subtraction errors."""
        snapshot_str = util.to_format(self.snapshot)
        bal_path = os.path.join(self.abs_out_path, f'bal_{snapshot_str}_feat.csv')
        loan_path = os.path.join(self.abs_out_path, f'loan_diff_{self.snapshot}_feat.csv')

        if not os.path.exists(bal_path) or not os.path.exists(loan_path):
            print("Required feature files missing for loan deduction.")
            return

        df_bal = pd.read_csv(bal_path)
        df_loan = pd.read_csv(loan_path, usecols=['cifno', 'loan_to_debit'])
        
        # 1. Standardize Case and Force Integer Join Key
        if 'CIFNO' in df_bal.columns: df_bal = df_bal.rename(columns={'CIFNO': 'cifno'})
        util.to_numeric(df_bal, np.int64, 'cifno')
        util.to_numeric(df_loan, np.int64, 'cifno')
        
        # 2. Join balance with loan data
        df = df_bal.merge(df_loan, on='cifno', how='left').fillna(0)
        
        # 3. TYPE HARDENING: Force loan_to_debit to numeric to avoid TypeError
        df['loan_to_debit'] = pd.to_numeric(df['loan_to_debit'], errors='coerce').fillna(0.0)
        
        # 4. Arithmetic Loop: Deduct loan from specified balance features
        # Includes checks to force each target column to numeric before math
        target_cols = [c for c in df.columns if any(x in c for x in ['AVG', 'MX', 'MN', 'EOM', 'max_bal', 'avg_bal'])]
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            df[f'{col}_loan'] = df[col] - df['loan_to_debit']

        output_path = os.path.join(self.abs_out_path, f'bal_loan_feat_{snapshot_str}.csv')
        self.safe_makedirs(os.path.dirname(output_path))
        df.to_csv(output_path, index=False)
        print(f"Final loan-adjusted matrix saved: {output_path}")