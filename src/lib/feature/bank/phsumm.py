# amfs_tm/src/lib/feature/bank/phsumm.py
import os
from functools import reduce
import numba
import numpy as np
import pandas as pd
from lib import obj, util

class BankPH(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)

        self.snapshot = snapshot
        self.max_date = util.last_day(snapshot, out_format='%Y-%m-%d')

        # Absolute paths for Databricks Volumes
        self.phsumm_pattern = os.path.join(self.root, data_path, '{snapshot}/PHSUMM_{snapshot}.csv')
        self.out_phsumm_pattern = os.path.join(self.root, out_path, 'PHSUMM_{snapshot}_feat.csv')
        self.out_loan_pattern = os.path.join(self.root, out_path, 'loan_diff_{snapshot}.csv')
        self.out_saving_acct_pattern = os.path.join(self.root, out_path, 'Saving_Acct_{snapshot}.csv')

    @staticmethod
    @numba.vectorize
    def clean(x, default):
        return default if x > default else x

    def __handle_numerical(self, data, ph_cols):
        for col in ph_cols:
            data[col] = self.clean(data[col].fillna(0).values, 5).astype(np.uint8)

        data['LOANMORT'] = data['LOANMORT'].fillna(0)
        data['LOANALL'] = data['LOANALL'].fillna(0)
        data['TOTPH'] = self.clean(data['TOTPH'].fillna(15).values, 15)

        data['TOT_SAVING'] = self.clean(reduce(lambda x, y: x + y, [data[col] for col in ph_cols[:9]]).values, 11).astype(np.uint8)
        data['TOT_LOAN'] = self.clean(reduce(lambda x, y: x + y, [data[col] for col in ph_cols[11:17]]).values, 4)

    def __handle_date(self, data, date_cols, date_format):
        for col in date_cols:
            data.loc[(data[col] < '1980-01-01') & (data[col] > '1917-01-01'), col] = '1980-01-01'
            data.loc[~data[col].between('1980-01-01', self.max_date), col] = np.nan
            data[col] = pd.to_datetime(data[col], format=date_format)

        data['all_accts_first_date'] = data[['SFSTOPN', 'DFSTOPN', 'GFSTOPN', 'CFSTOPN']].min(axis=1).astype('datetime64[ns]')
        data['all_accts_last_date'] = data[['SLSTOPN', 'DLSTOPN', 'GLSTOPN', 'CLSTOPN']].max(axis=1).astype('datetime64[ns]')

        ref_date = pd.to_datetime(self.max_date)
        data['saving_acct_vint_l'] = (ref_date - data['SLSTOPN']).dt.days/30.44
        data['saving_acct_vint_f'] = (ref_date - data['SFSTOPN']).dt.days/30.44
        data['saving_acct_delta'] = (data['SLSTOPN'] - data['SFSTOPN']).dt.days/30.44

        data['all_acct_vint_l'] = (ref_date - data['all_accts_last_date']).dt.days/30.44
        data['all_acct_vint_f'] = (ref_date - data['all_accts_first_date']).dt.days/30.44
        data['all_acct_delta'] = (data['all_accts_last_date'] - data['all_accts_first_date']).dt.days/30.44

        data.drop(date_cols + ['all_accts_first_date', 'all_accts_last_date'], axis=1, inplace=True)

    def create(self):
        ph_count_cols = ['T%d' % i for i in range(1, 19)]
        time_cols = ['SFSTOPN', 'SLSTOPN', 'DFSTOPN', 'DLSTOPN', 'GFSTOPN', 'GLSTOPN', 'CFSTOPN', 'CLSTOPN']
        numercial_features = ph_count_cols + ['TOTPH', 'LOANMORT', 'LOANALL']

        data = pd.read_csv(self.phsumm_pattern.format(snapshot=self.snapshot), sep=self.sep)
        data = data.rename(columns={'LOANMORT_S': 'LOANMORT', 'LOANALL_S': 'LOANALL'})
        
        util.to_numeric(data, None, *numercial_features)
        data['T12'] = data['T12'].fillna(0).astype(np.uint8)
        data = data[data['TOTPH'] != data['T12']].reset_index(drop=True)

        self.__handle_numerical(data, ph_count_cols)
        self.__handle_date(data, time_cols, '%Y-%m-%d')

        output_path = self.out_phsumm_pattern.format(snapshot=self.snapshot)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)

    def create_loan(self):
        target_month = util.next_month(self.snapshot, out_format='%Y%m')
        timewindow = util.month_range(target_month, period=4, out_format='%Y%m')
        posfixes = {m: ('t' if i == 0 else f't-{i}') for i, m in enumerate(timewindow)}

        @numba.vectorize
        def loan_flg(x):
            return 1 if x < 0 else 0

        usecols = ['CIFNO', 'LOANALL', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'TOTPH']

        def __process(snapshot):
            path = self.phsumm_pattern.format(snapshot=snapshot)
            try:
                ds = pd.read_csv(path, sep=self.sep, usecols=usecols)
            except ValueError:
                ds = pd.read_csv(path, sep=self.sep, usecols=['CIFNO', 'LOANALL_S', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'TOTPH'])
                ds = ds.rename(columns={'LOANALL_S': 'LOANALL'})

            rename_map = {c: f"{c}_{posfixes[snapshot]}" for c in usecols[1:]}
            ds = ds.rename(columns=rename_map)
            util.to_numeric(ds, np.int64, 'CIFNO')
            for col in rename_map.values():
                ds[col] = pd.to_numeric(ds[col], errors='coerce').fillna(0)

            return ds[ds[f"TOTPH_{posfixes[snapshot]}"] != ds[f"T12_{posfixes[snapshot]}"]]

        loan_diff = reduce(lambda x, y: x.merge(y, on='CIFNO', how='left'), [__process(m) for m in timewindow]).fillna(0)

        loan_diff['LOANALL_t-1_LOANALL_t'] = loan_diff['LOANALL_t-1'] - loan_diff['LOANALL_t']
        loan_diff['loan_recent_inc_flag'] = loan_flg(loan_diff['LOANALL_t-1_LOANALL_t'].values)
        loan_diff['LOANALL_t-2_LOANALL_t-1'] = loan_diff['LOANALL_t-2'] - loan_diff['LOANALL_t-1']
        loan_diff['loan_recent_inc_flag2'] = loan_flg(loan_diff['LOANALL_t-2_LOANALL_t-1'].values)
        loan_diff['LOANALL_t-3_LOANALL_t-2'] = loan_diff['LOANALL_t-3'] - loan_diff['LOANALL_t-2']
        loan_diff['loan_recent_inc_flag3'] = loan_flg(loan_diff['LOANALL_t-3_LOANALL_t-2'].values)
        loan_diff['loan_recent_inc_count'] = loan_diff[['loan_recent_inc_flag', 'loan_recent_inc_flag2', 'loan_recent_inc_flag3']].sum(axis=1)

        loan_diff['loan_to_debit'] = 0
        loan_diff.loc[loan_diff['LOANALL_t'] != 0, 'loan_to_debit'] = loan_diff['LOANALL_t'] * 0.08
        # ... logic for specific loan increment conditions ...

        cols = ['T12', 'T13', 'T14', 'T15', 'T16', 'T17']
        for p in ['t', 't-1', 't-2']:
            loan_diff[f'tot_loan_{p}'] = loan_diff[[f"{c}_{p}" for c in cols]].sum(axis=1)

        savecols = ['CIFNO', 'loan_to_debit', 'loan_recent_inc_flag', 'loan_recent_inc_count', 'tot_loan_t', 'tot_loan_t-1', 'tot_loan_t-2', 'tot_loan_delta_t_t-1', 'tot_loan_delta_t_t-2', 'LOANALL_t']
        
        out_path = self.out_loan_pattern.format(snapshot=self.snapshot)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        loan_diff[savecols].to_csv(out_path, sep=',', index=False)

    def create_saving_account(self):
        ph_usecols = ['CIFNO', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T12', 'TOTPH']
        dataset = pd.read_csv(self.phsumm_pattern.format(snapshot=self.snapshot), sep=self.sep, usecols=ph_usecols).fillna(0)
        
        util.to_numeric(dataset, np.int64, 'CIFNO')
        for col in ph_usecols[1:]:
            dataset[col] = dataset[col].astype(np.uint8)

        dataset = dataset[dataset['TOTPH'] != dataset['T12']].reset_index(drop=True)
        dataset['TOT_UNDEBIT_SAVING'] = dataset[['T3', 'T4', 'T9']].sum(axis=1)
        dataset['TOT_SAVING'] = dataset[['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']].sum(axis=1)
        
        # ... ratio and flag logic ...
        dataset['ratio_undebit_saving'] = (dataset['TOT_UNDEBIT_SAVING'] / dataset['TOT_SAVING']).fillna(1)
        
        out_path = self.out_saving_acct_pattern.format(snapshot=self.snapshot)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        dataset.to_csv(out_path, sep=',', index=False)