# amfs_tm/src/lib/feature/bank/balance.py
import datetime
import os
import numba
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from functools import reduce

from lib import obj, util

class BankBalance(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        """
        Updated for Databricks:
        root: Absolute path to the project (e.g., /Volumes/catalog/schema/volume/amfs_tm)
        data_path: Absolute path to processed data
        out_path: Absolute path for feature output
        """
        super().__init__(root, data_path, out_path, sep, snapshot)

        self.snapshot = snapshot
        self.target_month = util.next_month(snapshot, out_format='%Y%m')

        self.snapshot_date = datetime.datetime.strptime(snapshot, '%Y%m')
        self.min_date = self.snapshot_date - relativedelta(days=1)
        self.max_date = self.snapshot_date + relativedelta(months=1)

        # Use absolute paths provided by the config; standardizing to Volume/DBFS structure
        self.cifsumm_pattern = os.path.join(self.data_path, '{snapshot}/cifsumm_{snapshot}.csv')
        self.loandiff_pattern = os.path.join(self.out_path, 'loan_diff_{snapshot}.csv')
        self.clean_bal_pattern = os.path.join(self.data_path, '{snapshot}/CIFSUMM_{snapshot}_cleaned.csv')
        self.raw_balfeat_pattern = os.path.join(self.out_path, 'bal_{snapshot}_feat_raw.csv')

        self.out_bal_pattern = os.path.join(self.out_path, 'bal_{snapshot}_feat.csv')
        self.out_deduct_pattern = os.path.join(self.out_path, 'bal_deduct_{snapshot}.csv')
        self.out_real_pattern = os.path.join(self.out_path, 'bal_loan_feat_{snapshot}.csv')

    def create_clean(self):
        @numba.vectorize
        def __day_range_month(x):
            if 1 <= x <= 10:
                return 1
            elif 10 < x <= 20:
                return 2
            else:
                return 3

        input_file = self.cifsumm_pattern.format(snapshot=self.snapshot)
        df = pd.read_csv(input_file, sep=self.sep)
        df['YYMM'] = self.snapshot_date.strftime('%Y-%m')
        
        for date_col in ['MXDT', 'MNDT']:
            df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
            # clear outlier
            df.loc[(df[date_col] < self.min_date) | (df[date_col] > self.max_date), date_col] = np.nan

            # add more columns
            df['weekday_%s' % date_col] = df[date_col].dt.day_name()
            df['%s_range' % date_col] = df[date_col].dt.day
            df['%s_range' % date_col] = df['%s_range' % date_col].fillna(-1)
            df['%s_range' % date_col] = __day_range_month(df['%s_range' % date_col].values)

        df['delta_days'] = (df['MXDT'] - df['MNDT']).dt.days

        # diff between max and min
        df['delta_max_min_bal'] = pd.eval('df.MX - df.MN')
        
        output_file = self.clean_bal_pattern.format(snapshot=self.snapshot)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, sep=',')

    def create_raw(self):
        base_usecols = [
            'CIFNO', 'AVG', 'SDEV', 'MED', 'MX', 'MN', 'QT1', 'QT3', 'D20', 'EOM',
            'NOD', 'delta_days', 'MXDT_range', 'MNDT_range', 'delta_max_min_bal',
            'weekday_MXDT', 'weekday_MNDT',
        ]

        def __process(index, month):
            usecols = base_usecols
            if index >= 1:
                usecols = base_usecols[:-2]
            if index >= 3:
                usecols = ['CIFNO', 'AVG', 'MX', 'MN', 'EOM']

            path = self.clean_bal_pattern.format(snapshot=month)
            cifsumm = pd.read_csv(path, sep=',', usecols=usecols, index_col='CIFNO')

            cifsumm = cifsumm.add_suffix('_{0}'.format(util.to_format(month)))
            if index == 0:
                cifsumm = cifsumm.rename(columns=dict([(c + '_{0}'.format(util.to_format(month)), c)
                                                       for c in ['weekday_MXDT', 'weekday_MNDT']]))
            return cifsumm

        timewindow = util.month_range(self.target_month, period=6, out_format='%Y%m')
        balance_db = [__process(i, m) for i, m in enumerate(timewindow)]

        dataset = reduce(lambda x, y: x.join(y, how='left'), balance_db)
        dataset = dataset.reset_index()

        output_path = self.raw_balfeat_pattern.format(snapshot=util.to_format(self.snapshot))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_csv(output_path, index=False)

    def create(self):
        snapshot = util.to_format(self.snapshot)
        timewindow = util.month_range(self.target_month, period=6)

        input_path = self.raw_balfeat_pattern.format(snapshot=snapshot)
        dataset_raw = pd.read_csv(input_path)

        max_features = ['MX_{0}'.format(month) for month in timewindow]
        min_features = ['MN_{0}'.format(month) for month in timewindow]
        avg_features = ['AVG_{0}'.format(month) for month in timewindow]
        eom_features = ['EOM_{0}'.format(month) for month in timewindow]

        nod_3mth = ['NOD_{0}'.format(month) for month in timewindow[:3]]
        delta_days_3mth = ['delta_days_{0}'.format(month) for month in timewindow[:3]]
        delta_bal_3mth = ['delta_max_min_bal_{0}'.format(month) for month in timewindow[:3]]
        d20_3mth = ['D20_{0}'.format(month) for month in timewindow[:3]]

        map_max_bal = dict(zip(max_features, range(1, len(max_features) + 1)))
        map_min_bal = dict(zip(min_features, range(1, len(min_features) + 1)))

        useless_cols = reduce(lambda x, y: x + y, [
            max_features[1:], min_features[1:], avg_features[1:], eom_features[1:],
            nod_3mth[1:], delta_days_3mth[1:], delta_bal_3mth[1:], d20_3mth[1:]
        ])

        # Feature engineering logic (Max/Min/Avg/Ratio)
        dataset_raw['max_bal_3mth'] = dataset_raw[max_features[:3]].max(axis=1)
        dataset_raw['max_bal_6mth'] = dataset_raw[max_features].max(axis=1)
        dataset_raw['max_vintage_3mth'] = dataset_raw[max_features[:3]].idxmax(axis=1).apply(lambda x: map_max_bal[x])
        dataset_raw['max_vintage_6mth'] = dataset_raw[max_features].idxmax(axis=1).apply(lambda x: map_max_bal[x])

        dataset_raw['min_bal_3mth'] = dataset_raw[min_features[:3]].min(axis=1)
        dataset_raw['min_bal_6mth'] = dataset_raw[min_features].min(axis=1)
        dataset_raw['min_vintage_3mth'] = dataset_raw[min_features[:3]].idxmin(axis=1).apply(lambda x: map_min_bal[x])
        dataset_raw['min_vintage_6mth'] = dataset_raw[min_features].idxmin(axis=1).apply(lambda x: map_min_bal[x])

        dataset_raw['delta_vintage_3mth'] = dataset_raw['max_vintage_3mth'] - dataset_raw['min_vintage_3mth']
        dataset_raw['delta_vintage_6mth'] = dataset_raw['max_vintage_6mth'] - dataset_raw['min_vintage_6mth']
        dataset_raw['max_delta_bal_3mth'] = dataset_raw['max_bal_3mth'] - dataset_raw['min_bal_3mth']
        dataset_raw['max_delta_bal_6mth'] = dataset_raw['max_bal_6mth'] - dataset_raw['min_bal_6mth']

        dataset_raw['avg_lm2'] = dataset_raw[avg_features[1]]
        dataset_raw['avg_lm3'] = dataset_raw[avg_features[2]]
        dataset_raw['avg_bal_3mth'] = dataset_raw[avg_features[:3]].mean(axis=1)
        dataset_raw['avg_bal_6mth'] = dataset_raw[avg_features].mean(axis=1)
        dataset_raw['eom_3mth'] = dataset_raw[eom_features[:3]].mean(axis=1)
        dataset_raw['eom_6mth'] = dataset_raw[eom_features].mean(axis=1)

        # Delta Avg Calculations
        for i in range(3):
            t1, t2 = avg_features[i], avg_features[i+1]
            col = f'delta_avg_bal_t{i+1}_t{i+2}'
            ratio_col = f'ratio_delta_avg_bal_t{i+1}_t{i+2}'
            dataset_raw[col] = dataset_raw[t1] - dataset_raw[t2]
            dataset_raw[col] = dataset_raw[col].fillna(dataset_raw[t1])
            dataset_raw[ratio_col] = dataset_raw[col] / (dataset_raw[t2].replace(0, np.nan))
            dataset_raw[ratio_col] = dataset_raw[ratio_col].fillna(dataset_raw[col])

        dataset_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset_raw['std_avg_bal_3mth'] = dataset_raw[avg_features[:3]].std(axis=1)
        dataset_raw['std_avg_bal_6mth'] = dataset_raw[avg_features].std(axis=1)
        dataset_raw['avg_nod_3mth'] = dataset_raw[nod_3mth].mean(axis=1)
        dataset_raw['avg_delta_days_3mth'] = dataset_raw[delta_days_3mth].mean(axis=1)
        dataset_raw['avg_delta_bal_3mth'] = dataset_raw[delta_bal_3mth].mean(axis=1)
        dataset_raw['one_max_delta_bal_3mth'] = dataset_raw[delta_bal_3mth].max(axis=1)
        dataset_raw['d20_3mth'] = dataset_raw[d20_3mth].mean(axis=1)

        dataset_raw.drop(useless_cols, axis=1, inplace=True)
        
        output_path = self.out_bal_pattern.format(snapshot=snapshot)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset_raw.to_csv(output_path, index=False)

    def create_deduct(self):
        bal_usecols = ['CIFNO', 'AVGD', 'MND', 'EOMD']
        bal_deduct = pd.read_csv(self.cifsumm_pattern.format(snapshot=self.snapshot), sep=self.sep, usecols=bal_usecols)

        loan_usecols = ['CIFNO', 'loan_to_debit']
        loan = pd.read_csv(self.loandiff_pattern.format(snapshot=self.snapshot), sep=',', usecols=loan_usecols)

        bal_deduct = bal_deduct.merge(loan, on='CIFNO', how='left')
        bal_deduct['loan_to_debit'] = bal_deduct['loan_to_debit'].fillna(0)

        for col in ['AVGD', 'MND', 'EOMD']:
            bal_deduct[f'{col}_loan'] = bal_deduct[col] - bal_deduct['loan_to_debit']

        bal_deduct.drop(['loan_to_debit'], axis=1, inplace=True)
        output_path = self.out_deduct_pattern.format(snapshot=self.snapshot)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        bal_deduct.to_csv(output_path, sep=',', index=False)

    def create_real(self):
        dataset_bal = pd.read_csv(self.out_bal_pattern.format(snapshot=util.to_format(self.snapshot)))
        loan_usecols = ['CIFNO', 'loan_to_debit']
        dataset_loan = pd.read_csv(self.loandiff_pattern.format(snapshot=self.snapshot), sep=',', usecols=loan_usecols)
        
        dataset_bal = dataset_bal.merge(dataset_loan, how='left', on='CIFNO')
        dataset_bal['loan_to_debit'] = dataset_bal['loan_to_debit'].fillna(0)

        snapshot_str = util.to_format(self.snapshot)
        current_balcols = [f'{col}_{snapshot_str}' for col in ['AVG', 'MED', 'MX', 'MN', 'QT1', 'QT3', 'D20', 'EOM', 'delta_max_min_bal']]

        lastmonth = util.last_month(self.snapshot)
        lastmonth_balcols = [f'{col}_{lastmonth}' for col in ['MED', 'QT1', 'QT3']]

        last2month = util.last_month(lastmonth, in_format='%y%m')
        last2month_balcols = [f'{col}_{last2month}' for col in ['MED', 'QT1', 'QT3']]

        other_balcols = [
            'max_bal_3mth', 'max_bal_6mth', 'min_bal_3mth', 'min_bal_6mth',
            'max_delta_bal_3mth', 'max_delta_bal_6mth', 'avg_lm2', 'avg_lm3',
            'avg_bal_3mth', 'avg_bal_6mth', 'eom_3mth', 'eom_6mth', 'd20_3mth',
            'delta_avg_bal_t1_t2', 'delta_avg_bal_t2_t3', 'delta_avg_bal_t3_t4'
        ]

        for col in current_balcols + lastmonth_balcols + last2month_balcols + other_balcols:
            dataset_bal[f'{col}_loan'] = dataset_bal[col] - dataset_bal['loan_to_debit']

        output_path = self.out_real_pattern.format(snapshot=snapshot_str)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset_bal.to_csv(output_path, sep=',', index=False)