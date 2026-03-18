# amfs_tm/src/lib/feature/bank/trx/trx_debitedc.py
import os
import pandas as pd
from functools import reduce
from lib.feature.bank.trx.trx_lag import TrxLag

class TrxDebitEdc(TrxLag):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)

        # Updated to absolute Volume paths
        self.trxdebit_path = os.path.join(self.root, self.data_path, snapshot, 'debitcard_trx_edc_{0}.txt')
        self.feature_path = os.path.join(self.root, self.out_path, 'debitcard_trx_edc_{0}_feat.csv')
        self.feature_lag_path = os.path.join(self.root, self.out_path, 'debitcard_trx_edc_{0}_lag_feat.csv')

    def create(self):
        print('start ' + self.snapshot)
        input_file = self.trxdebit_path.format(self.snapshot)
        db_trx = pd.read_csv(input_file, sep=self.sep)
        db_trx = db_trx.fillna(0)

        group_maps = {
            'luxury': ['jewelry', 'antique'],
            'health': ['hospital', 'medical'],
            'vehicle': ['spbu', 'vehiclepart', 'vehicle'],
            'household': ['buildingmaterial', 'household', 'furniture'],
            'daily_life': ['fastfood', 'electrical', 'telco', 'utilities', 'transportation', 'othersprod', 'miscprod',
                           'hardware', 'directmark', 'mini_market', 'supermarket', 'deptstore'],
            'services': ['finservices', 'specialorg', 'profservices', 'eduserv', 'othersserv', 'miscserv', 'govserv'],
            'leisure_hobby': ['fashion', 'restaurant', 'bookstores', 'hobbies', 'electronic'],
            'offus': ['offus'],
            'travel': ['hotel', 'airlines']
        }

        for key in group_maps:
            new_col_freq = key + '_freq'
            new_col_sv = key + '_sv'
            db_trx[new_col_freq] = db_trx[[col + '_freq' for col in group_maps[key]]].sum(axis=1)
            db_trx[new_col_sv] = db_trx[[col + '_sv' for col in group_maps[key]]].sum(axis=1)

        col_use = ['cifno', 'trx_month'] + [col + '_freq' for col in group_maps.keys()] + \
                  [col + '_sv' for col in group_maps.keys()]
        db_trx_clean = db_trx[col_use].copy()

        db_trx_clean['sum_freq'] = db_trx_clean[[col + '_freq' for col in group_maps.keys()]].sum(axis=1)
        db_trx_clean['sum_sv'] = db_trx_clean[[col + '_sv' for col in group_maps.keys()]].sum(axis=1)

        for key in group_maps:
            new_col = key + '_sv_per'
            db_trx_clean[new_col] = 0
            db_trx_clean.loc[db_trx_clean['sum_sv'] > 0, new_col] = db_trx_clean[key + '_sv'] / db_trx_clean['sum_sv']

        output_file = self.feature_path.format(self.snapshot)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        db_trx_clean.to_csv(output_file, index=False)
        print('finish ' + self.snapshot)

    def create_lag(self):
        print('target month: ' + self.target_month + ', current month: ' + self.snapshot)
        usecols = ['cifno', 'sum_sv', 'sum_freq']

        def _rename(x):
            return {'sum_sv': 'sum_sv_' + x, 'sum_freq': 'sum_freq_' + x}

        trx_6m = [pd.read_csv(self.feature_path.format(month))[usecols].rename(columns=_rename(month))
                  for month in self.timewindow]

        df = reduce(lambda x, y: x.merge(y, on='cifno', how='left'), trx_6m)
        df = df.fillna(0)

        self._diff_between(df, newcol_pref='sv', usecol_pref='sum_sv', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='sv', usecol_pref='sum_sv', start='lm3', end='lm6')
        self._diff_between(df, newcol_pref='freq', usecol_pref='sum_freq', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='freq', usecol_pref='sum_freq', start='lm3', end='lm6')

        self._basic_stats(df, x='sv')
        self._basic_stats(df, x='freq')

        output_lag = self.feature_lag_path.format(self.snapshot)
        os.makedirs(os.path.dirname(output_lag), exist_ok=True)
        df.to_csv(output_lag, index=False)
        print('finish lag ' + self.snapshot)