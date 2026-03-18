# amfs_tm/src/lib/feature/bank/trx/trx_net.py
import os
import numpy as np
import pandas as pd
from functools import reduce # Added for Python 3 compatibility
from lib.feature.bank.trx.trx_lag import util, TrxLag

class TrxNet(TrxLag):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        TrxLag.__init__(self, root, data_path, out_path, sep, snapshot)

        # Standardizing to absolute Volume paths
        self.trxnet_path = os.path.join(self.root, self.data_path, snapshot, 'axa_tran_net_{0}.csv')
        self.feature_path = os.path.join(self.root, self.out_path, 'trxnet_{0}_feat.csv')
        self.feature_lag_path = os.path.join(self.root, self.out_path, 'trxnet_{0}_lag_feat.csv')

    def create(self):
        print('start ' + self.snapshot)
        input_path = self.trxnet_path.format(self.snapshot)
        trxnet = pd.read_csv(input_path, sep=self.sep)

        util.to_numeric(trxnet, None, 'sum_db', 'sum_cd', 'num_db', 'num_cd', 'num_db_0', 'num_cd_0')
        trxnet = trxnet.fillna(0)

        print('length before cifno cleaning: ' + str(len(trxnet)))
        util.to_numeric(trxnet, np.int64, 'cifno')
        print('length after cifno cleaning: ' + str(len(trxnet)))

        trxnet['sum_cd_db_diff'] = trxnet['sum_cd'] - trxnet['sum_db']
        trxnet['num_cd_db_diff'] = trxnet['num_cd'] - trxnet['num_db']
        
        # Safely drop columns if they exist
        drop_cols = [c for c in ['cifno_01', 'cifno_15'] if c in trxnet.columns]
        trxnet = trxnet.drop(drop_cols, axis=1)

        os.makedirs(os.path.dirname(self.feature_path.format(self.snapshot)), exist_ok=True)
        trxnet.to_csv(self.feature_path.format(self.snapshot), index=False)
        print('finish ' + self.snapshot)

    def create_lag(self):
        print('target month: ' + self.target_month + ', current month: ' + self.snapshot)

        def _rename(x):
            return {'sum_db': 'sum_db_' + x, 'sum_cd': 'sum_cd_' + x}

        usecols = ['cifno', 'sum_db', 'sum_cd']
        trx_6m = [pd.read_csv(self.feature_path.format(month))[usecols].rename(columns=_rename(month))
                  for month in self.timewindow]

        df = reduce(lambda x, y: x.merge(y, on='cifno', how='left'), trx_6m)
        df = df.fillna(0)

        self._diff_between(df, newcol_pref='db', usecol_pref='sum_db', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='db', usecol_pref='sum_db', start='lm3', end='lm6')
        self._diff_between(df, newcol_pref='cd', usecol_pref='sum_cd', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='cd', usecol_pref='sum_cd', start='lm3', end='lm6')

        self._basic_stats(df, x='db')
        self._basic_stats(df, x='cd')

        os.makedirs(os.path.dirname(self.feature_lag_path.format(self.snapshot)), exist_ok=True)
        df.to_csv(self.feature_lag_path.format(self.snapshot), index=False)