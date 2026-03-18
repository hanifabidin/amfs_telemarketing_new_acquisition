# amfs_tm/src/lib/feature/bank/trx/trx_outflow.py
import os
import numpy as np
import pandas as pd
from functools import reduce
from lib.feature.bank.trx.trx_lag import TrxLag, util

class TrxOutflow(TrxLag):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        TrxLag.__init__(self, root,  data_path, out_path, sep, snapshot)

        self.trxout_path = os.path.join(self.root, self.data_path, snapshot, 'trx_outflow_{0}.csv')
        self.feature_path = os.path.join(self.root, self.out_path, 'trxout_{0}_feat.csv')
        self.feature_lag_path = os.path.join(self.root, self.out_path, 'trxout_{0}_lag_feat.csv')

    def create(self):
        input_path = self.trxout_path.format(self.snapshot)
        trxout = pd.read_csv(input_path, sep=self.sep)
        trxout = trxout.fillna(0)

        print('length before cifno cleaning: ' + str(len(trxout)))
        util.to_numeric(trxout, np.int64, 'cifno_pengirim')
        print('length after cifno cleaning: ' + str(len(trxout)))

        # Group by cifno_pengirim
        trxout = trxout.drop(['rekening_pengirim'], axis=1, errors='ignore').groupby(['cifno_pengirim'], as_index=False).sum()

        # Percentage features
        trx_amt_cols = ['withdrawal', 'transfer_to_mandiri', 'transfer_to_others', 'bill_payment', 'trx_others']
        for col in trx_amt_cols:
            if col in trxout.columns:
                new_col = col + '_per'
                trxout[new_col] = 0
                trxout.loc[trxout['trx_total'] > 0, new_col] = trxout[col].astype(float) / trxout['trx_total']

        trx_chnl_amt_cols = ['ib_amt', 'mb_amt', 'branch_amt', 'nm_atm_amt', 'm_atm_amt', 'mcm_amt', 'chnl_others_amt']
        # Filter existing columns before sum
        existing_chnl_cols = [c for c in trx_chnl_amt_cols if c in trxout.columns]
        trxout['chnl_amt_sum'] = trxout[existing_chnl_cols].sum(axis=1)

        for col in existing_chnl_cols:
            new_col = col + '_per'
            trxout[new_col] = 0
            trxout.loc[trxout['chnl_amt_sum'] > 0, new_col] = trxout[col].astype(float) / trxout['chnl_amt_sum']

        trxout = trxout.drop(['chnl_amt_sum'], axis=1)

        os.makedirs(os.path.dirname(self.feature_path.format(self.snapshot)), exist_ok=True)
        trxout.to_csv(self.feature_path.format(self.snapshot), index=False)
        print('finish ' + self.snapshot)

    def create_lag(self):
        def _rename(x):
            return {'trx_total': 'trxout_' + x}

        usecols = ['cifno_pengirim', 'trx_total']
        trx_6m = [pd.read_csv(self.feature_path.format(month))[usecols].rename(columns=_rename(month))
                  for month in self.timewindow]
        
        df = reduce(lambda x, y: x.merge(y, on='cifno_pengirim', how='left'), trx_6m)
        df = df.fillna(0)

        self._diff_between(df, newcol_pref='trxout', usecol_pref='trxout', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='trxout', usecol_pref='trxout', start='lm3', end='lm6')
        self._basic_stats(df, x='trxout', sum_prefix=False)

        os.makedirs(os.path.dirname(self.feature_lag_path.format(self.snapshot)), exist_ok=True)
        df.to_csv(self.feature_lag_path.format(self.snapshot), index=False)