# amfs_tm/src/lib/feature/bank/trx/trx_outflow_purchase.py
import os
import numpy as np
import pandas as pd
from functools import reduce
from lib.feature.bank.trx.trx_lag import TrxLag, util

class TrxOutflowPayment(TrxLag):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)

        self.trxout_purchase_path = os.path.join(self.root, self.data_path, snapshot, 'trx_outflow_payment_purchase_{0}.csv')
        self.feature_path = os.path.join(self.root, self.out_path, 'trxout_purchase_{0}_feat.csv')
        self.feature_lag_path = os.path.join(self.root, self.out_path, 'trxout_purchase_{0}_lag_feat.csv')

    def create(self):
        print('start ' + self.snapshot)
        input_file = self.trxout_purchase_path.format(self.snapshot)
        trxout_pur = pd.read_csv(input_file, sep=self.sep)

        util.to_numeric(trxout_pur, np.int64, 'customer_no')
        unique_purchase_types = np.unique(trxout_pur['transaction_type'].dropna())
        sum_df = trxout_pur[['customer_no']].drop_duplicates()

        for pur in unique_purchase_types:
            df = trxout_pur[trxout_pur['transaction_type'] == pur][['customer_no', 'frek', 'nominal']]
            df = df.rename(columns={'frek': pur + '_freq', 'nominal': pur + '_amt'})
            sum_df = sum_df.merge(df, on='customer_no', how='left')

        sum_df = sum_df.fillna(0)
        sum_df['sum_purchase_freq'] = sum_df[[col for col in sum_df.columns if '_freq' in col]].sum(axis=1)
        sum_df['sum_purchase_amt'] = sum_df[[col for col in sum_df.columns if '_amt' in col]].sum(axis=1)

        for col in [col for col in sum_df.columns if '_amt' in col]:
            new_col = col + '_per'
            sum_df[new_col] = 0
            sum_df.loc[sum_df['sum_purchase_amt'] > 0, new_col] = sum_df[col] / sum_df['sum_purchase_amt']

        output_file = self.feature_path.format(self.snapshot)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sum_df.to_csv(output_file, index=False)
        print('finish ' + self.snapshot)

    def create_lag(self):
        def _rename(x):
            return {'sum_purchase_amt': 'purchase_amt_' + x}

        usecols = ['customer_no', 'sum_purchase_amt']
        trx_6m = [pd.read_csv(self.feature_path.format(month))[usecols].rename(columns=_rename(month))
                  for month in self.timewindow]
        
        df = reduce(lambda x, y: x.merge(y, on='customer_no', how='left'), trx_6m)
        df = df.fillna(0)

        self._diff_between(df, newcol_pref='purchase_amt', usecol_pref='purchase_amt', start='lm', end='lm3')
        self._diff_between(df, newcol_pref='purchase_amt', usecol_pref='purchase_amt', start='lm3', end='lm6')
        self._basic_stats(df, x='purchase_amt', sum_prefix=False)

        output_lag = self.feature_lag_path.format(self.snapshot)
        os.makedirs(os.path.dirname(output_lag), exist_ok=True)
        df.to_csv(output_lag, index=False)