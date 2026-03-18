# amfs_tm/src/lib/feature/bank/trx/trx_per.py
import os
import pandas as pd
from lib import obj

class TrxPer(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)

        self.trxper_cd_path = os.path.join(self.root, self.data_path, snapshot, 'axa_tran_allper_cd_{0}.csv')
        self.trxper_db_path = os.path.join(self.root, self.data_path, snapshot, 'axa_tran_allper_db_{0}.csv')
        self.feature_path = os.path.join(self.root, self.out_path, 'trxper_{0}_feat.csv')

    def create(self):
        print('start ' + self.snapshot)
        trxper_cd = pd.read_csv(self.trxper_cd_path.format(self.snapshot), sep=self.sep)
        trxper_db = pd.read_csv(self.trxper_db_path.format(self.snapshot), sep=self.sep)

        trxper = trxper_cd.merge(trxper_db, left_on='cifno_15', right_on='cifno_01', how='outer')
        trxper.loc[trxper['cifno_01'].isnull(), 'cifno_01'] = trxper['cifno_15']
        trxper['cifno'] = trxper['cifno_01']

        trxper['cifno'] = pd.to_numeric(trxper['cifno'], errors='coerce')
        trxper = trxper.dropna(subset=['cifno']).reset_index(drop=True)
        trxper = trxper.fillna(0)

        per_cols = ['q0', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90', 'q100']
        cd_cols = [f'sum_cd_{q}' for q in per_cols]
        db_cols = [f'sum_db_{q}' for q in per_cols]

        trxper['sum_cd'] = trxper[cd_cols].sum(axis=1)
        trxper['sum_db'] = trxper[db_cols].sum(axis=1)

        for col in cd_cols:
            trxper[col + '_per'] = 0
            trxper.loc[trxper['sum_cd'] > 0, col + '_per'] = trxper[col] / trxper['sum_cd']

        for col in db_cols:
            trxper[col + '_per'] = 0
            trxper.loc[trxper['sum_db'] > 0, col + '_per'] = trxper[col] / trxper['sum_db']

        trxper = trxper.drop(['cifno_15', 'cifno_01'], axis=1)
        
        output_file = self.feature_path.format(self.snapshot)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        trxper.to_csv(output_file, index=False)
        print('finish ' + self.snapshot)