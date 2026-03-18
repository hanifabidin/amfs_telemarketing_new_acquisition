# amfs_tm/src/lib/feature/bank/trx/max_trxdate.py
import os
import numpy as np
import pandas as pd
from lib import obj, util

class MaxTrDate(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)
        self.trx_cd_path = os.path.join(self.root, self.data_path, snapshot, 'axa_cust_cd_{0}.csv')
        self.feature_path = os.path.join(self.root, self.out_path, 'maxtrxdate_cddb_{0}_feat.csv')

    def create(self):
        trx_cd = pd.read_csv(self.trx_cd_path.format(self.snapshot), sep=self.sep)
        trx_cd = trx_cd.rename(columns={'cifno_15': 'cifno', 'cd_day': 'day'})
        util.to_numeric(trx_cd, np.int64, 'cifno', 'day')

        start_snapshot = util.to_format(self.snapshot, out_format='%Y-%m-%d')
        end_snapshot = util.last_day(self.snapshot, out_format='%Y-%m-%d')

        date_table = pd.DataFrame(pd.date_range(start=start_snapshot, end=end_snapshot, freq='D'), columns=['date'])
        date_table['day'] = date_table['date'].dt.day
        date_table['weekday'] = date_table['date'].dt.weekday
        date_table['day_1_10'] = (date_table['day'] <= 10).astype(int)
        date_table['day_11_20'] = ((date_table['day'] > 10) & (date_table['day'] <= 20)).astype(int)
        date_table['day_21_up'] = (date_table['day'] > 20).astype(int)

        weekdays = pd.get_dummies(date_table['weekday']).add_prefix('weekday_')
        date_table = pd.concat([date_table, weekdays], axis=1)

        def __process_trx(trx_df, pref='cd'):
            agg_dict = {
                f'max_{pref}_amt_new': 'max',
                'day': 'count',
                f'num_{pref}': 'sum',
                'day_1_10': 'sum', 'day_11_20': 'sum', 'day_21_up': 'sum'
            }
            # Add weekday sums
            for i in range(7):
                agg_dict[f'weekday_{i}'] = 'sum'

            trx_df = trx_df.merge(date_table, on='day', how='left')
            trx_gp = trx_df.groupby(['cifno'], as_index=False).aggregate(agg_dict).add_prefix(f'max{pref}_')
            
            # Re-mapping to your original naming convention
            trx_gp = trx_gp.rename(columns={
                f'max{pref}_cifno': 'cifno',
                f'max{pref}_max_{pref}_amt_new': f'max{pref}_amt',
                f'max{pref}_day': f'max{pref}_num_day',
                f'max{pref}_num_{pref}': f'max{pref}_num'
            })
            return trx_gp

        trxmax_final = __process_trx(trx_cd, pref='cd')
        trxmax_final['cifno'] = trxmax_final['cifno'].astype(int)

        output_file = self.feature_path.format(self.snapshot)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        trxmax_final.to_csv(output_file, index=False)