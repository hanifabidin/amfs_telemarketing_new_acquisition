# amfs_tm/src/lib/feature/bank/trx/trx_lag.py
import abc
from lib import obj, util

class TrxLag(obj.Feature):
    __metaclass__ = abc.ABCMeta

    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)

        self.target_month = util.next_month(self.snapshot, out_format='%Y%m')
        self.timewindow = util.month_range(self.target_month, period=6, out_format="%Y%m")
        self.yyyymm_dict = self._make_yyyymm_dict()

    def _make_yyyymm_dict(self):
        yyyymm_dict = {}
        # Python 3: range replaces xrange
        for i in range(len(self.timewindow)):
            posfix = 'lm' if i == 0 else 'lm{}'.format(i + 1)
            yyyymm_dict[posfix] = self.timewindow[i]
        return yyyymm_dict

    def _diff_between(self, df, newcol_pref, usecol_pref, start, end):
        lm_col = '{0}_{1}'.format(usecol_pref, self.yyyymm_dict[start])
        lm3_col = '{0}_{1}'.format(usecol_pref, self.yyyymm_dict[end])
        new_col = '{0}_{1}_{2}_diff'.format(newcol_pref, start, end)
        df[new_col] = 0
        df.loc[df[lm3_col] > 0, new_col] = 1.0 * (df[lm_col] - df[lm3_col]) / df[lm3_col]
        df.loc[(df[lm3_col] == 0) & (df[lm_col] != 0), new_col] = df[lm_col]

    def _basic_stats(self, df, x='db', sum_prefix=True):
        if sum_prefix:
            lm6_sum_cols = list(map(lambda m: 'sum_{0}_{1}'.format(x, m), self.timewindow))
        else:
            lm6_sum_cols = list(map(lambda m: '{0}_{1}'.format(x, m), self.timewindow))
        
        lm3_sum_cols = lm6_sum_cols[:3]
        
        df['max_{}_lm3'.format(x)] = df[lm3_sum_cols].max(axis=1)
        df['min_{}_lm3'.format(x)] = df[lm3_sum_cols].min(axis=1)
        df['mean_{}_lm3'.format(x)] = df[lm3_sum_cols].mean(axis=1)

        df['max_{}_lm6'.format(x)] = df[lm6_sum_cols].max(axis=1)
        df['min_{}_lm6'.format(x)] = df[lm6_sum_cols].min(axis=1)
        df['mean_{}_lm6'.format(x)] = df[lm6_sum_cols].mean(axis=1)

    @abc.abstractmethod
    def create_lag(self):
        return