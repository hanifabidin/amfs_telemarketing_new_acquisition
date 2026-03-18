# amfs_tm/src/lib/feature/bank/trx/trx_lag.py
import abc
import os
import pandas as pd
import numpy as np
from lib import obj, util

class TrxLag(obj.Feature, abc.ABC):
    """
    Base class for Transaction Lag features.
    Updated for Databricks Volumes and Cold-Start (missing history) safety.
    """

    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)

        # Standardizing on a 6-month window as per original logic
        self.target_month = util.next_month(self.snapshot, out_format='%Y%m')
        self.timewindow = util.month_range(self.target_month, period=6, out_format="%Y%m")
        self.yyyymm_dict = self._make_yyyymm_dict()

    def _make_yyyymm_dict(self):
        """Maps posfixes (lm, lm2, etc.) to specific months in the window."""
        yyyymm_dict = {}
        for i in range(len(self.timewindow)):
            # i=0 is 'lm', i=1 is 'lm2', etc.
            posfix = 'lm' if i == 0 else 'lm{}'.format(i + 1)
            yyyymm_dict[posfix] = self.timewindow[i]
        return yyyymm_dict

    def get_historical_file(self, pattern, month):
        """Safely load a historical feature file from the 03_features Volume."""
        fpath = os.path.join(self.abs_out_path, pattern.format(month=month))
        if os.path.exists(fpath):
            return pd.read_csv(fpath)
        return None

    def _diff_between(self, df, newcol_pref, usecol_pref, start, end):
        """Calculates % difference between two months with Cold-Start safety."""
        lm_month = self.yyyymm_dict.get(start)
        lm3_month = self.yyyymm_dict.get(end)
        
        lm_col = f'{usecol_pref}_{lm_month}'
        lm3_col = f'{usecol_pref}_{lm3_month}'
        new_col = f'{newcol_pref}_{start}_{end}_diff'

        # COLUMN INSURANCE: If history doesn't exist, create column as NaN
        if lm_col not in df.columns or lm3_col not in df.columns:
            df[new_col] = np.nan
            return

        df[new_col] = 0.0
        # Calculate diff: (Current - Past) / Past
        mask_past_exists = df[lm3_col] > 0
        df.loc[mask_past_exists, new_col] = (df[lm_col] - df[lm3_col]) / df[lm3_col]
        
        # If past was 0 but current is not, growth is essentially the current value
        mask_new_growth = (df[lm3_col] == 0) & (df[lm_col] != 0)
        df.loc[mask_new_growth, new_col] = df[lm_col]

    def _basic_stats(self, df, x='db', sum_prefix=True):
        """Calculates min/max/mean for 3m and 6m windows."""
        # Determine prefix
        pref = 'sum_{}_'.format(x) if sum_prefix else '{}_'.format(x)
        
        # COLD START GUARD: Only include columns that actually exist in the dataframe
        all_possible_cols = [f'{pref}{m}' for m in self.timewindow]
        existing_cols = [c for c in all_possible_cols if c in df.columns]

        if not existing_cols:
            # If no history exists, create dummy null columns so model schema is consistent
            for stat in ['max', 'min', 'mean']:
                df[f'{stat}_{x}_lm3'] = np.nan
                df[f'{stat}_{x}_lm6'] = np.nan
            return

        # 3-Month Window (Subset of existing)
        lm3_cols = [c for c in existing_cols if any(m in c for m in self.timewindow[:3])]
        
        df['max_{}_lm3'.format(x)] = df[lm3_cols].max(axis=1) if lm3_cols else np.nan
        df['min_{}_lm3'.format(x)] = df[lm3_cols].min(axis=1) if lm3_cols else np.nan
        df['mean_{}_lm3'.format(x)] = df[lm3_cols].mean(axis=1) if lm3_cols else np.nan

        # 6-Month Window
        df['max_{}_lm6'.format(x)] = df[existing_cols].max(axis=1)
        df['min_{}_lm6'.format(x)] = df[existing_cols].min(axis=1)
        df['mean_{}_lm6'.format(x)] = df[existing_cols].mean(axis=1)

    @abc.abstractmethod
    def create_lag(self):
        pass