# amfs_tm/src/lib/segmentation/target_flag.py
import os
import numpy as np
import pandas as pd
from lib.segmentation.apply_flag import ApplyFlag

class TargetFlag(ApplyFlag):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        """
        data_path: Dictionary containing 'cif_acct' and 'target_path'
        """
        # Call parent with specific cif_acct subdirectory
        super().__init__(root, data_path['cif_acct'], out_path, sep, snapshot)

        self.target_path = data_path['target_path']
        
        # Target month logic (assuming self.target_month is defined in parent or util)
        # Using self.snapshot as a fallback if target_month isn't initialized
        t_month = getattr(self, 'target_month', self.snapshot)
        self.target_contacted = f'contacted_{t_month}.csv'
        self.target_converted = f'converted_{t_month}.csv'
        self.target_resp = f'resp_{t_month}.csv'

    def create(self):
        """Processes each target file and merges with baseline flags."""
        cif_acc_df = self.read_cif2acct(self.snapshot)
        axa_cif = self.get_axa_cif(cif_acc_df)
        trx_cif = self.get_trx_cif()
        cifno = 'cifno'

        for target_fname in [self.target_contacted, self.target_converted, self.target_resp]:
            in_target_path = os.path.join(self.target_path, target_fname)
            out_target_path = os.path.join(self.out_path, target_fname)

            if not os.path.exists(in_target_path):
                print(f"Skipping missing target file: {in_target_path}")
                continue

            print(f'Processing Target: {in_target_path}')
            target_df = pd.read_csv(in_target_path, sep=self.sep)
            target_df[cifno] = target_df[cifno].astype(np.int64)

            # Deduplicate target CIFs
            target_cif = target_df[[cifno]].drop_duplicates().reset_index(drop=True)

            # Merge with insurance active flag
            target_cif = target_cif.merge(axa_cif, on=cifno, how='left')
            target_cif['axa_pol_flag'] = target_cif['axa_pol_flag'].fillna(0).astype(np.int8)

            # Merge with bank active flag
            target_cif = target_cif.merge(trx_cif, on=cifno, how='left')
            target_cif['bank_active_flag'] = target_cif['bank_active_flag'].fillna(0).astype(np.int8)

            # Create final target segment
            target_seg = target_df.merge(target_cif, on=cifno, how='left')
            
            os.makedirs(os.path.dirname(out_target_path), exist_ok=True)
            target_seg.to_csv(out_target_path, sep=self.sep, index=False)