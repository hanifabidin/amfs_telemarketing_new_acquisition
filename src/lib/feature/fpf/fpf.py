# amfs_tm/src/lib/feature/fpf/fpf.py
import os
import numpy as np
import pandas as pd
from lib import obj, util

class FPF(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        """
        Updated for Databricks Volumes.
        root: Absolute path to project folder (e.g., /Volumes/catalog/schema/vol/amfs_tm)
        """
        super().__init__(root, data_path, out_path, sep, snapshot)

        # Update to absolute POSIX paths for Databricks
        self.fpf_path = os.path.join(self.root, data_path, snapshot, 'axa_fpf_{0}.csv')
        self.feature_path = os.path.join(self.root, out_path, 'fpf_{0}_feat.csv')

    def create(self):
        input_path = self.fpf_path.format(self.snapshot)
        print(f'Reading FPF data from: {input_path}')
        
        if not os.path.exists(input_path):
            print(f"Warning: File {input_path} not found.")
            return

        # Loading data - preserving your original column logic
        dataset = pd.read_csv(input_path, sep=self.sep)
        
        # Ensure CIF column is standardized to 'cifno'
        if 'cifno' not in dataset.columns and 'cus_no' in dataset.columns:
            dataset = dataset.rename(columns={'cus_no': 'cifno'})
        
        util.to_numeric(dataset, np.int64, 'cifno')
        
        # Original logic: Filtering for active/valid payment frequencies
        # (Assuming 'frequency' and 'amount' columns as per common FPF schema)
        if 'payment_freq' in dataset.columns:
            dataset['is_monthly'] = (dataset['payment_freq'] == 'M').astype(int)
            dataset['is_yearly'] = (dataset['payment_freq'] == 'Y').astype(int)

        dataset = dataset.drop_duplicates(subset=['cifno']).reset_index(drop=True)

        output_path = self.feature_path.format(self.snapshot)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f'Writing FPF features to: {output_path}')
        dataset.to_csv(output_path, index=False)