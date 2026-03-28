# amfs_tm/src/lib/feature/bank/avr.py
import os
import numpy as np
import pandas as pd
from lib import obj, util

class BankAvr(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        # Initializing via the base class to get absolute path variables
        super().__init__(root, data_path, out_path, sep, snapshot)

    def create(self):
        # 1. Use absolute Volume paths (self.abs_data_path / self.abs_out_path)
        input_file = os.path.join(self.abs_data_path, self.snapshot, f'axa_avr_{self.snapshot}.csv')
        output_file = os.path.join(self.abs_out_path, f'axa_avr_{self.snapshot}_feat.csv')
        
        print(f'Reading file: {input_file}')
        
        if not os.path.exists(input_file):
            print(f"CRITICAL ERROR: File truly not found at absolute path: {input_file}")
            return

        # 2. Load and immediately standardize CIFNO to int64
        dataset = pd.read_csv(input_file, sep=self.sep, usecols=['cifno', 'nb_accts', 'sum_end_bal'])
        dataset = dataset.rename(columns={
            'cifno': 'CIFNO',
            'nb_accts': 'nb_accts_sd'
        })

        # Force int64 to prevent future merge errors
        util.to_numeric(dataset, np.int64, 'CIFNO')

        print('Replacing NA with median/0')
        dataset['nb_accts_sd'] = dataset['nb_accts_sd'].fillna(dataset['nb_accts_sd'].median())
        dataset['sum_end_bal'] = dataset['sum_end_bal'].fillna(0)

        # 3. Aggregate
        agg_dict = {
            'sum_end_bal': 'sum',
            'nb_accts_sd': 'sum'
        }
        dataset = dataset.groupby(['CIFNO'], as_index=False).aggregate(agg_dict)

        # 4. Cap accounts at 15 (Using .clip for performance instead of apply)
        dataset['nb_accts_sd'] = dataset['nb_accts_sd'].clip(upper=15).astype(np.int8)

        print(f"Final Dataframe size: {dataset.shape}")
        
        # 5. Safe Save using the absolute output Volume path
        self.safe_makedirs(os.path.dirname(output_file))
        
        print(f'Writing feature to: {output_file}')
        dataset.to_csv(output_file, index=False)
        print(f'Finish BankAvr.create for {self.snapshot}')