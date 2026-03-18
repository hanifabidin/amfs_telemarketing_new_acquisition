# amfs_tm/src/lib/feature/bank/avr.py
import os
import numpy as np
import pandas as pd
from lib import obj

class BankAvr(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        # Initializing via the updated base class
        obj.Feature.__init__(self, root, data_path, out_path, sep, snapshot)
        
        # Paths are constructed relative to the absolute Volume paths
        self.avr_path = os.path.join(self.data_path, snapshot, 'axa_avr_{0}.csv')
        self.feature_path = os.path.join(self.out_path, 'axa_avr_{0}_feat.csv')

    def create(self):
        input_file = self.avr_path.format(self.snapshot)
        print(f'Reading file {input_file}')
        
        dataset = pd.read_csv(input_file, sep=self.sep, usecols=['cifno', 'nb_accts', 'sum_end_bal'])
        dataset = dataset.rename(columns={
            'cifno': 'CIFNO',
            'nb_accts': 'nb_accts_sd',
            'sum_end_bal': 'sum_end_bal'
        })

        print('Replacing NA with median/0')
        # Fix: Python 3 compatibility for print statements
        dataset['nb_accts_sd'] = dataset['nb_accts_sd'].fillna(dataset['nb_accts_sd'].median())
        dataset['sum_end_bal'] = dataset['sum_end_bal'].fillna(0)

        agg_dict = {
            'sum_end_bal': 'sum',
            'nb_accts_sd': 'sum'
        }
        dataset = dataset.groupby(['CIFNO'], as_index=False).aggregate(agg_dict)

        # Cap accounts at 15
        dataset['nb_accts_sd'] = dataset['nb_accts_sd'].apply(lambda x: x if x <= 15 else 15).astype(np.int8)

        print(f"Final Dataframe size: {dataset.shape}")
        
        output_file = self.feature_path.format(self.snapshot)
        # Ensure the output directory exists in the Volume
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f'Writing feature to {output_file}')
        dataset.to_csv(output_file, index=False)