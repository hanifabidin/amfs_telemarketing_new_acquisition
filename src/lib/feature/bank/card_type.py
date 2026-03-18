# amfs_tm/src/lib/feature/bank/card_type.py
import os
import numpy as np
import pandas as pd
from lib import obj

class Type(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)
        self.nb_limit = {}
        self.type = None

    def create(self):
        # Read from 02_processed
        input_path = os.path.join(self.abs_data_path, self.snapshot, f'{self.type}_type_cnt_{self.snapshot}.csv')
        # Write to 03_features
        output_path = os.path.join(self.abs_out_path, f'{self.type}_type_cnt_{self.snapshot}_feat.csv')
        
        if not os.path.exists(input_path):
            print(f"Skipping {self.type}: Input file not found at {input_path}")
            return

        df = pd.read_csv(input_path, sep=self.sep)
        df.columns = df.columns.str.lower().str.strip()

        nb_cols = [c for c in self.nb_limit.keys() if c.startswith('nb_') and c in df.columns]
        total_col = f'total_nb_{self.type}'
        df[total_col] = df[nb_cols].sum(axis=1) if nb_cols else 0

        for col, limit in self.nb_limit.items():
            if col in df.columns:
                df[col] = df[col].fillna(0).clip(upper=limit).astype(np.int8)

        # MANDATORY: Use self.safe_makedirs
        self.safe_makedirs(os.path.dirname(output_path))
        df.to_csv(output_path, index=False)
        print(f"Successfully saved {self.type} card features.")

class CreditType(Type):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)
        self.nb_limit = {'nb_alop': 1, 'nb_classic': 3, 'nb_gold': 3, 'nb_platinum': 2, 'nb_signature': 1, 'total_nb_cc': 4}
        self.type = 'cc'

class DebitType(Type):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)
        self.nb_limit = {'nb_prioritas': 2, 'nb_silver': 5, 'nb_gold': 4, 'nb_platinum': 4, 'nb_others': 2, 'total_nb_dc': 7}
        self.type = 'dc'