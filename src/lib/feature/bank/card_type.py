# amfs_tm/src/lib/feature/bank/card_type.py
import abc
import os
import numpy as np
import pandas as pd
from lib import obj

class Type(obj.Feature):
    __metaclass__ = abc.ABCMeta

    def __init__(self, root, data_path, out_path, sep, snapshot):
        # root is the base Volume path, e.g., /Volumes/catalog/schema/vol/amfs_tm
        obj.Feature.__init__(self, root, data_path, out_path, sep, snapshot)

        self.snapshot = snapshot
        # Updated to use os.path.join for Databricks POSIX paths
        self.cardtype_path = os.path.join(self.root, data_path, snapshot, '{type}_type_cnt_{snapshot}.csv')
        self.feature_path = os.path.join(self.root, out_path, '{type}_type_cnt_{snapshot}_feat.csv')
        self.nb_limit = {}
        self.type = None

    def create(self):
        input_path = self.cardtype_path.format(type=self.type, snapshot=self.snapshot)
        print('reading file {}'.format(input_path))
        
        # Databricks FUSE allows standard pandas read_csv on Volume paths
        type_cnt = pd.read_csv(input_path, sep=self.sep)

        # Restoring original filter logic
        nb_cols = list(filter(lambda x: x.startswith('nb_'), self.nb_limit.keys()))
        total_col = 'total_nb_' + self.type
        type_cnt[total_col] = type_cnt[nb_cols].sum(axis=1)

        for col, limit in self.nb_limit.items():
            if col in type_cnt.columns:
                type_cnt[col] = type_cnt[col].apply(lambda x: limit if x >= limit else x)
                type_cnt[col] = type_cnt[col].astype(np.int8)

        type_cnt = type_cnt.drop_duplicates().reset_index(drop=True)
        print('number of lines {}'.format(type_cnt.shape[0]))
        print('number of cif {}'.format(type_cnt.cifno.nunique()))

        output_path = self.feature_path.format(type=self.type, snapshot=self.snapshot)
        print('writing {}_type_cnt_feat file {}'.format(self.type, output_path))
        
        # Ensure directory exists in Volume before writing
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        type_cnt.to_csv(output_path, index=False, sep=',')
        print('writing done')

class CreditType(Type):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        Type.__init__(self, root, data_path, out_path, sep, snapshot)
        self.nb_limit = {
            'nb_alop': 1, 'nb_classic': 3, 'nb_gold': 3,
            'nb_platinum': 2, 'nb_signature': 1, 'total_nb_cc': 4
        }
        self.type = 'cc'

class DebitType(Type):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        Type.__init__(self, root, data_path, out_path, sep, snapshot)
        self.nb_limit = {
            'nb_prioritas': 2, 'nb_silver': 5, 'nb_gold': 4,
            'nb_platinum': 4, 'nb_others': 2, 'total_nb_dc': 7
        }
        self.type = 'dc'