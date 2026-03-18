# amfs_tm/src/lib/feature/bank/cust_demo.py
import os
import numba
import numpy as np
import pandas as pd
from lib import obj, util

class CustDemo(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        """
        Updated for Databricks:
        root: Absolute Volume path (e.g., /Volumes/catalog/schema/vol/amfs_tm)
        """
        super().__init__(root, data_path, out_path, sep, snapshot)

        # Construct absolute POSIX paths for Databricks Volumes
        self.custdemo_path = os.path.join(self.root, data_path, snapshot, 'axa_cust_demo_{0}.csv')
        self.feature_path = os.path.join(self.root, out_path, 'axa_cust_demo_clean_{0}.csv')

    def create(self):
        input_path = self.custdemo_path.format(self.snapshot)
        usecols = ['cuscls_id', 'cusedu_id', 'gender_id', 'houstyp_id',
                   'occp_id', 'marital_id', 'year_addr', 'year_rel',
                   'year_work', 'intuser_fg', 'mbluser_fg', 'prior_fg', 'rsdnt_fg',
                   'vol_income', 'empinds_id', 'date_org', 'depend_id', 'age2', 'age']
        
        # Check schema for CIF column naming variants
        schema = pd.read_csv(input_path, sep=self.sep, encoding='utf-8', nrows=1)
        if (schema.columns == 'cus_no').any():
            cif_col = 'cus_no'
        elif (schema.columns == 'cusno').any():
            cif_col = 'cusno'
        else:
            raise ValueError('Not found cus_no/cusno column in raw data!')
        
        usecols.append(cif_col)

        # Load and rename age tiers as per original requirement
        customer_raw_indivual = pd.read_csv(input_path, sep='|', encoding='utf-8', usecols=usecols)
        customer_raw_indivual = customer_raw_indivual.rename(columns={
            'age2': 'age_tier',
            'age': 'age_tier2',
            cif_col: 'cusno'
        })

        print(f"Unique CIFs: {customer_raw_indivual.cusno.nunique()}")
        print(f"Data shape: {customer_raw_indivual.shape}")

        # Map depend_id to numeric
        depend_id_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        customer_raw_indivual['depend_nb'] = customer_raw_indivual['depend_id'].map(depend_id_map, na_action='ignore')

        # Convert columns to numeric
        util.to_numeric(customer_raw_indivual, np.int64, 'cusno')
        util.to_numeric(customer_raw_indivual, None, *['year_addr', 'year_rel', 'year_work', 'vol_income'])

        # Categorical Validation (keep only allowed values, else NaN)
        valid_values = {
            'cusedu_id': ['A', 'B', 'C', 'D', 'E', 'F', 'Z'],
            'gender_id': ['M', 'F'],
            'houstyp_id': ['MILIK', 'KANTR', 'KLRGA', 'SEWA', 'DINAS', 'LAIN', 'KOST', 'PROJ'],
            'occp_id': ["PSW", "WSW", "MHS", "IRT", "PROF", "LAIN", "PNS", "GR", "PBUM", "PTN", "PIF", "POL", "MIL",
                        "PENS", "PRMDS", "DKTR", "YYS", "KOP", "ENGR", "PKS", "KONS", "AKT", "WRTWN", "PNG", "PKJSN",
                        "NTRS", "PGCR"],
            'marital_id': ['A', 'B', 'Z', 'C', 'D'],
            'empinds_id': ['10', '08', '09', '03', '01', '06', '06009', '05', '02', '07', '04', '06008', '06006',
                           '07001', '08001', '06007', '06003', '06002', '09001', '06001', '06004', '08002', '06005'],
            'intuser_fg': ['Y', 'N'], 'prior_fg': ['Y', 'N'], 'mbluser_fg': ['Y', 'N'], 'rsdnt_fg': ['Y', 'N']
        }

        for col, values in valid_values.items():
            customer_raw_indivual.loc[~customer_raw_indivual[col].isin(values), col] = np.nan

        # Capping values using Numba
        @numba.vectorize
        def transform(x, minimum):
            return x if x <= minimum else minimum

        customer_raw_indivual['year_addr'] = transform(customer_raw_indivual['year_addr'].values, 60).astype(int)
        customer_raw_indivual['year_work'] = transform(customer_raw_indivual['year_work'].values, 50).astype(int)
        customer_raw_indivual.loc[~customer_raw_indivual['year_rel'].between(0, 30), 'year_rel'] = np.nan

        customer_raw_indivual = customer_raw_indivual.drop(['depend_id'], axis=1)

        # Output to Volume
        output_path = self.feature_path.format(self.snapshot)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f'Writing cleaned demographics to {output_path}')
        customer_raw_indivual.to_csv(output_path, sep=',', index=False)