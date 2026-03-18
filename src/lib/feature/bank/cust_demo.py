# amfs_tm/src/lib/feature/bank/cust_demo.py
import os
import numpy as np
import pandas as pd
from lib import obj, util

class CustDemo(obj.Feature):
    # No __init__ needed! It uses the one from obj.Feature automatically.

    def create(self):
        # 1. Define Paths using absolute Volume paths from the base class
        input_path = os.path.join(self.abs_data_path, self.snapshot, f'axa_cust_demo_{self.snapshot}.csv')
        output_path = os.path.join(self.abs_out_path, f'axa_cust_demo_clean_{self.snapshot}.csv')

        if not os.path.exists(input_path):
            print(f"CustDemo input not found: {input_path}")
            return

        # 2. Define columns to keep
        usecols = [
            'cuscls_id', 'cusedu_id', 'gender_id', 'houstyp_id',
            'occp_id', 'marital_id', 'year_addr', 'year_rel',
            'year_work', 'intuser_fg', 'mbluser_fg', 'prior_fg', 'rsdnt_fg',
            'vol_income', 'empinds_id', 'date_org', 'depend_id', 'age2', 'age'
        ]

        # 3. Check for CIF column naming variants (cus_no vs cusno)
        # Using self.sep ('|') as we established earlier
        schema = pd.read_csv(input_path, sep=self.sep, encoding='utf-8', nrows=1)
        if 'cus_no' in schema.columns:
            cif_col = 'cus_no'
        elif 'cusno' in schema.columns:
            cif_col = 'cusno'
        else:
            raise ValueError('Could not find cus_no or cusno column in raw data!')
        
        usecols.append(cif_col)

        # 4. Load Data
        print(f"Reading file: {input_path}")
        df = pd.read_csv(input_path, sep=self.sep, encoding='utf-8', usecols=usecols)
        
        # Rename for internal consistency
        df = df.rename(columns={
            'age2': 'age_tier',
            'age': 'age_tier2',
            cif_col: 'cusno'
        })

        print(f"Unique CIFs: {df.cusno.nunique()}")
        print(f"Data shape: {df.shape}")

        # 5. Feature Engineering & Mapping
        depend_id_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        df['depend_nb'] = df['depend_id'].map(depend_id_map, na_action='ignore')

        # Convert to numeric safely
        util.to_numeric(df, np.int64, 'cusno')
        for col in ['year_addr', 'year_rel', 'year_work', 'vol_income']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 6. Categorical Validation
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
            if col in df.columns:
                df.loc[~df[col].isin(values), col] = np.nan

        # 7. Capping (Replacing Numba with standard Pandas .clip)
        # Faster and avoids ModuleNotFoundError
        df['year_addr'] = df['year_addr'].clip(upper=60).fillna(0).astype(int)
        df['year_work'] = df['year_work'].clip(upper=50).fillna(0).astype(int)
        
        # Relationship year validation
        df.loc[~df['year_rel'].between(0, 30), 'year_rel'] = np.nan

        # Final cleanup
        df = df.drop(['depend_id'], axis=1)

        # 8. Output to Volume with Safe Directory Creation
        self.safe_makedirs(os.path.dirname(output_path))
        print(f'Writing cleaned demographics to {output_path}')
        df.to_csv(output_path, sep=',', index=False)