# amfs_tm/src/lib/feature/axa/tso.py
import os
import re
import numpy as np
import pandas as pd
from lib import obj, util

class TSO(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        # Initializing via base class
        super().__init__(root, data_path, out_path, sep, snapshot)
        
        # Source from Processed Volume (AMFS Data)
        self.tso_path = os.path.join(
            self.root, 
            '02_processed', 
            'data_from_AMFS', 
            self.snapshot, 
            f'TSO_{self.snapshot}.csv'
        )
        
        # FIXED OUTPUT PATH: Strictly inside AXA_Features
        self.feature_path = os.path.join(
            self.root, 
            '03_features', 
            'AXA_Features', 
            f'TSO_Features_{self.snapshot}.csv'
        )

    def create(self):
        """Processes TSO features with hardened column detection and MPPT/MPA/MSK/MSL logic."""
        print(f"--- Starting TSO.create: {self.snapshot} ---")
        print(f"Checking path: {self.tso_path}")
        
        if not os.path.exists(self.tso_path):
            print(f"TSO Source file missing: {self.tso_path}")
            return

        # 1. Load Data with auto-sep detection and BOM handling
        tso = pd.read_csv(self.tso_path, sep=None, engine='python', encoding='utf-8-sig')
        tso.columns = tso.columns.str.upper().str.strip()

        # Fix: Detect if the file was read as a single column (common when Tab vs Comma fails)
        if len(tso.columns) == 1 and not tso.empty:
            if '\t' in str(tso.iloc[0, 0]):
                print("Detected Tab-separation in CSV. Re-parsing...")
                tso = pd.read_csv(self.tso_path, sep='\t', encoding='utf-8-sig')
                tso.columns = tso.columns.str.upper().str.strip()

        # 2. Key Column Mapping (Handle aliases/prefixes)
        col_map = {
            'CALL_AGENT_ID': ['CALL_AGENT_ID', 'AGENT_ID', 'CALLID'],
            'TSO_LOCATION': ['TSO_LOCATION', 'LOCATION'],
            'TSO_LOS': ['TSO_LOS', 'LOS'],
            'TSO_AGE': ['TSO_AGE', 'AGE'],
            'TSO_DEPENDANT': ['TSO_DEPENDANT', 'DEPENDANT', 'TSO_DEPENDENT'],
            'TSO_STATUS': ['TSO_STATUS', 'STATUS']
        }

        for target, aliases in col_map.items():
            if target not in tso.columns:
                for alias in aliases:
                    if alias in tso.columns:
                        tso = tso.rename(columns={alias: target})
                        break

        # Safety Check
        critical_cols = ['CALL_AGENT_ID', 'TSO_LOS']
        missing = [c for c in critical_cols if c not in tso.columns]
        if missing:
            print(f"ERROR: Critical columns {missing} missing from TSO file.")
            print(f"Available columns: {tso.columns.tolist()}")
            return

        # 3. Type Hardening
        util.to_numeric(tso, np.int64, 'CALL_AGENT_ID')
        util.to_numeric(tso, None, 'TSO_LOS', 'TSO_AGE', 'TSO_DEPENDANT')

        # Identify performance columns
        perf_pattern = r'^TSO_.*_LM[1-3]*$'
        performance_cols = [c for c in tso.columns if re.search(perf_pattern, c)]
        for col in performance_cols:
            tso[col] = pd.to_numeric(tso[col], errors='coerce').fillna(0.0)

        # 4. Categorical Cleaning & Imputation
        categorical_features = {
            'TSO_LOCATION': ['01.BBD', '02.AXTO'],
            'TSO_GENDER': ['01.FEMALE', '02.MALE'],
            'TSO_CATEGORY': ['01.TOP GUN', '02.STRIKER', '03.PLAYMAKER', '04.DEFENDER'],
            'TSO_MARITAL_STATUS': ['01.SIN', '02.MAR'],
            'TSO_EDUCATION': ['01.S2', '02.S1', '03.AKADEMI', '04.SLTA']
        }

        for feat in categorical_features.keys():
            if feat not in tso.columns:
                tso[feat] = np.nan

        for feature, categories in categorical_features.items():
            tso.loc[~tso[feature].isin(categories), feature] = np.nan

        dict_imp = {'TSO_LOCATION': 'MODE', 'TSO_GENDER': 'MODE', 'TSO_CATEGORY': 'MODE',
                    'TSO_MARITAL_STATUS': 'MODE', 'TSO_EDUCATION': 'MODE',
                    'TSO_LOS': 'MEDIAN', 'TSO_AGE': 'MEDIAN', 'TSO_DEPENDANT': 'MEDIAN'}

        for feature, method in dict_imp.items():
            if tso[feature].isnull().all():
                val = 0 if method == 'MEDIAN' else 'UNKNOWN'
            else:
                val = tso[feature].value_counts().index[0] if method == 'MODE' else tso[feature].median()
            tso[feature] = tso[feature].fillna(val)

        # 5. Feature Aggregation
        feature_columns = ['CALL_AGENT_ID', 'ACTIVE_FLAG']
        posfixes = ['LM', 'LM2', 'LM3']
        
        # Base Performance (DB/Call)
        col_prefixes = ['TSO_DB', 'TSO_DB_CONTACTED', 'TSO_CALL', 'TSO_CALL_CONTACTED']
        for pref in col_prefixes:
            for pos in posfixes:
                f_col = f'{pref}_{pos}'
                cols = [c for c in tso.columns if re.search(f'^TSO_{pref}_SA_{pos}$', c)]
                if cols:
                    feature_columns.append(f_col)
                    tso[f_col] = tso[cols].sum(axis=1)

        # Product Performance (MPPT, MPA, MSK, MSL)
        products = ['MPPT', 'MPA', 'MSK', 'MSL', 'TOTAL']
        for prod in products:
            for pos in posfixes:
                for method in ['APE', 'CASES']:
                    f_col = f'TSO_{prod}_{method}_{pos}'
                    cols = [c for c in tso.columns if re.search(f'^TSO_{prod}_SA_{method}_{pos}$', c)]
                    if cols:
                        feature_columns.append(f_col)
                        tso[f_col] = tso[cols].sum(axis=1)

        # 6. Ratios & Dummies
        def safe_div(row, n, d):
            return row[n] / row[d] if d in row and row[d] > 0 else 0

        for pos in posfixes:
            # Contact rates
            if f'TSO_DB_{pos}' in tso.columns and f'TSO_DB_CONTACTED_{pos}' in tso.columns:
                f_rate = f'TSO_CONTACT_RATE_BY_DB_{pos}'
                feature_columns.append(f_rate)
                tso[f_rate] = tso.apply(lambda r: safe_div(r, f'TSO_DB_CONTACTED_{pos}', f'TSO_DB_{pos}'), axis=1)

        # Categorical Dummies
        for feature, categories in categorical_features.items():
            for category in categories:
                f_col = f'{feature}_{category}'
                feature_columns.append(f_col)
                tso[f_col] = (tso[feature] == category).astype(int)

        tso['ACTIVE_FLAG'] = (tso.get('TSO_STATUS', '') == 'ACTIVE').astype(int)

        # Final Save
        final_cols = sorted(list(set([c for c in feature_columns if c in tso.columns] + 
                                     ['TSO_LOS', 'TSO_AGE', 'TSO_DEPENDANT'])))
        
        self.safe_makedirs(os.path.dirname(self.feature_path))
        tso[final_cols].fillna(0).to_csv(self.feature_path, index=False)
        print(f"TSO Features successfully saved: {self.feature_path}")