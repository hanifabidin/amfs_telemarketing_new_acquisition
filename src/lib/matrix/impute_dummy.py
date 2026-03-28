import pandas as pd
import numpy as np
import os
import logging
import yaml
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 1. CONFIGURATION DICTIONARIES ---

# Dictionary for renaming columns to remove spaces (Standardizing for Models)
CHANGE_NAMES = {
    'bill payment_freq_trxout': 'billpayment_freq_trxout',
    'bill payment_amt_trxout': 'billpayment_amt_trxout',
    'bill payment_amt_per_trxout': 'billpayment_amt_per_trxout',
    'credit card_freq_trxout': 'creditcard_freq_trxout',
    'credit card_amt_trxout': 'creditcard_amt_trxout',
    'credit card_amt_per_trxout': 'creditcard_amt_per_trxout',
    'interbank transfer_freq_trxout': 'interbanktransfer_freq_trxout',
    'interbank transfer_amt_trxout': 'interbanktransfer_amt_trxout',
    'interbank transfer_amt_per_trxout': 'interbanktransfer_amt_per_trxout',
    'intrabank transfer_freq_trxout': 'intrabanktransfer_freq_trxout',
    'intrabank transfer_amt_trxout': 'intrabanktransfer_amt_trxout',
    'intrabank transfer_amt_per_trxout': 'intrabanktransfer_amt_per_trxout',
    'personal loan_freq_trxout': 'personalloan_freq_trxout',
    'personal loan_amt_trxout': 'personalloan_amt_trxout',
    'personal loan_amt_per_trxout': 'personalloan_amt_per_trxout',
    'tv cable_freq_trxout': 'tvcable_freq_trxout',
    'tv cable_amt_trxout': 'tvcable_amt_trxout',
    'tv cable_amt_per_trxout': 'tvcable_amt_per_trxout'
}

# Mapping for imputation strategy
IMPUTE_DICT = [
    {'feature': 'nominal_capita', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'ppp_capita', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'nominal', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'ppp', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'population', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'maxdt_range_cnts', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'mindt_range_cnts', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'age_tier', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'year_addr', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'year_rel', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'year_work', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'vol_income', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'date_org_vintage', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'depend_nb', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'totph_bank_ph', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'saving_acct_vint_l_bank_ph', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'saving_acct_vint_f_bank_ph', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'saving_acct_delta_bank_ph', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'all_acct_vint_l_bank_ph', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'all_acct_vint_f_bank_ph', 'impute_with': 'CUST_MEDIAN'},
    {'feature': 'houstyp_id', 'impute_with': 'CREATE_CATEGORY'},
    {'feature': 'occp_id', 'impute_with': 'CREATE_CATEGORY'},
    {'feature': 'cusedu_id', 'impute_with': 'CREATE_CATEGORY'},
    {'feature': 'marital_id', 'impute_with': 'CREATE_CATEGORY'},
    {'feature': 'empinds_id', 'impute_with': 'CREATE_CATEGORY'},
    {'feature': 'gender_id', 'impute_with': 'CUST_MODE'},
    {'feature': 'intuser_fg', 'impute_with': 'CUST_MODE'},
    {'feature': 'mbluser_fg', 'impute_with': 'CUST_MODE'},
    {'feature': 'prior_fg', 'impute_with': 'CUST_MODE'},
    {'feature': 'rsdnt_fg', 'impute_with': 'CUST_MODE'}
]

CATEGORICAL_FEATURES = {
    'intuser_fg': ['Y'],
    'mbluser_fg': ['Y'],
    'prior_fg': ['Y'],
    'rsdnt_fg': ['Y'],
    'gender_id': ['M'],
    'occp_id': ['PSW', 'PNS', 'PBUM', 'MHS', 'WSW', 'IRT', 'MIL', 'PRMDS', 'POL', 'PROF', 'GR', 'KOP', 'PTN',
                 'DKTR', 'PIF', 'ENGR', 'YYS', 'PKS', 'PENS', 'PGCR', 'PNG', 'NTRS', 'PKJSN', 'KONS', 'WRTWN', 'AKT'],
    'houstyp_id': ['MILIK', 'KANTR', 'KLRGA', 'KOST', 'DINAS', 'SEWA', 'PROJ'],
    'marital_id': ['B', 'A', 'C', 'D'],
    'cusedu_id': ['B', 'D', 'A', 'F', 'E', 'C'],
    'empinds_id': [9, 3, 7, 5, 1, 8, 6, 6006, 2, 6009, 6008, 7001, 4, 6003, 8001, 6007, 9001, 6002, 6001, 6005, 6004]
}

USELESS_COLUMNS = [
    'cuscls_id', 'sum_purchase_amt_per_trxout', 'sum_db_trxper',
    'sum_cd_trxper', 'mxdt_range_lm', 'mndt_range_lm', 'mxdt_range_lm2', 'mndt_range_lm2',
    'mxdt_range_lm3', 'mndt_range_lm3', 'delta_range_lm3', 'delta_range_lm2', 'weekday_mxdt',
    'weekday_mndt', 'trx_month_db_edc', 'east'
]

# --- 2. CORE LOGIC ---

def _impute_static(df: pd.DataFrame):
    """Applies the imputation rules based on IMPUTE_DICT."""
    categorical_cols_to_drop = []
    for item in IMPUTE_DICT:
        feature = item['feature']
        impute_with = item['impute_with']

        if feature not in df.columns:
            continue

        if df[feature].notnull().all():
            continue

        v = None
        if impute_with == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
            categorical_cols_to_drop.append(feature)
        elif impute_with == 'CUST_MEDIAN':
            # Force numeric for median safety
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            v = df[feature].median() 
            if pd.isna(v): v = 0
        elif impute_with == 'CUST_MODE':
            counts = df[feature].value_counts(dropna=True)
            v = counts.index[0] if not counts.empty else 'NULL_CATEGORY'
            categorical_cols_to_drop.append(feature)

        if v is not None:
            # Categorical modes must be strings
            if impute_with in ['CREATE_CATEGORY', 'CUST_MODE'] and not isinstance(v, (int, float)):
                 v = str(v)
            df[feature] = df[feature].fillna(v)
            logging.info(f"Imputed {feature} with {v}")

    return categorical_cols_to_drop

def impute_and_dummy_matrix(main_config_path, matrix_config_path, args):
    """
    Orchestrator for Matrix Imputation and Dummification.
    """
    snapshot = args.snapshot
    with open(main_config_path, 'r') as f:
        main_cfg = yaml.safe_load(f)
    with open(matrix_config_path, 'r') as f:
        matrix_cfg = yaml.safe_load(f)

    root = main_cfg['PATHS']['base_path']
    campaign = (datetime.strptime(snapshot, '%Y%m') + relativedelta(months=1)).strftime('%Y%m')

    # Resolve I/O Paths
    input_path = os.path.join(root, matrix_cfg['settings']['merge_output'].format(
        campaign=campaign, snapshot=snapshot, dil=args.dil))
    output_path = os.path.join(root, matrix_cfg['settings']['final_output'].format(
        campaign=campaign, snapshot=snapshot, dil=args.dil))

    logging.info(f"Loading raw matrix for imputation: {input_path}")
    
    # HARDENING: Fix mixed type warning
    df = pd.read_csv(input_path, low_memory=False)
    # Ensure all columns start as lowercase for matching dictionaries
    df.columns = df.columns.str.lower().str.strip()
    
    logging.info(f"Loaded matrix shape: {df.shape}")

    # 1. Date Preprocessing
    snapshot_date = pd.to_datetime(f"{snapshot[:4]}-{snapshot[4:]}-01")
    if 'date_org' in df.columns:
        # HARDENING: Fix dayfirst warning
        df['date_org'] = pd.to_datetime(df['date_org'], dayfirst=True, errors='coerce')
        df.loc[(df['date_org'] < '1910-01-01') | (df['date_org'] > '2200-01-01'), 'date_org'] = pd.NaT
        df['date_org_vintage'] = (snapshot_date - df['date_org']).dt.days / 30.4375

    # 2. Delta Range Calculations
    delta_cols = ['mxdt_range_lm', 'mndt_range_lm', 'mxdt_range_lm2', 'mndt_range_lm2', 'mxdt_range_lm3', 'mndt_range_lm3']
    if all(col in df.columns for col in delta_cols):
        for col in delta_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['delta_range_lm'] = df['mxdt_range_lm'] - df['mndt_range_lm']
        df['delta_range_lm2'] = df['mxdt_range_lm2'] - df['mndt_range_lm2']
        df['delta_range_lm3'] = df['mxdt_range_lm3'] - df['mndt_range_lm3']
        
        delta_list = ['delta_range_lm', 'delta_range_lm2', 'delta_range_lm3']
        df['delta_range_cnt'] = df[delta_list].apply(lambda x: x.nunique(), axis=1)
        df['delta_range_avg'] = df[delta_list].mean(axis=1)
        df.loc[df['delta_range_cnt'] == 0, 'delta_range_cnt'] = df['delta_range_cnt'].median()

    # 3. Rename, Impute, and Global Fill
    # This uses the CHANGE_NAMES dictionary defined at the top
    df = df.rename(columns=CHANGE_NAMES)
    categorical_cols_to_drop = _impute_static(df)
    df = df.fillna(0)

    # 4. Dummification
    for feature, values_to_dummy in CATEGORICAL_FEATURES.items():
        if feature not in df.columns:
            continue
            
        logging.info(f"Dummifying: {feature}")
        for value in values_to_dummy:
            new_col = f"{feature}_{value}".lower()
            df[new_col] = (df[feature].astype(str) == str(value)).astype(int)
            
        if len(values_to_dummy) > 1:
            other_col = f"{feature}_other_categories".lower()
            df[other_col] = (~df[feature].astype(str).isin([str(v) for v in values_to_dummy])).astype(int)

    # 5. Clean-up and Export
    final_drop = list(set(USELESS_COLUMNS + list(CATEGORICAL_FEATURES.keys()) + categorical_cols_to_drop))
    df = df.drop(columns=[c for c in final_drop if c in df.columns], errors='ignore')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"✅ Final matrix complete: {output_path}")

    return output_path