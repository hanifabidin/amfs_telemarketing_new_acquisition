import pandas as pd
import numpy as np
import os
import re
import logging
import yaml
import s3fs # Required for Pandas/S3 interaction

# Set up logging for output messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 1. Imputation and Dummification Dictionaries (from the provided template) ---

# Mapping for imputation strategy
impute_dict = (
    # CIF LEVEL MEDIAN IMPUTATION
    {'feature': 'nominal_capita', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'ppp_capita', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'nominal', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'ppp', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'population', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'EAST', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'WEST', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'maxdt_range_cnts', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'mindt_range_cnts', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'age_tier', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'year_addr', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'year_rel', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'year_work', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'vol_income', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'date_org_vintage', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'depend_nb', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'TOTPH_bank_ph', 'impute_with': 'CUST_MEDIAN'}
    # CHECK
    , {'feature': 'saving_acct_vint_l_bank_ph', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'saving_acct_vint_f_bank_ph', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'saving_acct_delta_bank_ph', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'all_acct_vint_l_bank_ph', 'impute_with': 'CUST_MEDIAN'}
    , {'feature': 'all_acct_vint_f_bank_ph', 'impute_with': 'CUST_MEDIAN'}
    # NEW CATEGORY IMPUTATION
    , {'feature': 'houstyp_id', 'impute_with': 'CREATE_CATEGORY'}
    , {'feature': 'occp_id', 'impute_with': 'CREATE_CATEGORY'}
    , {'feature': 'cusedu_id', 'impute_with': 'CREATE_CATEGORY'}
    , {'feature': 'marital_id', 'impute_with': 'CREATE_CATEGORY'}
    , {'feature': 'empinds_id', 'impute_with': 'CREATE_CATEGORY'}
    # MODE IMPUTATION
    , {'feature': 'gender_id', 'impute_with': 'CUST_MODE'}
    , {'feature': 'intuser_fg', 'impute_with': 'CUST_MODE'}
    , {'feature': 'mbluser_fg', 'impute_with': 'CUST_MODE'}
    , {'feature': 'prior_fg', 'impute_with': 'CUST_MODE'}
    , {'feature': 'rsdnt_fg', 'impute_with': 'CUST_MODE'}
)

# Values to be dummified (one-hot encoded)
categorical_features = {
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

# Column renaming logic
change_names = {
    'BILL PAYMENT_freq_trxout': 'BILLPAYMENT_freq_trxout',
    'BILL PAYMENT_amt_trxout': 'BILLPAYMENT_amt_trxout',
    'BILL PAYMENT_amt_per_trxout': 'BILLPAYMENT_amt_per_trxout',
    'CREDIT CARD_freq_trxout': 'CREDITCARD_freq_trxout',
    'CREDIT CARD_amt_trxout': 'CREDITCARD_amt_trxout',
    'CREDIT CARD_amt_per_trxout': 'CREDITCARD_amt_per_trxout',
    'INTERBANK TRANSFER_freq_trxout': 'INTERBANKTRANSFER_freq_trxout',
    'INTERBANK TRANSFER_amt_trxout': 'INTERBANKTRANSFER_amt_trxout',
    'INTERBANK TRANSFER_amt_per_trxout': 'INTERBANKTRANSFER_amt_per_trxout',
    'INTRABANK TRANSFER_freq_trxout': 'INTRABANKTRANSFER_freq_trxout',
    'INTRABANK TRANSFER_amt_trxout': 'INTRABANKTRANSFER_amt_trxout',
    'INTRABANK TRANSFER_amt_per_trxout': 'INTRABANKTRANSFER_amt_per_trxout',
    'PERSONAL LOAN_freq_trxout': 'PERSONALLOAN_freq_trxout',
    'PERSONAL LOAN_amt_trxout': 'PERSONALLOAN_amt_trxout',
    'PERSONAL LOAN_amt_per_trxout': 'PERSONALLOAN_amt_per_trxout',
    'TV CABLE_freq_trxout': 'TVCABLE_freq_trxout',
    'TV CABLE_amt_trxout': 'TVCABLE_amt_trxout',
    'TV CABLE_amt_per_trxout': 'TVCABLE_amt_per_trxout'
}

# Columns to be dropped
useless_columns = [
    'cuscls_id', 'sum_purchase_amt_per_trxout', 'sum_db_trxper',
    'sum_cd_trxper', 'MXDT_range_lm', 'MNDT_range_lm', 'MXDT_range_lm2', 'MNDT_range_lm2',
    'MXDT_range_lm3', 'MNDT_range_lm3', 'delta_range_lm3', 'delta_range_lm2', 'weekday_MXDT',
    'weekday_MNDT', 'trx_month_db_edc', 'EAST',
    # Add base categorical columns to be dropped after dummification
]

# --- 2. Core Logic Functions (Adapted from FinalMatrix class) ---

def _impute_static(data_set: pd.DataFrame):
    """Applies the imputation rules based on impute_dict."""
    for impute_item in impute_dict:
        feature = impute_item['feature']
        impute_with = impute_item['impute_with']

        if feature not in data_set.columns:
            continue

        if data_set[feature].notnull().all():
            continue

        v = None
        if impute_with == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif impute_with == 'CUST_MEDIAN':
            # Calculate median from unique values across all customers
            v = data_set[feature].median() 
        elif impute_with == 'CUST_MODE':
            # --- FIX APPLIED HERE ---
            value_counts = data_set[feature].value_counts(dropna=True)
            if value_counts.empty:
                v = 'NULL_CATEGORY' # Fallback to category if column is all NaN
            else:
                v = value_counts.index[0] # Get the most frequent non-NaN value
            # ------------------------

        if v is not None:
            # Use fillna() with value conversion to handle numeric/string types
            # Note: The original logic failed to impute NaN for median on numeric columns, 
            # so the logic below handles the imputation of `nan` values with the calculated median `v`.
            
            # Check if v is still NaN (happens if median is taken over only NaNs)
            if pd.isna(v) and impute_with == 'CUST_MEDIAN':
                 # Use 0 as final fallback for numeric medians that couldn't be calculated
                 v = 0 
                 
            # Cast v to string if it's a category creation
            if impute_with in ['CREATE_CATEGORY', 'CUST_MODE'] and v != 0:
                 v = str(v)
                 
            # Apply the final imputation
            data_set[feature] = data_set[feature].fillna(v)
            logging.info(f"Imputed missing values in feature {feature} with value {v}")

    # Return the list of categorical columns for dropping later
    return [item['feature'] for item in impute_dict if item['impute_with'] in ['CREATE_CATEGORY', 'CUST_MODE']]


def impute_and_dummy_matrix(main_config_file, arg_main):
    
    # --- 3. Path Calculation and Load Input ---
    snapshot = arg_main.snapshot
    
    with open(main_config_file, 'r') as f:
        main_config = yaml.safe_load(f)

    # Calculate input/output paths based on the previous merging script
    S3_BUCKET = main_config['PATHS']['bucket_name']
    PROJECT_ROOT = main_config['PATHS']['root_prefix']
    matrix_output_dir = main_config['PATHS']['matrix_output_dir_prefix']
    
    # Input path: Output from the merge script
    INPUT_FILENAME = f"final_modeling_matrix_{snapshot}.csv"
    INPUT_PATH = os.path.join('s3://', S3_BUCKET, PROJECT_ROOT, matrix_output_dir, INPUT_FILENAME)
    
    # Output path: New dummified file in the same directory
    OUTPUT_FILENAME = f"dummified_modeling_matrix_{snapshot}.csv"
    FINAL_OUTPUT_PATH = os.path.join('s3://', S3_BUCKET, PROJECT_ROOT, matrix_output_dir, OUTPUT_FILENAME)
    
    logging.info(f"Processing matrix from: {INPUT_PATH}")
    
    try:
        data_set = pd.read_csv(INPUT_PATH, index_col=False)
        logging.info(f"Loaded matrix with shape: {data_set.shape}")
    except Exception as e:
        logging.error(f"Failed to load input matrix from {INPUT_PATH}. Error: {e}")
        return

    # --- 4. Feature Engineering and Preprocessing ---
    
# Snapshot date for vintage calculation
    snapshot_date_str = f"{snapshot[:4]}-{snapshot[4:]}-01"
    snapshot_date = pd.to_datetime(snapshot_date_str)
    
    # Handle outlier and create date_org_vintage
    # 1. Convert 'date_org' to datetime, coercing errors to NaT
    data_set['date_org'] = pd.to_datetime(data_set['date_org'], format='%d-%m-%Y', errors='coerce')

    # 2. Handle outliers (dates before 1910 or after 2200)
    data_set.loc[
        (data_set['date_org'] < '1910-01-01') | (data_set['date_org'] > '2200-01-01'), 
        'date_org'
    ] = pd.NaT
    
    # 3. FIX: Calculate vintage in DAYS and divide by average days in a month (30.4375)
    data_set['date_org_vintage'] = (snapshot_date - data_set['date_org']).dt.days / 30.4375
    # Create delta_range features
    delta_range = ['delta_range_lm', 'delta_range_lm2', 'delta_range_lm3']
    if all(col in data_set.columns for col in ['MXDT_range_lm', 'MNDT_range_lm', 'MXDT_range_lm2', 'MNDT_range_lm2', 'MXDT_range_lm3', 'MNDT_range_lm3']):
        data_set['delta_range_lm'] = data_set['MXDT_range_lm'] - data_set['MNDT_range_lm']
        data_set['delta_range_lm2'] = data_set['MXDT_range_lm2'] - data_set['MNDT_range_lm2']
        data_set['delta_range_lm3'] = data_set['MXDT_range_lm3'] - data_set['MNDT_range_lm3']
        data_set['delta_range_cnt'] = data_set[delta_range].apply(lambda x: x.nunique(), axis=1)
        data_set['delta_range_avg'] = data_set[delta_range].mean(axis=1)

        # Simple impute for delta_range_cnt
        v_median = data_set['delta_range_cnt'].median()
        data_set.loc[data_set['delta_range_cnt'] == 0, 'delta_range_cnt'] = v_median
    else:
        logging.warning("Skipping delta_range feature creation: Missing required columns (MXDT/MNDT_range).")

    # Rename columns
    data_set = data_set.rename(columns=change_names)

    # Impute missing values
    categorical_cols_to_drop = _impute_static(data_set)
    
    # Fill every other remaining nan columns to ZERO (assuming numerics)
    data_set = data_set.fillna(0)

    # --- 5. Dummification ---
    fmt = '{0}_{1}'
    all_dummied_cols = []

    for feature, values_to_dummy in categorical_features.items():
        if feature not in data_set.columns:
            logging.warning(f"Skipping dummification for '{feature}': Column not found.")
            continue

        logging.info(f"Dummifying feature {feature}")
        
        # Create columns for specified values
        for value in values_to_dummy:
            new_col = fmt.format(feature, value)
            data_set[new_col] = 0
            # Use data_set[feature].astype(str) to handle mixed types gracefully
            data_set.loc[data_set[feature].astype(str) == str(value), new_col] = 1
            all_dummied_cols.append(new_col)

        # Create 'OTHER_CATEGORIES' column if more than one possible value is specified
        if len(values_to_dummy) > 1:
            new_col = fmt.format(feature, 'OTHER_CATEGORIES')
            data_set[new_col] = 0
            data_set.loc[~data_set[feature].astype(str).isin([str(v) for v in values_to_dummy]), new_col] = 1
            all_dummied_cols.append(new_col)

    # --- 6. Final Column Clean-up ---
    
    # Combine lists of columns to drop
    final_useless_cols = list(set(useless_columns + list(categorical_features.keys()) + categorical_cols_to_drop))
    
    # Drop columns that exist in the DataFrame
    cols_to_drop = [col for col in final_useless_cols if col in data_set.columns]
    
    data_set = data_set.drop(columns=cols_to_drop, errors='ignore')
    
    logging.info(f"Final preprocessed matrix shape: {data_set.shape}")
    
    # --- 7. Write Final Matrix to S3 ---
    logging.info(f"Writing final dummified matrix to {FINAL_OUTPUT_PATH}...")
    data_set.to_csv(FINAL_OUTPUT_PATH, index=False)
    logging.info("✨ Dummification and imputation complete. Matrix uploaded.")