# amfs_tm/src/lib/clean/clean_utils.py
import pandas as pd
import numpy as np
import logging
import re
import os

def apply_standard_cleaning(input_path: str, output_path: str, separator: str = ',') -> bool:
    """Applies basic cleaning using standard file I/O for Databricks compatibility."""
    if not input_path or not output_path:
        logging.error(f"Invalid paths for standard cleaning: {input_path} -> {output_path}")
        return False

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Standard file reading/writing works on Databricks Volumes/DBFS
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            first_line = True
            for line in f_in:
                if first_line:
                    # Mimics awk header quote removal
                    line = line.strip().strip('"') + '\n'
                    first_line = False

                # Clean date/time patterns
                line = re.sub(r'\b00:00:00\b', '', line)
                line = re.sub(r'(\d{4}-\d{2}-\d{2})0\b', r'\1', line)
                line = line.replace('(null)', '')

                f_out.write(line)

        logging.info(f"Applied standard cleaning to {input_path} -> {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error during standard cleaning of {input_path}: {e}")
        return False

def apply_balance_cleaning(input_path: str, output_path: str, separator: str = ',') -> bool:
    """Applies balance file specific cleaning with uppercase headers."""
    try:
        df = pd.read_csv(input_path, sep=separator)
        df.columns = df.columns.str.upper()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, sep=separator, index=False)
        logging.info(f"Applied balance cleaning to {input_path} -> {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error during balance cleaning of {input_path}: {e}")
        return False

def apply_custdemo_cleaning(input_path: str, output_path: str, separator: str = ',') -> bool:
    """Applies custdemo specific cleaning and row filtering."""
    try:
        df = pd.read_csv(input_path, sep=separator)
        
        # Original filtering logic based on 5th column
        if len(df.columns) > 4:
            col_index_4 = df.columns[4]
            df[col_index_4] = df[col_index_4].astype(str).str.strip()
            df_filtered = df[df[col_index_4].isin(['cuscls_id', 'A'])].copy()
        else:
            df_filtered = df.copy()

        # Filter based on non-empty first column
        first_col = df_filtered.columns[0]
        df_filtered = df_filtered[df_filtered[first_col].astype(str).str.strip() != ''].copy()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_filtered.to_csv(output_path, sep=separator, index=False)
        logging.info(f"Applied custdemo cleaning to {input_path} -> {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error during custdemo cleaning of {input_path}: {e}")
        return False

def apply_trxdebit_edc_cleaning(input_path: str, output_path: str, snapshot: str, separator: str = ',') -> bool:
    """Filters transaction data for the current snapshot."""
    try:
        df = pd.read_csv(input_path, sep=separator)
        if len(df.columns) > 1:
            col_index_1 = df.columns[1]
            df[col_index_1] = df[col_index_1].astype(str).str.strip()
            if df[col_index_1].str.contains(snapshot, na=False, case=False).any():
                df_filtered = df[df[col_index_1].isin(['trx_month', snapshot])].copy()
            else:
                df_filtered = df.copy()
        else:
            df_filtered = df.copy()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_filtered.to_csv(output_path, sep=separator, index=False)
        logging.info(f"Applied trxdebit_edc cleaning to {input_path} -> {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error during trxdebit_edc cleaning of {input_path}: {e}")
        return False

def apply_trxnet_cleaning(input_path: str, output_path: str, separator: str = '|') -> bool:
    """Standardizes CIF columns for transaction net data."""
    try:
        df = pd.read_csv(input_path, sep=separator, dtype={'cifno_01': str, 'cifno_15': str})
        df['cifno'] = df['cifno_01'].fillna(df['cifno_15'])
        df['cifno'] = pd.to_numeric(df['cifno'], errors='coerce')
        df.dropna(subset=['cifno'], how='any', inplace=True)
        df['cifno'] = df['cifno'].astype('Int64')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, sep=separator, index=False)
        logging.info(f"Applied trxnet cleaning to {input_path} -> {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error during trxnet cleaning of {input_path}: {e}")
        return False