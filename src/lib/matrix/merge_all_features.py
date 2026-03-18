# amfs_tm/src/lib/matrix/merge_all_features.py
import os
import glob
import pandas as pd
import yaml
import logging
from lib import util

def create_modeling_matrix_pandas_dynamic(main_config_file, arg_main):
    """
    Dynamically merges all feature CSVs into one master matrix.
    Updated to use Databricks Volume paths.
    """
    snapshot = arg_main.snapshot
    with open(main_config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = config['PATHS']['base_path']
    proc_dir = config['PATHS']['processed_dir']
    feat_dir = config['PATHS']['features_dir']
    bm_subdir = config['PATHS']['bm_features_subdir']
    matrix_dir = config['PATHS']['matrix_output_dir']

    # 1. Start with the Base Population (CIF with flags)
    # Standard Volume path: /Volumes/catalog/schema/vol/amfs_tm/data/03_features/BM_features/
    base_pop_path = os.path.join(base_path, feat_dir, bm_subdir, f"cif_with_flag_{snapshot}_active_complete.csv")
    
    print(f"Loading base population from: {base_pop_path}")
    df_matrix = pd.read_csv(base_pop_path)
    util.to_numeric(df_matrix, int, 'cifno')

    # 2. Identify all feature files in the BM_features Volume directory
    feature_folder = os.path.join(base_path, feat_dir, bm_subdir)
    # Find all CSVs for this snapshot, excluding the base population file itself
    feature_files = glob.glob(os.path.join(feature_folder, f"*_{snapshot}_feat.csv"))
    # Also catch those with 'lag' in the name
    feature_files += glob.glob(os.path.join(feature_folder, f"*_{snapshot}_lag_feat.csv"))

    print(f"Found {len(feature_files)} feature files to merge.")

    for fpath in feature_files:
        fname = os.path.basename(fpath)
        print(f"Merging: {fname}")
        
        df_feature = pd.read_csv(fpath)
        
        # Standardize key for merging
        if 'cifno_pengirim' in df_feature.columns:
            df_feature = df_feature.rename(columns={'cifno_pengirim': 'cifno'})
        if 'customer_no' in df_feature.columns:
            df_feature = df_feature.rename(columns={'customer_no': 'cifno'})
        
        util.to_numeric(df_feature, int, 'cifno')
        
        # Left join to maintain the base population size
        df_matrix = df_matrix.merge(df_feature, on='cifno', how='left')

    # 3. Save the final un-imputed matrix
    output_path = os.path.join(base_path, matrix_dir, f"matrix_raw_{snapshot}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving raw matrix to: {output_path}")
    df_matrix.to_csv(output_path, index=False)
    return output_path