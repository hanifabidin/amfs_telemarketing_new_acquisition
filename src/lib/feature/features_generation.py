# amfs_tm/src/lib/feature/features_generation.py
import argparse
import os
import sys
import warnings
from datetime import datetime

import logging
import yaml
import configparser

import pandas as pd

# These imports refer to your local library structure
from lib.feature import *
from lib.segmentation import ApplyFlag

warnings.filterwarnings("ignore")

# Mapping of aliases to class objects remains the same
dags = {
    'bankavr': BankAvr,
    'creditcard_type': CreditType,
    'debitcard_type': DebitType,
    'custdemo': CustDemo,
    'balance': BankBalance,
    'bankph': BankPH,
    'maxtrxdate': MaxTrDate,
    'trxdebit_edc': TrxDebitEdc,
    'trxnet': TrxNet,
    'trxoutflow': TrxOutflow,
    'trxoutflow_payment': TrxOutflowPayment,
    'trxper': TrxPer,
    'zipcode': CustZipcode,
    'tso': TSO,
    'flag': ApplyFlag,
    'fpf': FPF
}

def features_generation(main_config_file, features_config, arg_main):
    """
    Updated for Databricks: 
    - Removes S3 bucket/prefix logic.
    - Uses absolute paths for Volumes or DBFS.
    """
    
    with open(main_config_file, 'r') as f:
        main_config = yaml.safe_load(f)

    # In Databricks, we use the unified base_path defined in your updated YAML
    # Example: /Volumes/training/amfs_data/main_volume/amfs_tm
    base_path = main_config['PATHS']['base_path']
    
    # Sub-directory relative paths from config
    processed_dir = main_config['PATHS']['processed_dir']
    features_dir = main_config['PATHS']['features_dir']
    bm_feature_subdir = main_config['PATHS']['bm_features_subdir']

    # Construct absolute POSIX paths
    feature_output_path = os.path.join(base_path, features_dir, bm_feature_subdir)
    processed_data_path = os.path.join(base_path, processed_dir)
    
    config = configparser.ConfigParser()
    config.read(features_config)

    for section in config.sections():
        job_alias = section
        job_type = config[section]['job_type']
        job_functions = config[section]['job_functions'].split(',')
        is_job_run = config[section]['is_job_run']

        if is_job_run == '1':
            print(f'Start creating {job_type}: {job_alias}')
            
            # CHANGE: Removed 's3://' + bucket_name + project logic.
            # We pass the absolute base_path as the 'root'.
            # Your DAG classes (BankAvr, etc.) must be updated to use os.path.join 
            # instead of S3-specific logic inside their constructors.
            obj = dags[job_alias](
                base_path, 
                processed_data_path, 
                feature_output_path, 
                None, 
                arg_main.snapshot
            )

            for func in job_functions:
                print(f'>>> Start running: {obj.__class__.__name__}.{func}()\n')
                try:
                    # eval remains standard Python, but ensure the underlying methods 
                    # in your classes use local file I/O or Spark.
                    method = getattr(obj, func.strip())
                    method()
                except AttributeError:
                    print(f'Error: {obj.__class__.__name__} has no function {func}')
                except Exception as e:
                    print(f'Execution error in {func}: {e}')