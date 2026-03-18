# amfs_tm/src/lib/feature/features_generation.py
import os
import logging
import importlib
from lib.clean import clean_config_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def features_generation(main_config_path, features_config_path, args):
    main_cfg = clean_config_utils.load_config(main_config_path)
    feat_cfg = clean_config_utils.load_config(features_config_path)
    
    if not main_cfg or not feat_cfg:
        return

    paths = main_cfg.get('PATHS', {})
    base_path = paths.get('base_path')
    proc_vol = paths.get('processed_dir_prefix')  
    feat_vol = paths.get('features_dir_prefix')   
    bm_feat_dir = paths.get('bm_features_subdir')
    
    snapshot = args.snapshot
    sep = '|' 

    data_rel_path = os.path.join(proc_vol, "data_from_BM")
    out_rel_path = os.path.join(feat_vol, bm_feat_dir)

    logging.info(f"--- Feature Generation Start: {snapshot} ---")

    for feat_step in feat_cfg.get('features_to_run', []):
        if not feat_step.get('enabled', False):
            continue

        module_name = feat_step.get('module')
        class_name = feat_step.get('class')
        
        try:
            module = importlib.import_module(module_name)
            feat_class = getattr(module, class_name)
            
            instance = feat_class(
                root=base_path,
                data_path=data_rel_path,
                out_path=out_rel_path,
                sep=sep,
                snapshot=snapshot
            )

            methods = feat_step.get('methods', ['create'])
            for m_name in methods:
                logging.info(f"Executing {class_name}.{m_name}")
                getattr(instance, m_name)()

        except Exception as e:
            logging.error(f"Failed feature {class_name}: {e}", exc_info=True)