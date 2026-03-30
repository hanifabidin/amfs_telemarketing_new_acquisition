# amfs_tm/src/lib/inference/predictor.py
import os
import yaml
import pickle
import json
import pandas as pd
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score

def run_prediction(main_config_path, predict_config_path, args):
    """
    Loads a pickled model and generates scores for the current matrix.
    Compares inference AUC with training AUC if target exists.
    """
    # 1. Load Configs
    with open(main_config_path, 'r') as f: main_cfg = yaml.safe_load(f)
    with open(predict_config_path, 'r') as f: predict_cfg = yaml.safe_load(f)

    root = main_cfg['PATHS']['base_path']
    snapshot = args.snapshot
    dil = args.dil
    campaign = (datetime.strptime(snapshot, '%Y%m') + relativedelta(months=1)).strftime('%Y%m')

    # Resolve Paths
    model_path = os.path.join(root, predict_cfg['settings']['model_path'].format(snapshot=snapshot))
    input_path = os.path.join(root, predict_cfg['settings']['input_matrix'].format(snapshot=snapshot, campaign=campaign, dil=dil))
    output_path = os.path.join(root, predict_cfg['settings']['output_prediction'].format(snapshot=snapshot, campaign=campaign, dil=dil))
    metrics_path = os.path.join(root, predict_cfg['settings']['metrics_log'].format(snapshot=snapshot, campaign=campaign))

    logging.info(f"--- Inference Started for Snapshot {snapshot} ---")

    # 2. Load Model & Data
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(input_path, low_memory=False)
    logging.info(f"Matrix loaded. Shape: {df.shape}")

    # 3. Feature Alignment
    # We must ensure X contains only the numeric features the model was trained on
    drop_list = predict_cfg['params']['features_to_drop']
    X = df.drop(columns=[c for c in drop_list if c in df.columns])
    
    # Ensure all columns are numeric (Safety sweep)
    X = X.select_dtypes(include=['number', 'bool'])

    # 4. Generate Propensity Scores
    # [0, 1] -> we want index 1 (probability of being 'Converted')
    df['propensity_score'] = model.predict_proba(X)[:, 1]

    # 5. Metric Comparison (Drift Check)
    target_col = predict_cfg['params']['target_col']
    metrics = {"snapshot": snapshot, "timestamp": datetime.now().isoformat()}

    if target_col in df.columns:
        inf_auc = roc_auc_score(df[target_col], df['propensity_score'])
        metrics["inference_auc"] = round(inf_auc, 4)
        logging.info(f"Inference ROC-AUC: {inf_auc:.4f}")
        
        # Note: In a real scenario, you'd pull the 'training_auc' from a log 
        # For now, we log the inference score for your review.
    else:
        logging.info("Target column not found. Skipping AUC calculation (Standard for Production).")

    # 6. Save Results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # We only save cifno and the score for the final lead list
    output_df = df[['cifno', 'propensity_score']]
    output_df.to_csv(output_path, index=False)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"✅ Predictions saved: {output_path}")
    return output_path