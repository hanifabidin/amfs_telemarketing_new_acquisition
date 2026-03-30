# amfs_tm/src/lib/train/model_trainer.py
import os
import yaml
import pickle
import pandas as pd
import logging
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def train_model(main_config_path, train_config_path, model_config_path, args):
    """
    Trains an XGBoost propensity model. 
    Includes a fail-safe to drop non-numeric columns.
    """
    with open(main_config_path, 'r') as f: main_cfg = yaml.safe_load(f)
    with open(train_config_path, 'r') as f: train_cfg = yaml.safe_load(f)
    with open(model_config_path, 'r') as f: model_cfg = yaml.safe_load(f)

    root = main_cfg['PATHS']['base_path']
    mode = args.mode if hasattr(args, 'mode') else train_cfg['settings']['mode']
    target_col = train_cfg['target_logic']['target_col']
    snapshot = args.snapshot

    # 1. Load Data
    train_path = os.path.join(root, train_cfg['paths']['train_output'].format(mode=mode, snapshot=snapshot))
    df_train_raw = pd.read_csv(train_path, low_memory=False)

    if mode == "real":
        test_path = os.path.join(root, train_cfg['paths']['test_output'].format(mode=mode, snapshot=snapshot))
        df_test_raw = pd.read_csv(test_path, low_memory=False)
    else:
        df_train_raw, df_test_raw = train_test_split(
            df_train_raw, test_size=model_cfg['training_settings']['test_size'], random_state=42
        )

    # 2. Initial Feature Selection (Drop IDs/Target)
    drop_cols = model_cfg['training_settings'].get('features_to_drop', [])
    X_train = df_train_raw.drop(columns=[c for c in drop_cols if c in df_train_raw.columns])
    X_test = df_test_raw.drop(columns=[c for c in drop_cols if c in df_test_raw.columns])
    
    y_train = df_train_raw[target_col]
    y_test = df_test_raw[target_col]

    # --- 3. FAIL-SAFE: Automatic Numeric Filtering ---
    # XGBoost only accepts int, float, or bool. We must exclude 'object' (strings).
    non_numeric_cols = X_train.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    
    if non_numeric_cols:
        logging.warning(f"⚠️ Dropping non-numeric columns that escaped preprocessing: {non_numeric_cols}")
        X_train = X_train.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=non_numeric_cols)

    # 4. Model Training
    logging.info(f"Training XGBoost with {X_train.shape[1]} features...")
    model = xgb.XGBClassifier(**model_cfg['model_params'])
    model.fit(X_train, y_train)

    # 5. Evaluation & Save
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    logging.info(f"🏆 Model Training Success! ROC-AUC Score: {auc_score:.4f}")

    model_dir = os.path.join(root, '04_modelling', 'model_file')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_propensity_{mode}_{snapshot}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path