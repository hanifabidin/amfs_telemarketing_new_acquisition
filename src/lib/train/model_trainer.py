# amfs_tm/src/lib/train/model_trainer.py
import os
import yaml
import pickle
import pandas as pd
import logging
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def train_model(main_config_path, train_config_path, model_config_path, args):
    """
    Trains the model using pre-compiled files from the prep script.
    """
    with open(main_config_path, 'r') as f: main_cfg = yaml.safe_load(f)
    with open(train_config_path, 'r') as f: train_cfg = yaml.safe_load(f)
    with open(model_config_path, 'r') as f: model_cfg = yaml.safe_load(f)

    root = main_cfg['PATHS']['base_path']
    mode = train_cfg['settings']['mode']
    target_col = train_cfg['target_logic']['target_col']

    # Load compiled train data
    train_path = os.path.join(root, train_cfg['paths']['train_output'].format(mode=mode))
    df_train = pd.read_csv(train_path)
    
    # In dummy mode, we might split this one file. 
    # In real mode, we load the separate test file.
    if mode == "real":
        test_path = os.path.join(root, train_cfg['paths']['test_output'].format(mode=mode))
        df_test = pd.read_csv(test_path)
    else:
        # Split dummy data for testing
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)

    # Features split
    X_train = df_train.drop(columns=[target_col, 'cifno'], errors='ignore')
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col, 'cifno'], errors='ignore')
    y_test = df_test[target_col]

    # Train
    model = xgb.XGBClassifier(**model_cfg['model_params'])
    model.fit(X_train, y_train)

    # Eval
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    logging.info(f"Model ({mode}) ROC-AUC: {auc:.4f}")

    # Save Pickle
    model_dir = os.path.join(root, '05_modelling', 'model_file')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{mode}_{args.snapshot}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model_path