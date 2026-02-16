import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

results_file = "best_params_2.py"  # File to save hyperparameters

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def optimization(df_x_scaled, df_y, dataset_name="dataset"):
    def objective(trial, df_x, df_y):
        params = {
      
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
        
     
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 0.3),
        

        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 0.3, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        

        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        
  
        'tree_method': 'hist',           
        'grow_policy': 'depthwise',   
        'objective': 'reg:squarederror',
        

        'random_state': 42,
        'n_jobs': 1                  
    }
        model = XGBRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, test_idx in cv.split(df_x, df_y):
            x_train, x_test = df_x.iloc[train_idx], df_x.iloc[test_idx]
            y_train, y_test = df_y.iloc[train_idx], df_y.iloc[test_idx]
            model.fit(x_train, y_train)
            pred_test = model.predict(x_test)
            cv_scores.append(r2_score(y_test, pred_test))

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df_x_scaled, df_y), n_trials=100)

    best_params = study.best_params
    best_rmse = study.best_value

    print(f"\nOptimization complete for {dataset_name}")
    print(f"Best RMSE: {best_rmse:.5f}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Save best params to file
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("# --- XGBoost best hyperparameter results ---\n\n")

    # Create a unique variable name using dataset_name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    var_name = f"best_params_{dataset_name}_{timestamp}"

    with open(results_file, "a") as f:
        f.write(f"{var_name} = {best_params}\n\n")

    print(f"Saved hyperparameters to {results_file} as {var_name}")
    return best_params


final_df = pd.read_csv("df2_optimization.csv", index_col=0)
X_opt = final_df.drop("MIC_NP___g_mL_", axis=1)
Y_opt = final_df["MIC_NP___g_mL_"]

optimization(X_opt, Y_opt, dataset_name="df1")
optimization(X_opt, Y_opt, dataset_name="df1")
optimization(X_opt, Y_opt, dataset_name="df1")

optimization(X_opt, Y_opt, dataset_name="df1")
optimization(X_opt, Y_opt, dataset_name="df1")
optimization(X_opt, Y_opt, dataset_name="df1")
