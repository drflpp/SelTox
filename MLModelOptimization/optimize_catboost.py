import pandas as pd
import numpy as np
import optuna
import os
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import warnings
from catboost import CatBoostRegressor
import os
from datetime import datetime

warnings.filterwarnings('ignore')

results_file = "best_params_cat.py"  # File to save hyperparameters

dataset_name = 'df1'

def optimization(df_x_scaled, df_y):
    def objective(trial, df_x, df_y):

        # PARAMS CatBoost my best

        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])

        params = {
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength": trial.suggest_float("random_strength", 0.5, 3.0),
            "rsm": trial.suggest_float("rsm", 0.6, 1.0),
            "bootstrap_type": bootstrap_type,
            "grow_policy": trial.suggest_categorical("grow_policy", ["Depthwise", "Lossguide"]),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "verbose": 0,
            "random_state": 42
        }
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 1)
        # Do NOT add subsample
        else:  # Bernoulli
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        # Do NOT add bagging_temperature

        model = CatBoostRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        # Cross-validation loop
        for train_indices, test_indices in cv.split(df_x, df_y):
            x_train, x_test = df_x.iloc[train_indices], df_x.iloc[test_indices]
            y_train, y_test = df_y.iloc[train_indices], df_y.iloc[test_indices]

            model.fit(x_train, y_train)

            pred_test = model.predict(x_test)
            # cv_scores.append(root_mean_squared_error(y_test, pred_test))
            cv_scores.append(r2_score(y_test, pred_test))

        return np.mean(cv_scores)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df_x_scaled, df_y), n_trials=100)

    best_params = study.best_params
    best_rmse = study.best_value

    print(f"\tBest value (rmse): {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")

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

optimization(X_opt, Y_opt)
optimization(X_opt, Y_opt)
optimization(X_opt, Y_opt)

optimization(X_opt, Y_opt)
optimization(X_opt, Y_opt)
optimization(X_opt, Y_opt)

