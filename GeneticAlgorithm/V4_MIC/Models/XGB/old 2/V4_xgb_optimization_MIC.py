import pandas as pd
import numpy as np
import optuna
from V4_MIC.Models import V4_transform_MIC_trial
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

MIC_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\preprocessed\preprocessed_MIC_mod.csv')
MIC_df = MIC_data.drop(['Unnamed: 0','reference',], axis=1)

MIC_train_df = MIC_data[MIC_data['source'] == 0].drop(['source'], axis=1)
MIC_test_df = MIC_data[MIC_data['source'] > 0].drop(['source'], axis=1)
MIC_train_Y = MIC_train_df[['concentration']].copy()
MIC_test_Y = MIC_test_df[['concentration']].copy()

MIC_train_df = MIC_train_df.reset_index(drop=True)
MIC_test_df = MIC_test_df.reset_index(drop=True)
MIC_train_Y = MIC_train_Y.reset_index(drop=True)
MIC_test_Y = MIC_test_Y.reset_index(drop=True)

X_train_enc = V4_transform_MIC_trial.transform(MIC_train_df)
X_test_enc = V4_transform_MIC_trial.transform(MIC_test_df)
Y_train_enc = MIC_train_Y
Y_test_enc = MIC_test_Y

X = X_train_enc
Y = Y_train_enc

def objective(trial, x, y):
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 50, 2000, step=50),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 20.0)
    }
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)  # changing the n_split to 10 reduces the r2 score
    cv_scores = np.empty(10)
    for idx, (train_indices, test_indices) in enumerate(cv.split(X_train_enc, X_train_enc[['mol_weight (g/mol)']])):
        x_train, x_test = X_train_enc.iloc[train_indices], X_train_enc.iloc[test_indices]
        y_train, y_test = Y_train_enc.iloc[train_indices], Y_train_enc.iloc[test_indices]
    #
    # cv = KFold(n_splits=10, shuffle=True, random_state=42)
    #
    # cv_scores = np.empty(10)

    # for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
    #     x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    #     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = XGBRegressor(**param)
        model.fit(x_train, y_train)
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        print('training_r2', r2_score(y_train, pred_train))
        print('mean_sq_error', mean_squared_error(y_train, pred_train))
        cv_scores[idx] = r2_score(y_test, pred_test)
        print('r2 score', idx, cv_scores)

    return np.mean(cv_scores)
study = optuna.create_study(direction="maximize", study_name="XGB_Regressor")
func = lambda trial: objective(trial, X, Y)
study.optimize(func, n_trials=100)

print(f"\tBest value (r2 score): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")