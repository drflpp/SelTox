import pandas as pd
import numpy as np
import optuna
from V3_MIC.Models import transform_MIC_trial
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

MIC_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V3_MIC\data\preprocessed\Final_combined_MIC.csv')
MIC_data = MIC_data.drop(['reference'], axis=1)

MIC_train_df = MIC_data[MIC_data['source'] == 0].drop(['source'], axis=1)
MIC_test_df = MIC_data[MIC_data['source'] > 0].drop(['source'], axis=1)
MIC_train_Y = MIC_train_df[['concentration']].copy()
MIC_test_Y = MIC_test_df[['concentration']].copy()

MIC_train_df = MIC_train_df.reset_index(drop=True)
MIC_test_df = MIC_test_df.reset_index(drop=True)
MIC_train_Y = MIC_train_Y.reset_index(drop=True)
MIC_test_Y = MIC_test_Y.reset_index(drop=True)

X_train_enc = transform_MIC_trial.transform(MIC_train_df)
X_test_enc = transform_MIC_trial.transform(MIC_test_df)
Y_train_enc = np.log10(MIC_train_Y)

X = X_train_enc
Y = Y_train_enc


def objective(trial, x, y):
    param = {
        'depth': trial.suggest_int('depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=1),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 8),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0)
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = np.empty(10)

    for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = CatBoostRegressor(**param)
        model.fit(x_train, y_train, silent=True)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        cv_scores[idx] = r2_score(y_test, pred_test)

    return np.mean(cv_scores)


# Assuming X, Y are already defined as X_train_enc and Y_train_enc
study = optuna.create_study(direction="maximize", study_name="CatBoost_Regressor")
func = lambda trial: objective(trial, X, Y)
study.optimize(func, n_trials=100)

print(f"\tBest value (r2 score): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")
