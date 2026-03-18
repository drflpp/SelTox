import re
import time
import random
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
plt.rcParams['figure.dpi'] = 100

results_df = pd.DataFrame(columns=[
    'dataset_name',
    'dataset_type',
    'model_name',
    'data_shape',
    'r2_score',
    'rmse',
    'top_features'
])

cv_logs_df = pd.DataFrame(columns=[
    "timestamp",
    "dataset_name",
    "dataset_type",
    "model_name",
    "n_splits",
    "stratified",
    "hyperparameters",
    "train_R2",
    "train_RMSE",
    "val_R2",
    "val_RMSE",
    "test_R2",
    "test_RMSE",
    "train_MAE",
    "val_MAE",
    "test_MAE",
])


def clean_feature_names(feature_names):
    clean_names = []
    for name in feature_names:
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        clean_names.append(clean_name)
    return clean_names


def get_model_name(model):
    return model.__class__.__name__


def log_result(dataset_name, dataset_type, model, data_shape, r2, rmse, feature_names=None):
    global results_df
    model_name = get_model_name(model)

    top_features = None
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:10]
        top_features = [feature_names[i] for i in indices]

    new_row = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'model_name': model_name,
        'data_shape': f"{data_shape[0]}x{data_shape[1]}",
        'r2_score': r2,
        'rmse': rmse,
        'top_features': top_features
    }

    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)


def df_fit_transformer(df: pl.DataFrame):
    oe_dict = {}
    df_copy = df.clone()

    cat_cols = df_copy.select(pl.col(pl.String)).columns
    for col in cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        col_data = df_copy.select(col).to_numpy()
        oe.fit(col_data)
        transformed = oe.transform(col_data)
        df_copy = df_copy.with_columns(pl.Series(name=col, values=transformed.flatten()))
        oe_dict[col] = oe

    num_types = [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ]
    num_cols = df_copy.select(pl.col(num_types)).columns
    scaler = StandardScaler()
    num_data = df_copy.select(num_cols).to_numpy()
    scaler.fit(num_data)
    scaled = scaler.transform(num_data)
    scaled_df = pl.DataFrame(scaled, schema=num_cols)
    df_copy = df_copy.drop(num_cols).hstack(scaled_df)

    return df_copy, oe_dict, scaler


def reduce_dimensionality_fast(df: pl.DataFrame, var_thresh=1e-5, corr_thresh=0.95):
    num_types = [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ]
    num_cols = df.select(pl.col(num_types)).columns
    if not num_cols:
        return df

    num_df = df.select(num_cols)
    num_np = num_df.to_numpy()

    selector = VarianceThreshold(threshold=var_thresh)
    num_np_var = selector.fit_transform(num_np)
    support = selector.get_support()
    selected_cols_var = [num_cols[i] for i in range(len(num_cols)) if support[i]]
    num_df = pl.DataFrame(num_np_var, schema=selected_cols_var)

    corr_matrix = np.corrcoef(num_np_var, rowvar=False)
    to_drop = set()
    for i in range(len(selected_cols_var)):
        if selected_cols_var[i] in to_drop:
            continue
        for j in range(i + 1, len(selected_cols_var)):
            if selected_cols_var[j] in to_drop:
                continue
            if abs(corr_matrix[i, j]) > corr_thresh:
                to_drop.add(selected_cols_var[j])

    reduced_df = num_df.drop(list(to_drop))

    non_num_cols = [c for c in df.columns if c not in num_cols]
    non_num_df = df.select(non_num_cols)
    final_df = reduced_df.hstack(non_num_df)
    return final_df


class BasePipeline:
    def __init__(self, df, dataset_name, models, cutoff_FI=0.95, cutoff_PI=0.95, strat=None,
                 strat_feature=None, cv_type='skfold', n_splits=5):
        self.df = df.clone()
        self.dataset_name = dataset_name
        self.models = models
        self.cutoff_FI = cutoff_FI
        self.cutoff_PI = cutoff_PI
        self.strat = strat
        self.random_state = 42
        self.strat_feature = None
        self.cv_type = cv_type
        self.n_splits = n_splits

        np.random.seed(self.random_state)
        random.seed(self.random_state)

        self.df_cleaned = None
        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None
        self.encoders = None
        self.scaler = None

    def clean_data(self):
        df = self.df.drop(cols)
        df = df.unique(maintain_order=True)
        df = df.drop(['Strain', 'strain'])
        self.df_cleaned = df
        print(f"Cleaned data shape: {df.shape}")

    def reduce_dimensions(self):
        reduced_df = reduce_dimensionality_fast(self.df_cleaned.clone())
        print(f"Reduced dimensions: {reduced_df.shape}")
        return reduced_df

    def preprocess(self, df):
        df.columns = clean_feature_names(df.columns)
        X = df.drop('MIC_NP___g_mL_')
        Y = df.select('MIC_NP___g_mL_').to_series()
        X_transformed, encoders, scaler = df_fit_transformer(X)
        self.encoders = encoders
        self.scaler = scaler
        return X_transformed.to_pandas(), Y.to_pandas()

    def split_data(self, X, Y):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=self.random_state
        )
        print("Data split complete.")

    def run_model_pipeline(self):
        for model in self.models:
            print(f"Running model: {get_model_name(model)}")
            self.train_and_log(model)

    def train_and_log(self, model):
        model.fit(self.X_train, self.Y_train)
        preds = model.predict(self.X_test)
        r2 = r2_score(self.Y_test, preds)
        rmse = root_mean_squared_error(self.Y_test, preds)
        full = pd.concat([self.X_train, self.X_test])
        feature_names = full.columns
        log_result(self.dataset_name, 'preprocessed', model, full.shape, r2, rmse, feature_names)
        print(f"Initial Model {get_model_name(model)} | R2: {r2:.4f}, RMSE: {rmse:.4f}")
        self.plot_top_features(model, self.X_train.columns)

    def plot_top_features(self, model, feature_names, top_n=10):
        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names, scores = zip(*top_features)

        plt.figure(figsize=(10, 6))
        plt.barh(names, scores, color='skyblue')
        plt.xlabel("Feature Importance")
        plt.title(f'{get_model_name(model)} Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.savefig(f'{self.dataset_name}_{get_model_name(model)}.jpg', bbox_inches='tight')
        plt.show()

    def run(self):
        start = time.time()
        self.clean_data()
        reduced_df = self.reduce_dimensions()
        X, Y = self.preprocess(reduced_df)
        self.split_data(X, Y)
        self.run_model_pipeline()
        print(f"Pipeline completed in {(time.time() - start):.2f} seconds.")


class PipelineGetDF(BasePipeline):
    def run(self):
        start = time.time()
        self.clean_data()
        reduced_df = self.reduce_dimensions()
        return reduced_df


class PipelineFIPI(BasePipeline):

    def run_model_pipeline(self):
        for model in self.models:
            print(f"Running model: {get_model_name(model)}")
            self.train_and_log(model)
            self.feature_importance_selection(model)
            self.permutation_importance_selection(model)

    def feature_importance_selection(self, model):
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            print(f"Model {get_model_name(model)} has no feature_importances_. Skipping FI step.")
            return

        df_imp = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        df_imp['cumulative'] = df_imp['importance'].cumsum() / df_imp['importance'].sum()

        selected = df_imp[df_imp['cumulative'] <= self.cutoff_FI]['feature'].tolist()
        if len(selected) < len(df_imp):
            selected.append(df_imp.iloc[len(selected)]['feature'])

        self.X_train = self.X_train[selected]
        self.X_test = self.X_test[selected]
        full = pd.concat([self.X_train, self.X_test])
        model.fit(self.X_train, self.Y_train)
        preds = model.predict(self.X_test)
        r2 = r2_score(self.Y_test, preds)
        rmse = root_mean_squared_error(self.Y_test, preds)
        print(f"After FI {get_model_name(model)} | R2: {r2:.4f}, RMSE: {rmse:.4f}")
        log_result(
            self.dataset_name,
            f'after keeping {self.cutoff_FI} FI',
            model,
            full.shape,
            r2,
            rmse,
            self.X_train.columns
        )
        print(f"Feature Importance selection done. Remaining features: {len(selected)}")
        self.plot_top_features(model, self.X_train.columns)

    def permutation_importance_selection(self, model):
        print("Calculating permutation importance...")
        result = permutation_importance(
            model, self.X_test, self.Y_test, n_repeats=10, random_state=self.random_state
        )

        df_perm = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance_mean': result.importances_mean
        }).sort_values(by='importance_mean', ascending=False)

        df_perm = df_perm[df_perm['importance_mean'] > 0].dropna()
        df_perm['cumulative_importance'] = (
            df_perm['importance_mean'].cumsum() / df_perm['importance_mean'].sum()
        )

        selected = df_perm[df_perm['cumulative_importance'] <= self.cutoff_PI]['feature'].tolist()
        if len(selected) < len(df_perm):
            selected.append(df_perm.iloc[len(selected)]['feature'])

        self.X_train = self.X_train[selected]
        self.X_test = self.X_test[selected]
        full = pd.concat([self.X_train, self.X_test])
        model.fit(self.X_train, self.Y_train)
        preds = model.predict(self.X_test)
        r2 = r2_score(self.Y_test, preds)
        rmse = root_mean_squared_error(self.Y_test, preds)
        print(f"After PI {get_model_name(model)} | R2: {r2:.4f}, RMSE: {rmse:.4f}")
        log_result(
            self.dataset_name,
            f'after keeping {self.cutoff_FI} FI + {self.cutoff_PI} PI',
            model,
            full.shape,
            r2,
            rmse,
            self.X_train.columns
        )
        print(f"Permutation Importance selection done. Remaining features: {len(selected)}")
        self.plot_top_features(model, self.X_train.columns)

    def run(self):
        start = time.time()
        self.clean_data()
        reduced_df = self.reduce_dimensions()
        X, Y = self.preprocess(reduced_df)
        self.split_data(X, Y)
        self.run_model_pipeline()
        final_df = pd.concat([
            pd.concat([self.X_train, self.Y_train], axis=1),
            pd.concat([self.X_test, self.Y_test], axis=1)
        ], axis=0).reset_index(drop=True)
        print(f"Pipeline completed in {(time.time() - start):.2f} seconds.")
        print(f"Final DF shape after PI: {final_df.shape}")
        return final_df


class BasePipelineCV:
    def __init__(self, df, dataset_name, models, cutoff_FI=0.95, cutoff_PI=0.95,
                 strat=None, strat_feature=None, cv_type='skfold', n_splits=5):
        self.df = df.clone()
        self.dataset_name = dataset_name
        self.models = models
        self.cutoff_FI = cutoff_FI
        self.cutoff_PI = cutoff_PI
        self.strat = strat
        self.random_state = 42
        self.strat_feature = strat_feature
        self.cv_type = cv_type
        self.n_splits = n_splits

        np.random.seed(self.random_state)
        random.seed(self.random_state)

        self.df_cleaned = None
        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None
        self.encoders = None
        self.scaler = None

    def clean_data(self):
        df = self.df.drop(cols)
        df = df.unique(maintain_order=True)
        df = df.drop(['Strain', 'strain'])
        self.df_cleaned = df
        print(f"Cleaned data shape: {df.shape}")

    def reduce_dimensions(self):
        reduced_df = reduce_dimensionality_fast(self.df_cleaned.clone())
        print(f"Reduced dimensions: {reduced_df.shape}")
        return reduced_df

    def preprocess(self, df):
        df.columns = clean_feature_names(df.columns)
        X = df.drop('MIC_NP___g_mL_')
        Y = df.select('MIC_NP___g_mL_').to_series()
        X_transformed, encoders, scaler = df_fit_transformer(X)
        self.encoders = encoders
        self.scaler = scaler
        return X_transformed.to_pandas(), Y.to_pandas()

    def split_data(self, X, Y):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=self.random_state
        )
        print("Data split complete.")

    def cross_validate(self, model, X_train, Y_train, X_test, Y_test):
        print(f"\nStarting cross-validation for {get_model_name(model)} ...")

        if self.cv_type == 'skfold' and self.strat_feature is not None:
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = cv.split(X_train, X_train[self.strat_feature])
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = cv.split(X_train, Y_train)

        metrics = {'train_r2': [], 'val_r2': [], 'test_r2': [],
                   'train_rmse': [], 'val_rmse': [], 'test_rmse': [],
                   'train_mae': [], 'val_mae': [], 'test_mae': []}

        for i, (train_idx, val_idx) in enumerate(splits):
            print(f"Fold {i + 1}/{self.n_splits}")
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)

            y_pred_train = model.predict(X_tr)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)

            metrics['train_r2'].append(r2_score(y_tr, y_pred_train))
            metrics['val_r2'].append(r2_score(y_val, y_pred_val))
            metrics['test_r2'].append(r2_score(Y_test, y_pred_test))

            metrics['train_rmse'].append(np.sqrt(mean_squared_error(y_tr, y_pred_train)))
            metrics['val_rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            metrics['test_rmse'].append(np.sqrt(mean_squared_error(Y_test, y_pred_test)))

            metrics['train_mae'].append(mean_absolute_error(y_tr, y_pred_train))
            metrics['val_mae'].append(mean_absolute_error(y_val, y_pred_val))
            metrics['test_mae'].append(mean_absolute_error(Y_test, y_pred_test))

        self.log_cv_results(model, metrics, pd.concat([X_train, X_test]).shape)
        return metrics

    def log_cv_results(self, model, metrics, dataset_shape):
        global cv_logs_df

        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": self.dataset_name,
            "dataset_shape": str(dataset_shape),
            "model_name": get_model_name(model),
            "n_splits": self.n_splits,
            "cv_type": self.cv_type,
            "train_R2": np.mean(metrics['train_r2']),
            "val_R2": np.mean(metrics['val_r2']),
            "test_R2": np.mean(metrics['test_r2']),
            "train_RMSE": np.mean(metrics['train_rmse']),
            "val_RMSE": np.mean(metrics['val_rmse']),
            "test_RMSE": np.mean(metrics['test_rmse']),
            "train_MAE": np.mean(metrics['train_mae']),
            "val_MAE": np.mean(metrics['val_mae']),
            "test_MAE": np.mean(metrics['test_mae']),
        }

        cv_logs_df = pd.concat([cv_logs_df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Logged CV results for {get_model_name(model)}")

    def run_model_pipeline(self):
        for model in self.models:
            print(f"\nRunning model: {get_model_name(model)}")
            self.cross_validate(model, self.X_train, self.Y_train, self.X_test, self.Y_test)

    def run(self):
        start = time.time()
        self.clean_data()
        reduced_df = self.reduce_dimensions()
        X, Y = self.preprocess(reduced_df)
        self.split_data(X, Y)
        self.run_model_pipeline()
        print(f"\nPipeline completed in {(time.time() - start):.2f} seconds.")


class PipelineFIPICV(BasePipelineCV):

    def cross_validate(self, model, X_train, Y_train, X_test, Y_test):
        print(f"\nStarting cross-validation + FI/PI for {get_model_name(model)} ...")

        if self.cv_type == 'skfold' and self.strat_feature is not None:
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = cv.split(X_train, X_train[self.strat_feature])
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = cv.split(X_train, Y_train)

        metrics = {'train_r2': [], 'val_r2': [], 'test_r2': [],
                   'train_rmse': [], 'val_rmse': [], 'test_rmse': [],
                   'train_mae': [], 'val_mae': [], 'test_mae': []}

        for i, (train_idx, val_idx) in enumerate(splits):
            print(f"\nFold {i + 1}/{self.n_splits}")
            X_tr, X_val = X_train.iloc[train_idx].copy(), X_train.iloc[val_idx].copy()
            y_tr, y_val = Y_train.iloc[train_idx].copy(), Y_train.iloc[val_idx].copy()

            model.fit(X_tr, y_tr)
            self._log_fold_metrics(model, X_tr, y_tr, X_val, y_val, X_test, Y_test, metrics, stage="initial")

            X_tr_fi, X_val_fi, X_test_fi = self.feature_importance_selection(model, X_tr, X_val, X_test)
            model.fit(X_tr_fi, y_tr)
            self._log_fold_metrics(model, X_tr_fi, y_tr, X_val_fi, y_val, X_test_fi, Y_test, metrics, stage="FI")

            X_tr_pi, X_val_pi, X_test_pi = self.permutation_importance_selection(model, X_tr_fi, X_val_fi, X_test_fi, y_tr, y_val)
            model.fit(X_tr_pi, y_tr)
            self._log_fold_metrics(model, X_tr_pi, y_tr, X_val_pi, y_val, X_test_pi, Y_test, metrics, stage="FI+PI")

        self.log_cv_results(model, metrics, X_train.shape)
        return metrics

    def _log_fold_metrics(self, model, X_tr, y_tr, X_val, y_val, X_test, Y_test, metrics, stage):
        train_pred = model.predict(X_tr)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        metrics['train_r2'].append(r2_score(y_tr, train_pred))
        metrics['val_r2'].append(r2_score(y_val, val_pred))
        metrics['test_r2'].append(r2_score(Y_test, test_pred))

        metrics['train_rmse'].append(np.sqrt(mean_squared_error(y_tr, train_pred)))
        metrics['val_rmse'].append(np.sqrt(mean_squared_error(y_val, val_pred)))
        metrics['test_rmse'].append(np.sqrt(mean_squared_error(Y_test, test_pred)))

        metrics['train_mae'].append(mean_absolute_error(y_tr, train_pred))
        metrics['val_mae'].append(mean_absolute_error(y_val, val_pred))
        metrics['test_mae'].append(mean_absolute_error(Y_test, test_pred))

        print(f"{stage} | R²(val): {metrics['val_r2'][-1]:.4f} | RMSE(val): {metrics['val_rmse'][-1]:.4f}")

    def feature_importance_selection(self, model, X_tr, X_val, X_test):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            'feature': X_tr.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        df_imp['cumulative'] = df_imp['importance'].cumsum()

        selected = df_imp[df_imp['cumulative'] <= self.cutoff_FI]['feature'].tolist()
        if len(selected) < len(df_imp):
            selected.append(df_imp.iloc[len(selected)]['feature'])

        print(f"FI selected {len(selected)}/{len(df_imp)} features")
        return X_tr[selected], X_val[selected], X_test[selected]

    def permutation_importance_selection(self, model, X_tr, X_val, X_test, y_tr, y_val):
        result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=self.random_state)
        df_perm = pd.DataFrame({
            'feature': X_val.columns,
            'importance_mean': result.importances_mean
        }).sort_values(by='importance_mean', ascending=False)

        df_perm = df_perm[df_perm['importance_mean'] > 0].dropna()
        df_perm['cumulative_importance'] = df_perm['importance_mean'].cumsum()
        df_perm['cumulative_importance'] /= df_perm['importance_mean'].sum()

        selected = df_perm[df_perm['cumulative_importance'] <= self.cutoff_PI]['feature'].tolist()
        if len(selected) < len(df_perm):
            selected.append(df_perm.iloc[len(selected)]['feature'])

        print(f"PI selected {len(selected)}/{len(df_perm)} features")
        return X_tr[selected], X_val[selected], X_test[selected]


class PipelineCV1(BasePipeline):
    def __init__(self, df, dataset_name, best_hyperparameters, strat_feature=None,
                 n_splits=10, stratified=False, model_class=CatBoostRegressor, save_model_path=None):
        super().__init__(df, dataset_name, models=None, n_splits=n_splits)
        self.best_hyperparameters = best_hyperparameters
        self.strat_feature = strat_feature
        self.stratified = stratified
        self.model_class = model_class
        self.save_model_path = save_model_path
        self.trained_model = None
        self.dataset_shape = None

        self.train_metrics = {'r2': [], 'mse': [], 'mae': []}
        self.val_metrics = {'r2': [], 'mse': [], 'mae': []}
        self.test_metrics = {'r2': [], 'mse': [], 'mae': []}

    def cross_validate(self, X_train_enc, Y_train_enc, X_test_enc, Y_test_enc):
        print("Starting Cross-Validation...")

        if self.stratified and self.strat_feature is not None:
            cv = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=42)
            split_gen = cv.split(X_train_enc, X_train_enc[[self.strat_feature]])
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            split_gen = cv.split(X_train_enc, Y_train_enc)

        for idx, (train_idx, val_idx) in enumerate(split_gen):
            print(f"\nFold {idx + 1}/{self.n_splits}")
            x_train, x_val = X_train_enc.iloc[train_idx], X_train_enc.iloc[val_idx]
            y_train, y_val = Y_train_enc.iloc[train_idx], Y_train_enc.iloc[val_idx]

            model = self.model_class(**self.best_hyperparameters)
            model.fit(x_train, y_train)

            train_pred = model.predict(x_train)
            val_pred = model.predict(x_val)
            test_pred = model.predict(X_test_enc)

            self.train_metrics['r2'].append(r2_score(y_train, train_pred))
            self.train_metrics['mse'].append(mean_squared_error(y_train, train_pred))
            self.train_metrics['mae'].append(mean_absolute_error(y_train, train_pred))

            self.val_metrics['r2'].append(r2_score(y_val, val_pred))
            self.val_metrics['mse'].append(mean_squared_error(y_val, val_pred))
            self.val_metrics['mae'].append(mean_absolute_error(y_val, val_pred))

            self.test_metrics['r2'].append(r2_score(Y_test_enc, test_pred))
            self.test_metrics['mse'].append(mean_squared_error(Y_test_enc, test_pred))
            self.test_metrics['mae'].append(mean_absolute_error(Y_test_enc, test_pred))

            self.trained_model = model

        print("Cross-Validation complete.")

    def summarize_results(self):
        def summarize(name, metrics):
            print(f"\n{name.upper()} RESULTS")
            print(f"R²: {np.mean(metrics['r2']):.4f}")
            print(f"MAE: {np.mean(metrics['mae']):.4f}")
            print(f"RMSE: {np.sqrt(np.mean(metrics['mse'])):.4f}")

        summarize("train", self.train_metrics)
        summarize("validation", self.val_metrics)
        summarize("test", self.test_metrics)

    def log_run(self):
        global cv_logs_df

        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": self.dataset_name,
            "dataset_type": "encoded",
            "model_name": self.model_class.__name__,
            "n_splits": self.n_splits,
            "stratified": self.stratified,
            "hyperparameters": str(self.best_hyperparameters),
            "train_R2": np.mean(self.train_metrics['r2']),
            "train_RMSE": np.sqrt(np.mean(self.train_metrics['mse'])),
            "val_R2": np.mean(self.val_metrics['r2']),
            "val_RMSE": np.sqrt(np.mean(self.val_metrics['mse'])),
            "test_R2": np.mean(self.test_metrics['r2']),
            "test_RMSE": np.sqrt(np.mean(self.test_metrics['mse'])),
            "train_MAE": np.mean(self.train_metrics['mae']),
            "val_MAE": np.mean(self.val_metrics['mae']),
            "test_MAE": np.mean(self.test_metrics['mae']),
        }

        cv_logs_df = pd.concat([cv_logs_df, pd.DataFrame([new_row])], ignore_index=True)
        cv_logs_df.to_csv("cv_run_logs.csv", index=False)
        print("CV run logged successfully to cv_run_logs.csv")

    def run(self):
        start = time.time()
        print("\nCleaning and preprocessing data...")
        dataset_shape = self.df.shape
        print("dataset_shape: ", dataset_shape)
        X_df, Y_df = self.preprocess(self.df)
        print(f'X_df_shape: {X_df.shape}')
        self.split_data(X_df, Y_df)
        print("\nRunning Cross-Validation...")
        self.cross_validate(self.X_train, self.Y_train, self.X_test, self.Y_test)
        self.summarize_results()
        self.log_run()
        print(f"\nPipeline completed in {(time.time() - start):.2f} seconds.")


if __name__ == '__main__':
    cols = ['', 'reference', 'doi', 'Unnamed: 0', 'CID', 'np', 'Canonical_smiles', 'np_synthesis']

    df0 = pl.read_parquet('final_df0.parquet')
    df1 = pl.read_parquet('final_df1.parquet')
    df2 = pl.read_parquet('final_df2.parquet')

    models = [CatBoostRegressor(random_state=42, verbose=False)]

    pipeline0 = PipelineFIPI(df0, dataset_name='df0', models=models, cutoff_FI=0.95, cutoff_PI=0.99)
    pipeline1 = PipelineFIPI(df1, dataset_name='df1', models=models, cutoff_FI=0.95, cutoff_PI=0.97)
    pipeline2 = PipelineFIPI(df2, dataset_name='df2', models=models, cutoff_FI=0.95, cutoff_PI=0.99)

    final_df0 = pipeline0.run()
    final_df1 = pipeline1.run()
    final_df2 = pipeline2.run()

    final_df0.to_csv('final_df0_selected.csv', index=False)
    final_df1.to_csv('final_df1_selected.csv', index=False)
    final_df2.to_csv('final_df2_selected.csv', index=False)

    results_df.to_csv('feature_selection_results.csv', index=False)
