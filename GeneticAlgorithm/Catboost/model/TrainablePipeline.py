import joblib
import numpy as np
import shap
import polars as pl
import pandas as pd
from sklearn.model_selection import KFold
from matplotlib.colors import to_hex
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from .BasePipeline import BasicPipeline, clean_feature_names
cols = []


def get_model_name(model):
    return model.__class__.__name__
class TrainablePipeline1(BasicPipeline):
    def __init__(self, df=None, models=None, dataset_name=None, **kwargs):

        super().__init__(
            df=df if df is not None else None,
            models=models if models is not None else [],
            dataset_name=dataset_name,
            **kwargs
        )

        self.model = None
        self.feature_names = None
        self.cat_cols = None
        self.num_cols = None
        self.dataset_name = dataset_name

    def _fit_transform_X(self, X):
        X = X.to_pandas().copy()

        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        self.encoders = {}

        for col in self.cat_cols:
            categories = sorted(X[col].astype(str).unique())
            oe = OrdinalEncoder(
                categories=[categories],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            X[col] = oe.fit_transform(X[[col]].astype(str)).ravel()
            self.encoders[col] = oe

        self.scaler = StandardScaler()
        X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])

        return X


    def _transform_X(self, X):
        X = X.copy()

        # categorical
        for col in self.cat_cols:
            X[col] = self.encoders[col].transform(
                X[[col]].astype(str)
            ).ravel()

        X[self.num_cols] = self.scaler.transform(X[self.num_cols])

        return X


    def plot_top_features_gradient1(
            self,
            top_n: int = 40,
            fname: str | None = None,
            figsize = (8, 8)
    ):


        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib as mpl
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.colors import to_hex


        if self.model is None:
            raise RuntimeError("Model is not trained or loaded")

        if not hasattr(self.model, "feature_importances_"):
            raise TypeError(
                f"{type(self.model).__name__} does not support feature_importances_"
            )

        if self.feature_names is None:
            raise RuntimeError("Feature names are not available")


        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["DejaVu Serif"]
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42


        importances = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importances))

        top_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        names = [x[0] for x in top_features]
        scores = np.array([x[1] for x in top_features])

        plotly_purple = LinearSegmentedColormap.from_list(
            "plotly_purple",
            list(reversed([
                to_hex((249 / 255, 221 / 255, 218 / 255)),
                to_hex((242 / 255, 185 / 255, 196 / 255)),
                to_hex((229 / 255, 151 / 255, 185 / 255)),
                to_hex((206 / 255, 120 / 255, 179 / 255)),
                to_hex((173 / 255, 95 / 255, 173 / 255)),
                to_hex((131 / 255, 75 / 255, 160 / 255)),
                to_hex((87 / 255, 59 / 255, 136 / 255)),
            ]))
        )

        colors = plotly_purple(np.linspace(0, 1, len(names)))


        plt.figure(figsize=figsize)
        plt.barh(
            names,
            scores,
            color=colors,
            edgecolor="black",
            linewidth=0.4
        )

        plt.xlabel("Feature importance", fontsize=12)
        plt.title(
            f"{type(self.model).__name__} – Top {len(names)} features",
            fontsize=14,
            pad=12
        )

        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.3)

        plt.tight_layout()

        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_shap(self, max_display=30):
        import shap

        if self.model is None:
            raise RuntimeError("Model is not trained")


        X_train = self.X_train.to_pandas() if hasattr(self.X_train, "to_pandas") else self.X_train
        X_test = self.X_test.to_pandas() if hasattr(self.X_test, "to_pandas") else self.X_test

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=self.feature_names,
            max_display=max_display
        )

    def cross_validate(
            self,
            model_cls,
            model_params: dict,
            test_df: pd.DataFrame,
            n_splits: int = 10,
            shuffle: bool = True,
            random_state: int = 42
    ):

        self.clean_data()
        reduced_df = self.reduce_dimensions()
        reduced_df.columns = clean_feature_names(reduced_df.columns)

        train_pd = reduced_df.to_pandas()
        cols_part = train_pd.columns.to_list()

        X_raw = train_pd.drop('MIC_NP___g_mL_', axis=1)
        y = train_pd['MIC_NP___g_mL_'].values


        test_pd = test_df.copy()
        test_pd = test_pd[cols_part]
        test_pd.columns = clean_feature_names(test_pd.columns)
        test_pd = test_pd

        X_test_raw = test_pd.drop('MIC_NP___g_mL_', axis=1)
        y_test = test_pd['MIC_NP___g_mL_'].values

        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )


        metrics = {
            "train_r2": [],
            "val_r2": [],
            "test_r2": [],
            "train_rmse": [],
            "val_rmse": [],
            "test_rmse": [],
            "test_mae": []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw), 1):
            print(f"[CV] Fold {fold}/{n_splits}")

            X_train_raw = X_raw.iloc[train_idx]
            X_val_raw = X_raw.iloc[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]


            X_train = self._fit_transform_X(pl.from_pandas(X_train_raw))
            X_val = self._transform_X(X_val_raw)
            X_test = self._transform_X(X_test_raw)


            model = model_cls(**model_params)
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)

            metrics["train_r2"].append(r2_score(y_train, train_preds))
            metrics["val_r2"].append(r2_score(y_val, val_preds))
            metrics["test_r2"].append(r2_score(y_test, test_preds))

            metrics["train_rmse"].append(root_mean_squared_error(y_train, train_preds))
            metrics["val_rmse"].append(root_mean_squared_error(y_val, val_preds))
            metrics["test_rmse"].append(root_mean_squared_error(y_test, test_preds))
            metrics["test_mae"].append(mean_absolute_error(y_test, test_preds))


        log_row = {
            "model": model_cls.__name__,
            "dataset": self.dataset_name,
            "n_splits": n_splits,
            "hyperparams": model_params,

            "train_r2_mean": np.mean(metrics["train_r2"]),
            "val_r2_mean": np.mean(metrics["val_r2"]),
            "test_r2_mean": np.mean(metrics["test_r2"]),

            "train_rmse_mean": np.mean(metrics["train_rmse"]),
            "val_rmse_mean": np.mean(metrics["val_rmse"]),
            "test_rmse_mean": np.mean(metrics["test_rmse"]),
            "test_mae_mean": np.mean(metrics["test_mae"]),

            "train_samples": len(train_pd),
            "val_samples": len(train_pd) // n_splits,
            "test_samples": len(test_pd),
            "n_features": X_train.shape[1]
        }

        log_df = pd.DataFrame([log_row])

        print("\n[CV RUN SUMMARY]")
        print(log_df.T.round(4))

        return log_df


    def fit(self, model):
        self.clean_data()
        reduced_df = self.reduce_dimensions()

        reduced_df.columns = clean_feature_names(reduced_df.columns)
        # reduced_df = reduced_df.reset_index(drop=True)

        X = reduced_df.drop('MIC_NP___g_mL_')
        y = reduced_df['MIC_NP___g_mL_'].to_pandas().values

        X = self._fit_transform_X(X)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        self.feature_names = self.X_train.columns.tolist()

        self.model = model
        self.model.fit(self.X_train, self.Y_train)

        self.feature_names = self.X_train.columns.tolist()

        preds = self.model.predict(self.X_test)
        r2 = r2_score(self.Y_test, preds)
        rmse = root_mean_squared_error(self.Y_test, preds)

        print(f"[FIT] {get_model_name(model)} | R2={r2:.4f} | RMSE={rmse:.4f}")
        return r2, rmse

    def fit1(self, model, fname='fit1.png'):
        import matplotlib.pyplot as plt

        self.clean_data()
        reduced_df = self.reduce_dimensions()

        reduced_df.columns = clean_feature_names(reduced_df.columns)


        X = reduced_df.drop('MIC_NP___g_mL_')
        y = reduced_df['MIC_NP___g_mL_'].to_pandas().values

        X = self._fit_transform_X(X)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        self.feature_names = self.X_train.columns.tolist()

        self.model = model
        self.model.fit(self.X_train, self.Y_train)


        train_preds = self.model.predict(self.X_train)
        test_preds = self.model.predict(self.X_test)

        r2_train = r2_score(self.Y_train, train_preds)
        rmse_train = root_mean_squared_error(self.Y_train, train_preds)

        r2_test = r2_score(self.Y_test, test_preds)
        rmse_test = root_mean_squared_error(self.Y_test, test_preds)

        print(f"[FIT] {get_model_name(model)} | Train R2={r2_train:.4f} | Train RMSE={rmse_train:.4f}")
        print(f"[FIT] {get_model_name(model)} | Test R2={r2_test:.4f} | Test RMSE={rmse_test:.4f}")

        plt.figure(figsize=(6, 6))
        plt.scatter(self.Y_train, train_preds, label="Train", color=to_hex((87 / 255, 59 / 255, 136 / 255)), alpha=1, s=15)
        plt.scatter(self.Y_test, test_preds, label="Test", color=to_hex((173 / 255, 95 / 255, 173 / 255)), alpha=1, s=15)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
        plt.xlabel("Actual log concentration")
        plt.ylabel("Predicted log concentration")
        plt.title(f"{get_model_name(self.model)} model's predictions")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()

        return r2_test, rmse_test

    def predict(self, df):
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded")

        df = df.clone()
        df.columns = clean_feature_names(df.columns)
        if 'MIC_NP___g_mL_' in df.columns:
            X = df.to_pandas().drop('MIC_NP___g_mL_', axis=1)
        else:
            X = df.to_pandas()
        X = self._transform_X(X)
        X = X.reindex(columns=self.feature_names, fill_value=0)  # fill_value для новых/отсутствующих фич
        X = X.astype(np.float32)


        return self.model.predict(X)

    def predict1(self, df, show_plot=True, test_set_name='Test'):
        import matplotlib.pyplot as plt

        if self.model is None:
            raise RuntimeError("Model is not trained or loaded")

        df = df.clone()
        df.columns = clean_feature_names(df.columns)
        if 'MIC_NP___g_mL_' in df.columns:
            X = df.to_pandas().drop('MIC_NP___g_mL_', axis=1)
            y_true = df.to_pandas()['MIC_NP___g_mL_'].values
            has_true = True
        else:
            X = df.to_pandas()
            y_true = None
            has_true = False

        X = self._transform_X(X)
        X = X.reindex(columns=self.feature_names, fill_value=0)  # fill_value для новых/отсутствующих фич
        X = X.astype(np.float32)

        preds = self.model.predict(X)


        if show_plot and has_true:
            plt.figure(figsize=(6, 6))

            if hasattr(self, 'Y_train') and hasattr(self, 'Y_test'):
                plt.scatter(self.Y_train, self.model.predict(self.X_train),
                            label="Train", color=to_hex((87 / 255, 59 / 255, 136 / 255)), alpha=0.2, s=15)
                plt.plot([self.Y_train.min(), self.Y_train.max()], [self.Y_train.min(), self.Y_train.max()], 'k--', lw=1)
                plt.scatter(self.Y_test, self.model.predict(self.X_test),
                            label="Validation", color=to_hex((173 / 255, 95 / 255, 173 / 255)), alpha=0.2, s=15)

            plt.scatter(y_true, preds, label=test_set_name, color='red', alpha=1.0, s=15)
            plt.xlabel("Actual log concentration")
            plt.ylabel("Predicted log concentration")
            plt.title(f"{get_model_name(self.model)} model's predictions")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{test_set_name}.png',  transparent=True, dpi=300,    bbox_inches="tight")
            plt.show()

        return preds


    def save(self, path):
        joblib.dump({
            "model": self.model,
            "encoders": self.encoders,
            "scaler": self.scaler,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "feature_names": self.feature_names,
            "dataset_name": self.dataset_name,
            "model_params": self.model.get_params(),
            "n_features": len(self.feature_names)

        }, path)

        print(f"[SAVED] {path}")


    def load(self, path):
        payload = joblib.load(path)

        self.model = payload["model"]
        self.encoders = payload["encoders"]
        self.scaler = payload["scaler"]
        self.cat_cols = payload["cat_cols"]
        self.num_cols = payload["num_cols"]
        self.feature_names = payload["feature_names"]
        self.dataset_name = payload["dataset_name"]

        print(f"[LOADED] {path}")


