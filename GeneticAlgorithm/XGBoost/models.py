import pandas as pd
import pickle
import warnings
import polars as pl
import joblib
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from Models.TrainablePipeline import TrainablePipeline1
warnings.filterwarnings('ignore')

Cat_model_path = r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\Models\CatBoost_v2\model_cat_p17.joblib'
x3 = {'n_estimators': 1400, 'learning_rate': 0.09300039576787837, 'max_depth': 4, 'min_child_weight': 0.2531466898550219, 'max_leaves': 177, 'gamma': 0.026592255537413102, 'reg_alpha': 0.0028243435758436423, 'reg_lambda': 2.4729681422233027, 'subsample': 0.9397657727939068, 'colsample_bytree': 0.9163614175818913, 'grow_policy': 'depthwise', 'random_state':42, 'n_jobs':1}


# pipeline = TrainablePipeline1(dataset_name="MIC_dataset", df=pl.DataFrame(),
#     models=[])
# pipeline.load(Cat_model_path)
# pipeline_xgb = TrainablePipeline1(dataset_name='df1_MIC_xgb', df=pl.DataFrame(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC_XGB\data\preprocessed\final_df1_catboost_orig.csv'))
# pipeline_xgb.fit(XGBRegressor(**x3))
pipeline = TrainablePipeline1(dataset_name='df1_MIC_xgb', df=pl.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC_XGB\data\preprocessed\final_df1_catboost_orig.csv').drop(''))
pipeline.fit(XGBRegressor(**x3))

# mod = joblib.load(Cat_model_path)

'''
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
'''
# def model_load(path):
#     with open(path, 'rb') as f:
#         cat_model = pickle.load(f)
#         return cat_model
# mod = model_load(Cat_model_path)

def cat_predict(input_data):
    expected_features = ['np_size_max__nm_',
                         'np_synthesis',
                         'np_size_min__nm_',
                         'np_size_avg__nm_',
                         'method',
                         'Temperature_for_extract__C',
                         'Duration_preparing_extract__min',
                         'Solvent_for_extract',
                         'time_set__hours_',
                         'shape',
                         'coating',
                         'Valance_electron',
                         'prim_specific_habitat',
                         'chi0v',
                         'amw',
                         'min_Incub_period__h',
                         'K00058',
                         'max_Incub_period__h',
                         'K12472',
                         'mdr',
                         'K07009',
                         'K17939',
                         'K14198',
                         'K07486']


    x = input_data
    x = x[expected_features]
    predict = mod.predict(x)
    return predict
