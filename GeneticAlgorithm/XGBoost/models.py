import pandas as pd
import pickle
import warnings
import polars as pl
import joblib
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from model.TrainablePipeline import TrainablePipeline1
warnings.filterwarnings('ignore')

x3 = {'n_estimators': 1400, 'learning_rate': 0.09300039576787837, 'max_depth': 4, 'min_child_weight': 0.2531466898550219, 'max_leaves': 177, 'gamma': 0.026592255537413102, 'reg_alpha': 0.0028243435758436423, 'reg_lambda': 2.4729681422233027, 'subsample': 0.9397657727939068, 'colsample_bytree': 0.9163614175818913, 'grow_policy': 'depthwise', 'random_state':42, 'n_jobs':1}


pipeline = TrainablePipeline1(dataset_name='df1_MIC_xgb', df=pl.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC_XGB\data\preprocessed\final_df1_catboost_orig.csv').drop(''))
pipeline.fit(XGBRegressor(**x3))

