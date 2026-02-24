import pickle
import pandas as pd
import numpy as np
from V4_MIC.Models import V4_transform_MIC_trial
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

MIC_data = pd.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\data\preprocessed\preprocessed_MIC_mod.csv')
# MIC_data=MIC_data[MIC_data['concentration']>0]
MIC_df = MIC_data.drop(['Unnamed: 0','np','reference',], axis=1)

# df = MIC_data[MIC_data['source'].isin([1, 6])]
df = MIC_data[MIC_data['source'] >0]
df = df.reset_index(drop=True)
print(df)
df_enc = V4_transform_MIC_trial.transform(df)

Cat_model_path = r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\Models\CatBoost_v2\catboost_model_df1.pkl'

def model_load(path):
    with open(path, 'rb') as f:
        cat_model = pickle.load(f)
        return cat_model
mod = model_load(Cat_model_path)

def cat_predict(input_data):
    x = input_data
    predict = mod.predict(x)
    return predict

predicted_df_y = cat_predict(df_enc)
print(predicted_df_y)

df['predicted'] = predicted_df_y
df.to_csv('validaiton.csv')