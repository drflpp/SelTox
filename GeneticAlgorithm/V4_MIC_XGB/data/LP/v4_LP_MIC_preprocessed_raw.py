import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from V4_MIC.Models import V4_transform_MIC_trial
from sklearn.model_selection import train_test_split


raw = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\raw\merged_df.csv')
raw = raw.reset_index(drop=True)
raw_X = raw.drop(['concentration'], axis=1)
raw_Y = raw[['concentration']].copy()

rX = V4_transform_MIC_trial.first_transform(raw_X)
# rY = np.log10(raw_Y)

X_train, X_test, Y_train, Y_test = train_test_split(rX, raw_Y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=3,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
train.to_csv('Model_comparision_raw_MIC_not_log.csv')
