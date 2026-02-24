import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from V4_MIC.Models import V4_transform_MIC_trial
from sklearn.model_selection import train_test_split

processed_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\preprocessed\preprocessed_MIC_original.csv')
# print(processed_data.columns)
X = processed_data.drop(['Unnamed: 0','source','concentration','reference',], axis=1)
print(X.columns)
Y = processed_data[['concentration']].copy()

Xt = V4_transform_MIC_trial.first_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(Xt, Y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=3,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
train.to_csv('Model_comparision_processed_MIC.csv')
