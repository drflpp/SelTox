from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from V4_MIC.Models import V4_transform_MIC_trial
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

MIC_data = pd.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\data\preprocessed\preprocessed_MIC_original.csv')
MIC_df = MIC_data.drop(['Unnamed: 0','np','reference',], axis=1)


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

X_train, X_test, Y_train, Y_test = train_test_split(X_train_enc, Y_train_enc, test_size=0.2, random_state=42)
model = CatBoostRegressor()
model.fit(X_train, Y_train)
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Evaluation metrics
train_r2 = r2_score(Y_train, train_predictions)
train_mae = mean_absolute_error(Y_train, train_predictions)
train_mse = mean_squared_error(Y_train, train_predictions)**0.5  # RMSE

test_r2 = r2_score(Y_test, test_predictions)
test_mae = mean_absolute_error(Y_test, test_predictions)
test_mse = mean_squared_error(Y_test, test_predictions)**0.5  # RMSE

# Print metrics
print('Train')
print('Train R-square:', train_r2)
print('Mean Absolute Error:', train_mae)
print('Root Mean Squared Error:', train_mse)

print('Test')
print('Test R-square:', test_r2)
print('Mean Absolute Error:', test_mae)
print('Root Mean Squared Error:', test_mse)

'''visualization'''
import matplotlib.pyplot as plt
import seaborn as sns

# Set custom parameters for the plot
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)

# Create a figure and axis
f, ax = plt.subplots(figsize=(13, 10))

# Scatter plot for train and validation data
plt.scatter(Y_train, train_predictions, color='#2d4d85', s=50, label='Train Data', alpha=0.7)
plt.scatter(Y_test, test_predictions, color='#951d6d', s=50, label='Validation', alpha=0.7)

# Plotting the diagonal line (perfect prediction)
plt.plot(Y_train, Y_train, color='#444444', linewidth=2)

# Set labels and title
plt.title('XGB Regressor - Predicted vs Actual')
plt.xlabel('Actual Data')
plt.ylabel('Predicted Data')
plt.legend()
plt.savefig('default_xgb.png', transparent=True)
# Show the plot
plt.show()

