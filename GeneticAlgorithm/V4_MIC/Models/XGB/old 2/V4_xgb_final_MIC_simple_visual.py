import pandas as pd
import numpy as np
from V4_MIC.Models import V4_transform_MIC_trial
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

best_hyperparameters = {
    'max_depth': 8,
    'learning_rate': 0.020284140784294108,
    'n_estimators': 996,
    'min_child_weight': 2,
    'gamma': 0.08212914149351197,
    'subsample': 0.6617647490743902,
    'colsample_bytree': 0.847302653540521,
    'reg_lambda': 0.649338114238663,
    'reg_alpha': 0.5921286695333474,

}

X_train, X_test, Y_train, Y_test = train_test_split(X_train_enc, Y_train_enc, test_size=0.2, random_state=42)
model = XGBRegressor(**best_hyperparameters)
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
from sklearn import metrics

# Set custom parameters for the plot
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)

# Create a figure and axis
f, ax = plt.subplots(figsize=(10, 10))

# Scatter plot for train and validation data
plt.scatter(Y_train, train_predictions, color='#2d4d85', s=50, label='Train Data', alpha=1)
plt.scatter(Y_test, test_predictions, color='#951d6d', s=50, label='Validation', alpha=1)
plt.plot(Y_test, (Y_test - 2*np.sqrt(metrics.mean_squared_error(Y_test, test_predictions))), color='#444444', linewidth = 1)
plt.plot(Y_test, (Y_test + 2*np.sqrt(metrics.mean_squared_error(Y_test, test_predictions))), color='#444444', linewidth = 1)
# Plotting the diagonal line (perfect prediction)
plt.plot(Y_train, Y_train, color='#444444', linewidth=2)

# Change the color and thickness of x and y axis
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['left'].set_linewidth(3)

# Set labels and title
plt.title('XGB Regressor')
plt.xlabel('Actual Data')
plt.ylabel('Predicted Data')
plt.legend()
plt.savefig('Xgb_optimized_error.png', transparent=True)
# Show the plot
plt.show()

