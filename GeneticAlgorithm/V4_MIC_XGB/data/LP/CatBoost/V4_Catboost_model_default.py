import pandas as pd
import numpy as np
from V4_MIC.Models import V4_transform_MIC_trial
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

raw = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\raw\merged_df.csv')
raw = raw.reset_index(drop=True)
raw_X = raw.drop(['concentration'], axis=1)
print(raw_X.columns)
raw_Y = raw[['concentration']].copy()
rX = V4_transform_MIC_trial.first_transform(raw_X)

processed_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\preprocessed\preprocessed_MIC_original.csv')
X = processed_data.drop(['Unnamed: 0','source','concentration','reference',], axis=1)
Y = processed_data[['concentration']].copy()
Xt = V4_transform_MIC_trial.first_transform(X)


start_time =time.time()
# X_train, X_test, Y_train, Y_test = train_test_split(rX, raw_Y, test_size=0.2, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(Xt, Y, test_size=0.2, random_state = 42)

model = CatBoostRegressor()
model.fit(X_train, Y_train)
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
end_time = time.time()
model_training_duration = end_time - start_time
# Evaluation metrics
train_r2 = r2_score(Y_train, train_predictions)
train_mae = mean_absolute_error(Y_train, train_predictions)
train_mse = mean_squared_error(Y_train, train_predictions)**0.5  # RMSE

test_r2 = r2_score(Y_test, test_predictions)
test_mae = mean_absolute_error(Y_test, test_predictions)
test_mse = mean_squared_error(Y_test, test_predictions)**0.5  # RMSE

# Calculate adjusted R-squared for train and test sets
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

# Number of samples
n_train = len(Y_train)
n_test = len(Y_test)

# Number of features in the model
p = X_train.shape[1]

# Calculate R-squared for train and test sets
train_adj_r2 = adjusted_r2(train_r2, n_train, p)
test_adj_r2 = adjusted_r2(test_r2, n_test, p)

# Print adjusted R-squared scores
print('Train Adjusted R-square:', train_adj_r2)
print('Test Adjusted R-square:', test_adj_r2)
print('Time required:', model_training_duration)


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
plt.title('Random Forest Regressor - Predicted vs Actual')
plt.xlabel('Actual Data')
plt.ylabel('Predicted Data')
plt.legend()

# Show the plot
plt.show()
