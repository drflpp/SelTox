import pandas as pd
import numpy as np
from V4_MIC.Models import V4_transform_MIC_trial
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
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

train_R2_metric_results = []
train_mse_metric_results= []
train_mae_metric_results = []
Validation_R2_metric_results = []
Validation_mse_metric_results= []
Validation_mae_metric_results = []

test_R2_metric_results = []
test_mse_metric_results = []
test_mae_metric_results = []

val_R2_metric_results = []
val_mse_metric_results = []
val_mae_metric_results = []

cv = KFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = np.empty(10)
for idx, (train_indices, test_indices) in enumerate(cv.split(X_train_enc, Y_train_enc)):
    x_train, x_test = X_train_enc.iloc[train_indices], X_train_enc.iloc[test_indices]
    y_train, y_test = Y_train_enc.iloc[train_indices], Y_train_enc.iloc[test_indices]
    model = XGBRegressor()

    lgb_model = model.fit(x_train, y_train)
    train = model.predict(x_train)
    validation = model.predict(x_test)
    test = model.predict(X_test_enc)

    train_R2_metric_results.append(r2_score(y_train, train))
    train_mse_metric_results.append(mean_squared_error(y_train, train))
    train_mae_metric_results.append(mean_absolute_error(y_train, train))

    test_R2_metric_results.append(r2_score(y_test, validation))
    test_mse_metric_results.append(mean_squared_error(y_test, validation))
    test_mae_metric_results.append(mean_absolute_error(y_test, validation))

    val_R2_metric_results.append(r2_score(Y_test_enc, test))
    val_mse_metric_results.append(mean_squared_error(Y_test_enc, test))
    val_mae_metric_results.append(mean_absolute_error(Y_test_enc, test))

print('Train')
print('Train R-square:', np.mean(train_R2_metric_results))
print('Mean Absolute Error:', np.mean(train_mae_metric_results))
print('Mean Squared Error:', (np.mean(train_mse_metric_results)))
print('Root Mean Squared Error:', (np.mean(train_mse_metric_results)**(1/2)))

print('validation')
print('one-out cross-validation (R-square):', r2_score(y_test, validation))
print('10-fold cross-validation result (R-square):', np.mean(test_R2_metric_results))
print('10-fold cross-validation result (MAE):', np.mean(test_mae_metric_results))
print('10-fold cross-validation result (MSE):', (np.mean(test_mse_metric_results)))
print('10-fold cross-validation result (RMSE):', (np.mean(test_mse_metric_results)**(1/2)))


print('testing1')
print('one-out cross-validation (R-square):', r2_score(y_test, validation))
print('10-fold cross-validation result (R-square):', np.mean(val_R2_metric_results))
print('10-fold cross-validation result (MAE):', np.mean(val_mae_metric_results))
print('10-fold cross-validation result (MSE):', (np.mean(val_mse_metric_results)))
print('10-fold cross-validation result (RMSE):', (np.mean(val_mse_metric_results)**(1/2)))

# import pickle
# with open('xgb_model_MIC_final.pkl', 'wb') as file:
#     pickle.dump(model, file)

"""#Visualization"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)

f, ax = plt.subplots(figsize=(12, 10))

plt.scatter(y_train, train, color='#2d4d85', s=50, label='train data', alpha=0.3)
plt.scatter(y_test, validation, color='#951d6d', s=50, label='validation', alpha=0.3)
plt.scatter(Y_test_enc, test, color='#f62d2d', s=50, label='test', alpha=1)
# plt.plot(Y_test, (y_test - 2*np.sqrt(metrics.mean_squared_error(y_test, validation))), color='#444444', linewidth = 1)
# plt.plot(Y_test, (y_test + 2*np.sqrt(metrics.mean_squared_error(y_test, validation))), color='#444444', linewidth = 1)
plt.plot(y_train, y_train, color='#444444', linewidth=2)

plt.title('XGB Regressor')
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
# plt.xlim(0, 45)
# plt.ylim(0, 45)

# Save the figure with transparency
# plt.savefig('model_xgb_mic_optimized_validation.png', transparent=True)
# plt.show()

# Getting feature importance from the model
feature_importance = model.feature_importances_
feature_names = x_train.columns

# Getting feature importance from the model
feature_importance = model.feature_importances_
feature_names = x_train.columns

# Creating a DataFrame to organize feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Define a custom color palette as a gradient from red to blue
custom_palette = sns.color_palette("RdBu", n_colors=len(feature_importance_df))

# # Plotting feature importance with custom color gradient
# plt.figure(figsize=(10, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette=custom_palette)
# plt.title('XGB Feature Importance')
# plt.xlabel('Importance')
# plt.ylabel('Features')
# plt.tight_layout()
#
# # Save the plot
# plt.savefig('xgb_feature_importance_optimized.png', transparent=True)
#
# # Show the plot if needed
# # plt.show()

import shap
import matplotlib.pyplot as plt

# Assuming you have X_test and model defined

X_importance = x_train
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)
shap.summary_plot(shap_values, X_importance, show=False)
plt.tight_layout()  # Ensures plots are properly arranged

# Save the plot before showing it
# plt.savefig('important_features_xgb_MIC.png', transparent = True)

# Show the plot if needed
# plt.show()
