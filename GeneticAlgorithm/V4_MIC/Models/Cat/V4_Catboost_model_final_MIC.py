import pandas as pd
import numpy as np
from sklearn import metrics

from V4_MIC.Models import V4_transform_MIC_trial
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

MIC_data = pd.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\data\preprocessed\preprocessed_MIC_mod.csv')
# MIC_data=MIC_data[MIC_data['concentration']>0]
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

cat_best_hyperparameters = {
    'depth': 5,
    'learning_rate': 0.1397280623494613,
    'n_estimators': 855,
    'min_child_samples': 6,
    'border_count': 75,
    'subsample': 0.7895495648303331,
    'colsample_bylevel': 0.5135233479163253,
    'l2_leaf_reg': 8.96719105386514,
    'random_strength': 4.455968402164433,
}

best_hyperparameters = {
    'depth': 4,
    'learning_rate': 0.2627967247501425,
    'n_estimators': 702,
    'min_child_samples': 1,
    'border_count': 19,
    'subsample': 0.6049698537585045,
    'colsample_bylevel': 0.9793852005634773,
    'l2_leaf_reg': 8.752032888414995,
    'random_strength': 5.7408966964866766,
}

best_hyperparameters_mod = {
    'depth': 5,
    'learning_rate': 0.23630161689686982,
    'n_estimators': 774,
    'min_child_samples': 6,
    'border_count': 178,
    'subsample': 0.7205585920561297,
    'colsample_bylevel': 0.9246560514791413,
    'l2_leaf_reg': 5.567508702983153,
    'random_strength': 0.4857958992981014,
}


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

print(X_train_enc.columns)
cv = StratifiedShuffleSplit(n_splits=10, test_size = 0.2, random_state = 42) # changing the n_split to 10 reduces the r2 score
cv_scores = np.empty(10)
for idx, (train_indices, test_indices) in enumerate(cv.split(X_train_enc, X_train_enc[['mol_weight (g/mol)']])):
    X_train, X_test = X_train_enc.iloc[train_indices], X_train_enc.iloc[test_indices]
    Y_train, Y_test = Y_train_enc.iloc[train_indices], Y_train_enc.iloc[test_indices]

#
# cv = KFold(n_splits=10, shuffle=True, random_state=42)
# cv_scores = np.empty(10)
# for idx, (train_indices, test_indices) in enumerate(cv.split(X_train_enc, Y_train_enc)):
#     X_train, X_test = X_train_enc.iloc[train_indices], X_train_enc.iloc[test_indices]
#     Y_train, Y_test = Y_train_enc.iloc[train_indices], Y_train_enc.iloc[test_indices]
    model = CatBoostRegressor(**best_hyperparameters_mod)

    lgb_model = model.fit(X_train, Y_train)
    train = model.predict(X_train)
    validation = model.predict(X_test)
    test = model.predict(X_test_enc)

    train_R2_metric_results.append(r2_score(Y_train, train))
    train_mse_metric_results.append(mean_squared_error(Y_train, train))
    train_mae_metric_results.append(mean_absolute_error(Y_train, train))

    test_R2_metric_results.append(r2_score(Y_test, validation))
    test_mse_metric_results.append(mean_squared_error(Y_test, validation))
    test_mae_metric_results.append(mean_absolute_error(Y_test, validation))

    val_R2_metric_results.append(r2_score(Y_test_enc, test))
    val_mse_metric_results.append(mean_squared_error(Y_test_enc, test))
    val_mae_metric_results.append(mean_absolute_error(Y_test_enc, test))

print('Train')
print('Train R-square:', np.mean(train_R2_metric_results))
print('Mean Absolute Error:', np.mean(train_mae_metric_results))
print('Mean Squared Error:', (np.mean(train_mse_metric_results)))
print('Root Mean Squared Error:', (np.mean(train_mse_metric_results)**(1/2)))

print('validation')
print('one-out cross-validation (R-square):', r2_score(Y_test, validation))
print('10-fold cross-validation result (R-square):', np.mean(test_R2_metric_results))
print('10-fold cross-validation result (MAE):', np.mean(test_mae_metric_results))
print('10-fold cross-validation result (MSE):', (np.mean(test_mse_metric_results)))
print('10-fold cross-validation result (RMSE):', (np.mean(test_mse_metric_results)**(1/2)))


print('testing')
print('one-out cross-validation (R-square):', r2_score(Y_test, validation))
print('10-fold cross-validation result (R-square):', np.mean(val_R2_metric_results))
print('10-fold cross-validation result (MAE):', np.mean(val_mae_metric_results))
print('10-fold cross-validation result (MSE):', (np.mean(val_mse_metric_results)))
print('10-fold cross-validation result (RMSE):', (np.mean(val_mse_metric_results)**(1/2)))


'''save validation set with predicted value'''
MIC_test_df['predicted'] = test
MIC_test_df.to_csv('validation_set_with_predicted_results.csv')

# import pickle
# with open('Catboost_model_MIC_final.pkl', 'wb') as file:
#     pickle.dump(model, file)

"""#Visualization"""
import seaborn as sns
import shap
import matplotlib.pyplot as plt


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)

f, ax = plt.subplots(figsize=(10, 10))
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams.update({'font.size': 12})
plt.scatter(Y_train, train, color='#2d4d85', s=50, label='train data', alpha=1)
plt.scatter(Y_test, validation, color='#951d6d', s=50, label='validation', alpha=1)
# plt.scatter(Y_test_enc, test, color='#f62d2d', s=50, label='test', alpha=1)
# plt.plot(Y_test, (Y_test - 2*np.sqrt(metrics.mean_squared_error(Y_test, validation))), color='#444444', linewidth = 1)
# plt.plot(Y_test, (Y_test + 2*np.sqrt(metrics.mean_squared_error(Y_test, validation))), color='#444444', linewidth = 1)
plt.plot(Y_train, Y_train, color='#444444', linewidth=2)

# Change the color and thickness of x and y axis
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['left'].set_linewidth(3)

plt.title('CatBoost Regressor')
plt.xlabel('Actual data')
plt.ylabel('Predicted data')
# plt.legend()
plt.xlim(-2, 4.5)
plt.ylim(-2, 4.5)

# Save the figure with transparency
plt.savefig('model_cat_MIC_optimized_MIC.png', transparent=True)
plt.show()


'''Feature Importance'''
feature_importance = model.feature_importances_
feature_names = X_train.columns
cols = ['mol_weight (g/mol)', 'np_size_avg (nm)', 'time_set', 'Valance_electron',
       'labuteASA',]

# Creating a DataFrame to organize feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Define a custom color palette as a gradient from red to blue
custom_palette = sns.color_palette("Blues_r", n_colors=len(feature_importance_df))

# Filter feature importance for selected columns
selected_feature_importance = feature_importance_df[feature_importance_df['Feature'].isin(cols)]

# Plotting feature importance with custom color gradient for selected columns

plt.figure(figsize=(8, 8))
sns.barplot(x='Importance', y='Feature', data=selected_feature_importance, palette=custom_palette)

plt.title('CatBoost Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()

# Save the plot
plt.savefig('cat_feature_importance_MIC_optimized_5.png', transparent=True)
plt.show()


'''SHAP value plot'''
X_importance = X_train
selected_X_importance = X_importance[cols]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)
# Filter SHAP values for selected columns
selected_shap_values = shap_values[:, X_importance.columns.isin(cols)]

# Create the SHAP summary plot with selected columns
# shap.summary_plot(shap_values, X_importance, show=False)
shap.summary_plot(selected_shap_values, selected_X_importance, show=False)
plt.gcf().set_size_inches(8, 8)
plt.tight_layout()  # Ensures plots are properly arranged

plt.savefig('important_features_cat_MIC_optimized.png', transparent = True)
plt.show()
