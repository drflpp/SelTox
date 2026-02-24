import pandas as pd
import numpy as np
from V3_MIC.Models import transform_MIC_trial
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# MIC_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V3_MIC\data\preprocessed\Final_combined_MIC.csv')
# MIC_data_add =pd.read_csv(r'C:\Users\user\Desktop\Valya\V3_MIC\data\validation\final_validation_set2_synergy.csv')
# MIC_data = pd.concat([MIC_data, MIC_data_add], ignore_index=True)
# MIC_data.to_csv('Additional_final_MIC.csv')
raw = pd.read_csv(r'C:\Users\user\Desktop\Valya\V3_MIC\data\raw\final_merged_df_v2.csv')
raw =raw[raw['concentration'] > 0]
MIC_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V3_MIC\data\preprocessed\Final_combined_MIC.csv')
MIC_df = MIC_data.drop(['reference'], axis=1)

MIC_train_df = MIC_df[MIC_data['source'] == 0].drop(['source'], axis=1)
MIC_test_df = MIC_df[MIC_data['source'] >0].drop(['source'], axis=1)
# MIC_test_df = MIC_test_df[MIC_test_df['concentration'] > 10 ]
# MIC_test_ecoli = MIC_test_df[MIC_test_df['bacteria'] == 'Staphylococcus aureus'] # 'Staphylococcus aureus' 'Enterococcus faecalis' 'Escherichia coli' 'Pseudomonas aeruginosa'
# MIC_test_df = MIC_test_ecoli
# MIC_test_ZnO = MIC_test_df[MIC_test_df['np'] == 'Au'] #Au, ZnO,
# MIC_test_df = MIC_test_ZnO
raw_X = raw.drop(['concentration'], axis=1)
raw_Y = raw[['concentration']].copy()
MIC_train_Y = MIC_train_df[['concentration']].copy()
MIC_test_Y = MIC_test_df[['concentration']].copy()

MIC_train_df = MIC_train_df.reset_index(drop=True)
MIC_test_df = MIC_test_df.reset_index(drop=True)
MIC_train_Y = MIC_train_Y.reset_index(drop=True)
MIC_test_Y = MIC_test_Y.reset_index(drop=True)

# print('MIC_train_df', MIC_train_df)
X_train_enc = transform_MIC_trial.first_transform(raw_X)
Y_train_enc = np.log10(raw_Y)
# X_train_enc = transform_MIC_trial.transform(MIC_train_df)
X_test_enc = transform_MIC_trial.transform(MIC_test_df)
# Y_train_enc = np.log10(MIC_train_Y)
Y_test_enc = np.log10(MIC_test_Y)

cat_best_hyperparameters = {
    'depth': 4,
    'learning_rate': 0.1810725583490497,
    'n_estimators': 987,
    'min_child_samples': 8,
    'border_count': 91,
    'subsample': 0.7369296095155401,
    'colsample_bylevel': 0.9961237686452229,
    'l2_leaf_reg': 6.973740346425227,
    'random_strength': 0.8282290435602956,

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


cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = np.empty(10)
for idx, (train_indices, test_indices) in enumerate(cv.split(X_train_enc, Y_train_enc)):
    X_train, X_test = X_train_enc.iloc[train_indices], X_train_enc.iloc[test_indices]
    Y_train, Y_test = Y_train_enc.iloc[train_indices], Y_train_enc.iloc[test_indices]
    model = CatBoostRegressor()

    lgb_model = model.fit(X_train, Y_train)
    train = model.predict(X_train)
    validation = model.predict(X_test)
    # test = model.predict(X_test_enc)

    train_R2_metric_results.append(r2_score(Y_train, train))
    train_mse_metric_results.append(mean_squared_error(Y_train, train))
    train_mae_metric_results.append(mean_absolute_error(Y_train, train))

    test_R2_metric_results.append(r2_score(Y_test, validation))
    test_mse_metric_results.append(mean_squared_error(Y_test, validation))
    test_mae_metric_results.append(mean_absolute_error(Y_test, validation))

    # val_R2_metric_results.append(r2_score(Y_test_enc, test))
    # val_mse_metric_results.append(mean_squared_error(Y_test_enc, test))
    # val_mae_metric_results.append(mean_absolute_error(Y_test_enc, test))

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

# import pickle
# with open('xgb_model_MIC_final.pkl', 'wb') as file:
#     pickle.dump(model, file)

"""#Visualization"""
import seaborn as sns
import shap
import matplotlib.pyplot as plt

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)

f, ax = plt.subplots(figsize=(13, 10))

plt.scatter(Y_train, train, color='#2d4d85', s=50, label='train data', alpha=0.7)
plt.scatter(Y_test, validation, color='#951d6d', s=50, label='validation', alpha=0.7)
# plt.scatter(Y_test_enc, test, color='#f62d2d', s=50, label='test', alpha=0.7)
plt.plot(Y_train, Y_train, color='#444444', linewidth=2)

plt.title('CatBoost Regressor')
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
# plt.xlim(0, 45)
# plt.ylim(0, 45)

# Save the figure with transparency
plt.savefig('model_cat_MIC_default.png', transparent=True)
plt.show()


'''Feature Importance'''
feature_importance = model.feature_importances_
feature_names = X_train.columns

# Getting feature importance from the model
feature_importance = model.feature_importances_
feature_names = X_train.columns

# Creating a DataFrame to organize feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Define a custom color palette as a gradient from red to blue
custom_palette = sns.color_palette("RdBu", n_colors=len(feature_importance_df))

# Plotting feature importance with custom color gradient
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette=custom_palette)
plt.title('XGB Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()

# Save the plot
plt.savefig('cat_feature_importance_MIC.png', transparent=True)
plt.show()


'''SHAP value plot'''
X_importance = X_train
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)
shap.summary_plot(shap_values, X_importance, show=False)
plt.tight_layout()
plt.savefig('important_features_cat_MIC.png', transparent = True)
plt.show()
