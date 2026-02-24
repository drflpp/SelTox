import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

MIC_data_raw = pd.read_csv(r'C:\Users\user\Desktop\Antimicrobial NP\V4_MIC\data\raw\final_merged_df_with_ref.csv')
print(MIC_data_raw.columns)

# Calculate the percentage of missing values in each column
# missing_percentage = MIC_data_raw.isnull().mean() * 100
# missing_percentage.to_csv('missing_info_raw.csv', header=True) # subkindom has 98% and clade has 48% missing values
# remove these columns


MIC_1 = MIC_data_raw.drop(['combination', 'bac_type_enc', 'strain','mdr', 'id_bac','subkingdom', 'clade', 'species','CID', 'Canonical_smiles', 'amw', 'np_size_min (nm)', 'np_size_max (nm)'], axis=1)

def rename_method(value):
    if value.startswith('c'):
        return 'chemical_synthesis'
    elif value.startswith('g'):
        return 'green_synthesis'
    else:
        return value

MIC_1['np_synthesis'] = MIC_1['np_synthesis'].apply(lambda x: rename_method(x.lower()))


#remove data where concentration is zero and higher than 2000
# MIC_1 =MIC_data.rename(columns={'normalized_conc (ug/ml)': 'concentration'})
MIC_1 =MIC_1[MIC_1['concentration'] > 0]
MIC_1['concentration'] = np.log10(MIC_1['concentration'])
q = MIC_1['np_size_avg (nm)'].quantile(0.99)
print(q)
MIC_2 = MIC_1[MIC_1['np_size_avg (nm)'] <= 100] # now the max size is 140; with quantile 0.99 it is 100
print(MIC_2['time_set'].quantile(0.99))
MIC_3 = MIC_2[MIC_2['time_set'] <= MIC_2['time_set'].quantile(0.99)] # now the max time_set is 24hr
print(MIC_2['avg_Incub_period, h'].quantile(0.985)) #not enough so removed little more; now the max inc period is 84
MIC_4 = MIC_3[MIC_3['avg_Incub_period, h'] <=  MIC_2['avg_Incub_period, h'].quantile(0.985)] #this function is used and the after removing 1% data max value was 84

plt.figure(figsize=(8, 6))
sns.kdeplot(data=MIC_3['np_size_avg (nm)'])
plt.title('KDE Plot for size')
plt.xlabel('size')
plt.ylabel('Density')
plt.show()

# about mdr there are 11.46% drug resistant strain
# there are total 51.8% 'None' value in ATCC strain column
# there are 86.41% of 'None' value in combination column which have info for nano-composite
numerical_columns = MIC_4.select_dtypes(include=['float64', 'int64'])
categorical_columns = MIC_4.select_dtypes(include=['object'])
# print('ss', numerical_columns)
'''heatmap before variance threshold and highly related column removal'''
# # Create a correlation matrix
# correlation_matrix = numerical_columns.corr()
#
# # Generate the heatmap without displaying values
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
# plt.title('Correlation Heatmap of Numerical Columns')
# plt.show()

from sklearn.feature_selection import VarianceThreshold
# Fit VarianceThreshold with a threshold of 0.9
threshold = 0.85
selector = VarianceThreshold(threshold)
selector.fit(numerical_columns)

# Get the indices of columns to keep
columns_to_keep = numerical_columns.columns[selector.get_support()]

# Filter the DataFrame with columns having variance above the threshold
MIC_data_filtered = numerical_columns[columns_to_keep]

# Create a correlation matrix after variance thresholding
correlation_matrix_variance = MIC_data_filtered.corr()

# Remove highly correlated columns based on Pearson correlation coefficient threshold
high_corr_columns = set()  # To store highly correlated columns
corr_threshold = 0.85  # Pearson correlation coefficient threshold

for i in range(len(correlation_matrix_variance.columns)):
    for j in range(i):
        if abs(correlation_matrix_variance.iloc[i, j]) > corr_threshold:
            colname = correlation_matrix_variance.columns[i]
            high_corr_columns.add(colname)

# Create a DataFrame without highly correlated columns
MIC_data_no_high_corr = MIC_data_filtered.drop(columns=high_corr_columns)

# Generate the heatmap after removing highly correlated columns
plt.figure(figsize=(10, 8))
sns.heatmap(MIC_data_no_high_corr.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap after Variance and Pearson Correlation Threshold Filtering')
plt.show()

final_MIC = pd.concat([categorical_columns, MIC_data_no_high_corr], axis=1)
final_MIC=final_MIC.drop_duplicates()
final_MIC.reset_index(drop=True)
final_MIC.to_csv('preprocessed_MIC_original8585.csv')
print('final_MIC', final_MIC.info())