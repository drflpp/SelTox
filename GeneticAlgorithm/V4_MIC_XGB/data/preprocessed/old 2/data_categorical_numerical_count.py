import pandas as pd

# Read the preprocessed data
preprocessed_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\preprocessed\preprocessed_MIC_mod.csv')

# Identify categorical and numerical columns
categorical_columns = preprocessed_data.select_dtypes(include=['object']).columns
numerical_columns = preprocessed_data.select_dtypes(include=['float64', 'int64']).columns

# Save categorical information in a CSV file
categorical_info = pd.DataFrame({
    'Categorical Columns': categorical_columns,
    'Number of Unique Values': [preprocessed_data[col].nunique() for col in categorical_columns]
})

categorical_info.to_csv('categorical_information_processed.csv', index=False)

# Save numerical information in a CSV file
numerical_info = pd.DataFrame({
    'Numerical Columns': numerical_columns
})

# Add describe() output for numerical columns
numerical_info = pd.concat([numerical_info, preprocessed_data[numerical_columns].describe().transpose()], axis=1)

numerical_info.to_csv('numerical_information_processed.csv', index=False)
