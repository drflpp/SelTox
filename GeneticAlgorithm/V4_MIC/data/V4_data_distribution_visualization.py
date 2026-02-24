import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MIC_data_raw = pd.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\data\raw\final_merged_df_with_ref.csv')
MIC_data_raw['concentration'] = np.log10(MIC_data_raw['concentration'])
MIC_data_raw = MIC_data_raw.reset_index(drop=True)

MIC_data = pd.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\data\preprocessed\preprocessed_MIC_mod.csv')
MIC_data = MIC_data.drop(['Unnamed: 0'], axis=1)
''' Distribution of NP in MIC database'''
# Count the occurrences of each nanoparticles
nanoparticle_counts = MIC_data['np'].value_counts()
# print('nanoparticle_counts', nanoparticle_counts)
# Calculate total count
total_count = nanoparticle_counts.sum()

# Find categories with less than 2% occurrence
threshold = 0.02 * total_count
rare_categories = nanoparticle_counts[nanoparticle_counts < threshold].index

# Group rare categories into 'Other'
nanoparticle_counts.loc[nanoparticle_counts.index.isin(rare_categories)] = \
    nanoparticle_counts.loc[nanoparticle_counts.index.isin(rare_categories)].sum()

# Remove rare categories from the index
nanoparticle_counts = nanoparticle_counts[~nanoparticle_counts.index.isin(rare_categories)]

# Create a separate category for 'Other'
other_count = rare_categories.value_counts().sum()
nanoparticle_counts['Other'] = other_count

# Define a color palette using hex codes for nanoparticles including 'Other'
nanoparticle_colors = ['#B7094C', '#A01A58', '#892B64', '#723C70', '#5C4D7D', '#455E89', '#2E6F95', '#1780A1', '#0091AD', '#48BFE3']
# Add more colors or modify the list as needed for your categories
# Create explode values for separation between slices
explode = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)  # Adjust the values for separation
font_size = 12
# Plotting the pie chart with 'Other' category and increased separation
plt.figure(figsize=(8, 8))
plt.pie(nanoparticle_counts, labels=nanoparticle_counts.index, colors=nanoparticle_colors,
        autopct='%1.1f%%', startangle=0, pctdistance=0.85, explode=explode,
        textprops={'fontsize': font_size, 'fontweight': 'bold'})  # Adjust font size and family as needed
plt.axis('equal')
# plt.savefig('np_pie_chart_processed_final.png', bbox_inches='tight', transparent=True)
plt.show()

''' Distribution of NP in MIC database'''
# Count the occurrences of each unique bacteria
bacteria_counts = MIC_data['bacteria'].value_counts()

# Calculate total count
total_count = bacteria_counts.sum()

# Find categories with less than 2% occurrence
threshold = 0.02 * total_count
rare_bacteria = bacteria_counts[bacteria_counts < threshold].index

# Group rare bacteria into 'Other'
bacteria_counts.loc[bacteria_counts.index.isin(rare_bacteria)] = \
    bacteria_counts.loc[bacteria_counts.index.isin(rare_bacteria)].sum()

# Remove rare bacteria from the index
bacteria_counts = bacteria_counts[~bacteria_counts.index.isin(rare_bacteria)]

# Create a separate category for 'Other'
other_bacteria_count = rare_bacteria.value_counts().sum()
bacteria_counts['Other'] = other_bacteria_count

# Define a color palette using hex codes for bacteria including 'Other'
bacteria_colors = ['#B7094C', '#A01A58', '#892B64', '#723C70', '#5C4D7D', '#455E89', '#2E6F95', '#1780A1', '#0091AD', '#48BFE3']
# Add more colors or modify the list as needed for your categories

# Create explode values for separation between slices
explode_bacteria = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)  # Adjust the values for separation

# Plotting the pie chart for bacteria with 'Other' category and increased separation
plt.figure(figsize=(8, 8))
plt.pie(bacteria_counts, labels=bacteria_counts.index, colors=bacteria_colors,
        autopct='%1.1f%%', startangle=0, pctdistance=0.85, explode=explode_bacteria,
        textprops={'fontsize': font_size,  'fontweight': 'bold'})  # Adjust font size and family as needed
plt.axis('equal')
# plt.savefig('bacteria_pie_chart_MIC_preprocessed_final.png', bbox_inches='tight', transparent=True)
plt.show()

columns_to_plot = ['concentration', 'np_size_avg (nm)', 'time_set', 'min_Incub_period, h']
cols = ['concentration (ug/ml)', 'size (nm)', 'time set (h)', 'Incubation period (h)']

num_cols = len(columns_to_plot)
fig, axes = plt.subplots(1, num_cols, figsize=(18, 6))

for i, column in enumerate(columns_to_plot):
    sns.kdeplot(x=MIC_data_raw[column], ax=axes[i], color='#B7094C', linewidth = 2)
    sns.kdeplot(x=MIC_data[column], ax=axes[i], color='#0091AD', linewidth = 2)

    max_value = MIC_data[column].max()  # Maximum value of preprocessed data
    print('max', max_value)
    axes[i].axvline(max_value, color='#0091AD', linestyle='--', linewidth=2)

    axes[i].set_xlabel(cols[i])  # Change x-label to the corresponding value in 'cols'
    axes[i].set_ylabel('Density')

plt.tight_layout()
# plt.savefig('kde_plot.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

numerical_cols_raw = MIC_data_raw.select_dtypes(include=['float64', 'int64'])

# Dropping 'source' column from processed data
processed_without_source = MIC_data.select_dtypes(include=['float64', 'int64']).drop(columns=['source'], errors='ignore')

# Create a mask to display only the lower triangle
mask = np.triu(np.ones_like(numerical_cols_raw.corr(), dtype=bool))

# Creating a visually appealing heatmap for raw numerical columns
plt.figure(figsize=(24, 20))
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Blue color palette
sns.heatmap(numerical_cols_raw.corr(), annot=False, cmap=cmap, mask=mask, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-0.9, vmax=0.9)  # Set correlation range
plt.title('Heatmap raw data')
plt.tight_layout()
# plt.savefig('heatmap_raw.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()



# Create a mask to display only the lower triangle
mask = np.triu(np.ones_like(processed_without_source.corr(), dtype=bool))
cmap = sns.diverging_palette(250, 10, as_cmap=True)
# Creating a visually appealing heatmap for preprocessed numerical columns without 'source' column
plt.figure(figsize=(12, 10))
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Blue color palette
sns.heatmap(processed_without_source.corr(), annot=False, cmap=cmap, mask=mask, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-0.9, vmax=0.9)  # Set correlation range
plt.title('Heatmap processed data')
plt.tight_layout()
# plt.savefig('heatmap_preprocessed.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Drop rows with NaN or infinite values in the specified columns_to_plot
MIC_data_raw = MIC_data_raw.dropna(subset=columns_to_plot)
MIC_data_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
MIC_data_raw = MIC_data_raw.dropna(subset=columns_to_plot)

# Set up the figure and axes
num_cols = len(columns_to_plot)
fig, axes = plt.subplots(1, num_cols, figsize=(9, 3))  # Adjust the figure size as needed
max_values = [4.09, 100, 30, 60]

# Loop through each column and create a horizontal violin plot
for i, column in enumerate(columns_to_plot):
    sns.violinplot(x=MIC_data_raw[column], ax=axes[i], color='#005b96')  # Use 'Blues_r' for reversed colormap

    # Access individual elements of the list for max values
    max_val = max_values[i]

    axes[i].axvline(max_val, linestyle='--', color='#03396c', linewidth=2, label=f'Max value: {max_val:.2f}')  # Use 'blue' for line color
    axes[i].set_xlabel(column)  # Set x-label to the corresponding column name
    axes[i].set_ylabel('Density')

plt.tight_layout()
# plt.savefig('violin_plot_raw_horizontal.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
