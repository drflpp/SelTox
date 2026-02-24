import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

MIC_raw = pd.read_csv(r'C:\Users\user\Desktop\Antimicrobial NP\V4_MIC\data\raw\final_merged_df_with_ref.csv')
Preprocessed = pd.read_csv(r'C:\Users\user\Desktop\Antimicrobial NP\V4_MIC\data\preprocessed\preprocessed_MIC_mod.csv')
print('Preprocessed', Preprocessed.columns)
MIC_raw = MIC_raw.drop(['source'], axis=1)
Preprocessed = Preprocessed.drop(['Unnamed: 0', 'source'], axis=1)
MIC_data = Preprocessed  # Renamed ZOI_data to MIC_data

''' Distribution of NP in MIC database'''
# Count the occurrences of each nanoparticles
nanoparticle_counts = MIC_data['np'].value_counts()
print('nanoparticle_counts', nanoparticle_counts)
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

# Plotting the pie chart with 'Other' category and increased separation
plt.figure(figsize=(8, 8))
plt.pie(nanoparticle_counts, labels=nanoparticle_counts.index, colors=nanoparticle_colors,
        autopct='%1.1f%%', startangle=0, pctdistance=0.85, explode=explode)
plt.axis('equal')
plt.savefig('np_pie_chart_preprocessed_MIC.png', bbox_inches='tight', transparent=True)
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
bacteria_colors = ['#B7094C', '#A01A58', '#892B64', '#723C70', '#5C4D7D', '#455E89', '#2E6F95', '#1780A1', '#0091AD', '#48BFE3', '#56CFE1', '#64DFDF']

# Add more colors or modify the list as needed for your categories

# Create explode values for separation between slices
explode_bacteria = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)  # Adjust the values for separation

# Plotting the pie chart for bacteria with 'Other' category and increased separation
plt.figure(figsize=(8, 8))
plt.pie(bacteria_counts, labels=bacteria_counts.index, colors=bacteria_colors,
        autopct='%1.1f%%', startangle=0, pctdistance=0.85, explode=explode_bacteria)
plt.axis('equal')
plt.savefig('bacteria_pie_chart_MIC_preprocessed.png', bbox_inches='tight', transparent=True)
plt.show()

columns_to_plot = ['concentration', 'np_size_avg (nm)', 'time_set', 'min_Incub_period, h']
cols = ['log_concentration (ug/ml)', 'size (nm)', 'reaction time (h)', 'incubation period (h)']

log_scale_columns = ['concentration']
num_cols = len(columns_to_plot)
fig, axes = plt.subplots(1, num_cols, figsize=(18, 6))

for i, column in enumerate(columns_to_plot):
    if column in log_scale_columns:
        # Create log scale plot for 'concentration'
        sns.kdeplot(x=np.log10(MIC_raw[column]), ax=axes[i], color='#B7094C', linewidth=2)
        sns.kdeplot(x=Preprocessed[column], ax=axes[i], color='#0091AD', linewidth=2)

    else:
        sns.kdeplot(x=MIC_raw[column], ax=axes[i], color='#B7094C', linewidth=2, )
        sns.kdeplot(x=Preprocessed[column], ax=axes[i], color='#0091AD', linewidth=2, )

        # Add dashed line at the end of the plot
        axes[i].axvline(Preprocessed[column].max(), color='#0091AD', linestyle='--', linewidth=2)

    axes[i].set_xlabel(cols[i])  # Change x-label to the corresponding value in 'cols'
    axes[i].set_ylabel('Density')
    # axes[i].legend()

    axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

plt.tight_layout()
plt.savefig('kde_plot_MIC_log_scale.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

numerical_cols_raw = MIC_raw.select_dtypes(include=['float64', 'int64'])

# Dropping 'source' column from processed data
processed_without_source = Preprocessed.select_dtypes(include=['float64', 'int64']).drop(columns=['source'], errors='ignore')

# Create a mask to display only the lower triangle
mask = np.triu(np.ones_like(numerical_cols_raw.corr(), dtype=bool))

# Creating a visually appealing heatmap for raw numerical columns
plt.figure(figsize=(24, 20))
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Blue color palette
sns.heatmap(numerical_cols_raw.corr(), annot=False, cmap=cmap, mask=mask, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-0.9, vmax=0.9)  # Set correlation range
plt.title('Heatmap raw data')
plt.tight_layout()
plt.savefig('heatmap_raw_MIC.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Create a mask to display only the lower triangle
mask = np.triu(np.ones_like(processed_without_source.corr(), dtype=bool))

# Creating a visually appealing heatmap for preprocessed numerical columns without 'source' column
plt.figure(figsize=(12, 10))
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Blue color palette
sns.heatmap(processed_without_source.corr(), annot=True, cmap=cmap, mask=mask, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-0.9, vmax=0.9)  # Set correlation range
plt.title('Heatmap processed data')
plt.tight_layout()
plt.savefig('heatmap_preprocessed_MIC_annot.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
