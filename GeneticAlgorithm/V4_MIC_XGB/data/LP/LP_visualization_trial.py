import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df1 = pd.read_csv(r'C:\Users\user\Desktop\Antimicrobial NP\V4_MIC\data\LP\Final_Model_comparision_raw_MIC.csv')
df2 = pd.read_csv(r'C:\Users\user\Desktop\Antimicrobial NP\V4_MIC\data\LP\Final_Model_comparision_processed_MIC.csv')

# Add a column to each data frame indicating the source
df1['Source'] = 'Raw Data'
df2['Source'] = 'Processed Data'

# Filter top 10 models by Adjusted R-Squared
top_10_r_squared = df2.nlargest(10, 'Adjusted R-Squared')

# Select relevant raw data for the top 10 preprocessed models
raw_data_top_10 = df1[df1['Model'].isin(top_10_r_squared['Model'])]

# Combine the top 10 preprocessed and raw data
combined_top_10 = pd.concat([top_10_r_squared, raw_data_top_10])

# Filter out rows with negative R-Squared values
df_r_squared = combined_top_10[combined_top_10['Adjusted R-Squared'] >= 0]

# Set plot style and font sizes
sns.set_style("whitegrid")
sns.set(font="Arial", rc={"axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10})

# Define visually appealing color palettes
palette_r_squared = sns.color_palette("Blues_r", 10)  # Blue palette for R-Squared
palette_rmse = sns.color_palette("Blues_r", 10)  # Purple palette for RMSE


# Create the figure and axis for R-Squared
fig, ax1 = plt.subplots(figsize=(8, 8))
sns.barplot(x="Adjusted R-Squared", y="Model", hue="Source", data=df_r_squared, palette=[palette_r_squared[0], palette_r_squared[5]], ax=ax1)
ax1.set_xlabel("Adjusted R-Squared", fontsize=12)
ax1.set_ylabel("Model", fontsize=12)
ax1.set_title("Comparison of Regression Models", fontsize=14, fontweight="bold")
ax1.set_xticks([0.2, 0.4, 0.6, 0.8])
ax1.legend().set_visible(False)  # Turn off the legend
sns.despine(ax=ax1, left=True, bottom=True)
ax1.xaxis.grid(color='black', alpha=0.3)

# Save the R-Squared plot
plt.tight_layout()
plt.savefig("top_10_models_r_squared_MIC_final.png", dpi=300, transparent=True)
plt.show()

# Filter top 10 preprocessed data by RMSE
top_10_rmse = df2.nsmallest(35, 'RMSE')

# Select relevant raw data for the top 10 preprocessed models
raw_data_top_10_rmse = df1[df1['Model'].isin(top_10_rmse['Model'])]

# Combine the top 10 preprocessed and raw data
combined_top_10_rmse = pd.concat([top_10_rmse, raw_data_top_10_rmse])

# Create the figure and axis for combined data
fig, ax = plt.subplots(figsize=(8, 10))

# Plot RMSE for top 10 preprocessed and related raw data
sns.barplot(x="RMSE", y="Model", hue="Source", data=combined_top_10_rmse, palette=[palette_rmse[0], palette_rmse[5]], ax=ax)
ax.set_xlabel("RMSE", fontname="Arial", fontsize=12)
ax.set_ylabel("Model", fontname="Arial", fontsize=12)
ax.set_title("Comparison of RMSE of Preprocessed and Raw Data", fontname="Arial", fontsize=14, fontweight="bold")
ax.legend().set_visible(False)  # Turn off the legend
sns.despine(ax=ax, left=True, bottom=True)
ax.xaxis.grid(color='black', alpha=0.3)

# Save the plot
plt.tight_layout()
plt.savefig("top_35_model_comparision_RMSE_MIC.png", dpi=300, transparent=True)
plt.show()
