import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
MIC_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\preprocessed\preprocessed_MIC_mod.csv')
print(MIC_data.columns)
# Convert 'np_size_avg (nm)' to categorical variable with 10 bins
MIC_data['size_bins'] = pd.cut(MIC_data[ 'np_size_avg (nm)'], bins=20)

# Create a dot plot using seaborn
plt.figure(figsize=(12, 8))
sns.stripplot(x= 'size_bins', y='concentration', data=MIC_data, jitter=True, size=8, alpha=0.7)

# Set plot labels and title
plt.xlabel('Size Bins')
plt.ylabel('Concentration')
plt.title('Dot Plot of Concentration vs. Size')

# Show the plot
plt.show()
