import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
validation = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\Models\Cat\validation_set_with_predicted_results.csv')

validation['error'] = abs(validation['concentration']-validation['predicted'])
sns.displot(data=validation, x = 'error', color = 'red', kind='hist', bins= 20)
# Add labels and a title
plt.xlabel('Error', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Distribution of Error in test data', fontsize=16)

# Show the plot
plt.savefig('validation_set_error_amount.png', transparent=True)
plt.show()



fig, ax1 = plt.subplots(figsize=(10, 8))

sns.scatterplot(data=validation, x='concentration', y ='error', color='red', ax=ax1)
ax1.set_xlabel('concentration')
ax1.set_ylabel('Error')
ax1.set_title('Concentration and prediction error distribution ', fontweight='bold')
ax1.grid(True, linestyle='--', color='gray', alpha=0.7)

ax2 = ax1.twinx()
sns.distplot(validation['concentration'], kde=True, color='blue', ax=ax2)
ax2.set_ylabel('Concentration diistribution')
ax2.grid(False)

# Add legend and adjust plot layout
ax1.legend(['Prediction Error'], loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig('validation_error_with_concentration_distribution.png', transparent=True)
plt.show()