import pandas as pd

raw = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\raw\data_with_desc_raw_MIC.csv')
val = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_MIC\data\raw\val2_with_descriptor.csv')
combined_data = pd.concat([raw, val], ignore_index=True)



# dfn = pd.merge(raw, val, left_on='bacteria', right_on='bacteria', how='left')
combined_data.to_csv('merged_df.csv')
print(combined_data)