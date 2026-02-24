import time
import numpy as np
import pandas as pd
# import V4_models
import random
import polars as pl
import math
import pyarrow
# from Models import V4_transform_MIC_trial
from V4_models import pipeline

# df_MIC = pd.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\data\preprocessed\final_df1_catboost_orig.csv', low_memory=False, index_col=0)
# df_MIC_bacteria = pd.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\data\preprocessed\final_df1_catboost_orig_bact.csv', low_memory=False, index_col=0)

# X = df_MIC_bacteria.drop(['MIC_NP___g_mL_'], axis=1) # no need for concentration, zoi or gi as all of these parameters will be predicted

pipeline.plot_top_features_gradient1(top_n=40)