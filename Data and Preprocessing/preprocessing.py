import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def fill_na_mode(df, col_name):
    df[col_name] = df[col_name].fillna(df[col_name].mode()[0])
    return df


def fill_na_mean(df, col_name):
    df[col_name] = df[col_name].fillna(df[col_name].mean())
    return df


def fill_size_missing_values(df, avg_col='np_size_avg (nm)', max_col='np_size_max (nm)', min_col='np_size_min (nm)'):
    result_df = df.copy()
    avg_mask = result_df[avg_col].isna()
    result_df.loc[avg_mask, avg_col] = (
        result_df.loc[avg_mask, max_col] + result_df.loc[avg_mask, min_col]
    ) / 2
    max_mask = result_df[max_col].isna()
    result_df.loc[max_mask, max_col] = result_df.loc[max_mask, avg_col]
    min_mask = result_df[min_col].isna()
    result_df.loc[min_mask, min_col] = result_df.loc[min_mask, avg_col]
    return result_df




if __name__ == '__main__':
    MIC_method = ['MIC', 'MBC', 'MBEC', 'MBIC', 'MIc', 'MFC', 'MMC']

    data = pd.read_excel('validated_data_merged_tax.xlsx', index_col=0)

    cols_to_drop = [
        'new sn', 'sn', 'new', 'article_list', 'journal_name', 'publisher', 'year', 'title',
        'journal_is_oa', 'is_oa', 'oa_status', 'verification required',
        'verified_by', 'verification_date', 'has_mistake_in_data',
        'has_mistake_in_matadata', 'entry_status', 'comment', 'accept/reject',
        'Unnamed: 44', 'IdList', 'zeta_potential (mV)', 'pH during synthesis',
        'Concentration of precursor (mM)', 'hydrodynamic diameter', 'Precursor of NP',
        'Clade', 'Class', 'Family'
    ]

    df = data.copy()
    df['np_size_avg (nm)'] = df['np_size_avg (nm)'].astype('float64')

    fill_mode = ['time_set (hours)', 'Solvent for extract', 'shape']
    fill_mean = ['Duration preparing extract, min', 'Temperature for extract, C']
    for col_name in fill_mode:
        fill_na_mode(df, col_name)
    for col_name in fill_mean:
        fill_na_mean(df, col_name)

    filled_df = fill_size_missing_values(df)

    MIC_df = filled_df[filled_df['method'].isin(MIC_method)]
    MIC_df = MIC_df.drop(['concentration for ZOI (µg/ml)', 'zoi_np (mm)'], axis=1)


    MIC_df.to_csv('MIC_df.csv')
