import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

df_MIC = pd.read_csv(r'D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\data\preprocessed\final_df1_catboost_orig.csv.csv', low_memory=False)
# df_model = df_MIC.drop(['Unnamed: 0'], axis=1)
df_model = df_MIC.copy()
all = df_model.reset_index(drop=True)

# Extract numerical and categorical columns
numerical = all.select_dtypes(include=['int64', 'float64'])
categorical = all.select_dtypes(include=['object'])

cat_col = categorical.columns
num_col = numerical.columns
print('ca', cat_col, 'nc', num_col)

# cat_col = ['np_synthesis', 'method', 'Solvent_for_extract', 'bacteria', 'Order',
#        'prim_specific_habitat', 'Genus', 'sec_habitat', 'shape',
#        'common_environment']
# num_col = ['np_size_min__nm_', 'Temperature_for_extract__C',
#        'Duration_preparing_extract__min', 'np_size_avg__nm_', 'amw',
#        'time_set__hours_', 'Valance_electron', 'avg_Incub_period__h',
#        'coating', 'mdr', 'max_Incub_period__h', 'min_Incub_period__h',
#        'MIC_NP___g_mL_']

# Standard scaler and label encoding
def transform(data, reference_data=None):
    if reference_data is None:
        reference_data = data

    # ----- Encode categorical columns -----
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(reference_data[cat_col])
    X_cat = pd.DataFrame(oe.transform(data[cat_col]), columns=cat_col, index=data.index)

    # ----- Scale numerical columns -----
    sc = StandardScaler()
    sc.fit(reference_data[num_col])
    X_num = pd.DataFrame(sc.transform(data[num_col]), columns=num_col, index=data.index)

    # ----- Combine -----
    X_transformed = pd.concat([X_cat, X_num], axis=1)

    return X_transformed
    '''
    le = LabelEncoder()
    le.fit(categorical.values.flatten())  # Fit the encoder on all categorical data
    Xc_all = categorical.apply(le.transform)
    Xct = data[cat_col].apply(le.transform)

    sc = StandardScaler()
    X_all = sc.fit_transform(numerical)
    X_ss = sc.transform(data[num_col])
    X_sc = pd.DataFrame(X_ss, columns=num_col)
    join = pd.concat([Xct, X_sc], axis=1)
    return join
    '''

print(all, transform(all))

def first_transform(data):
    # le = LabelEncoder()
    # le.fit(data.select_dtypes(include=['object']).values.flatten())  # Fit the encoder on all categorical data
    # Xct = data.select_dtypes(include=['object']).apply(le.transform)
    # # Xct = data[cat_col].apply(le.transform)
    # Xct.reset_index(drop=True)
    #
    # sc = StandardScaler()
    # X_all = sc.fit_transform(data.select_dtypes(include=['int64', 'float64']))
    # X_ss = sc.transform(data[data.select_dtypes(include=['int64', 'float64']).columns])
    # X_sc = pd.DataFrame(X_ss, columns=data.select_dtypes(include=['int64', 'float64']).columns)
    # join = pd.concat([Xct, X_sc], axis=1)
    # return join

    cat_cols = data.select_dtypes(include=['object']).columns
    oe = OrdinalEncoder()
    Xct = pd.DataFrame(oe.fit_transform(data[cat_cols]), columns=cat_cols)

    # Standardize numerical columns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(data[num_cols]), columns=num_cols)

    # Combine categorical and numerical features
    join = pd.concat([Xct.reset_index(drop=True), X_sc.reset_index(drop=True)], axis=1)

    return join

print(all, first_transform(all))