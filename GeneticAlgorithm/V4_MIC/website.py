import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # to load your pretrained model


st.set_page_config(page_title="SelTox NPs", layout="wide")

st.title("Selectively Toxic Nanoparticles")

# Load your dataset for statistics/visualization
@st.cache_data
def load_dataset():
    df = pd.read_csv(r"D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\data\preprocessed\final_df1_catboost_orig_bact.csv", index_col=0).reset_index(drop=True)
        # .drop('Unnamed: 0', axis=0)  # your dataset
    return df

data = load_dataset()
st.subheader("Dataset Statistics")
st.write(data.describe())

# Visualization example: distribution of a numeric feature
st.subheader("Feature Distribution")
feature = st.selectbox("Select feature to visualize", data.select_dtypes(include=np.number).columns)
plt.figure(figsize=(8, 4))
sns.histplot(data[feature], kde=True, bins=30)
st.pyplot(plt)

# Load pretrained model
model = joblib.load(r"D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\Models\CatBoost_v2\model_cat_p17.joblib")  # your model

# User input via widgets
st.subheader("Make a Prediction")

# Example: assume your model uses these features
feature_inputs = {}
for col in data.select_dtypes(include=np.number).columns:  # replace with your model features
    min_val = float(data[col].min())
    max_val = float(data[col].max())
    mean_val = float(data[col].mean())
    feature_inputs[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

# Convert input to dataframe
input_df = pd.DataFrame([feature_inputs])

# Prediction
if st.button("Predict Toxicity"):
    pred = model.predict(input_df)
    pred_proba = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None
    st.success(f"Predicted Toxicity: {pred[0]}")
    if pred_proba is not None:
        st.info(f"Prediction Probabilities: {pred_proba[0]}")

# Optional: Show comparison with dataset
st.subheader("Comparison with Dataset")
st.write(data.describe())