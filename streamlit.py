import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Phishing Detector", layout="centered")

@st.cache_resource
def load_and_train_model():
    # 1. Load your dataset (ensure StealthPhisher2025.csv is in the same folder)
    df = pd.read_csv('StealthPhisher_mini.csv')
    generated_df = df.sample(n=5000, random_state=42)
    
    # 2. Data Preparation
    features_only = generated_df.drop(columns=['Label', 'url']) if 'url' in generated_df.columns else generated_df.drop(columns=['Label'])
    numerical_cols = features_only.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = features_only.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 3. Build Pipeline (Using RobustScaler for your anomalies)
    preprocessor = ColumnTransformer(transformers=[
        ('num', RobustScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', IsolationForest(contamination=0.5, random_state=42))
    ])
    
    model.fit(features_only)
    
    # 4. Create a baseline for user inputs
    baseline = features_only.iloc[[0]].copy()
    baseline[numerical_cols] = features_only[numerical_cols].median().values
    baseline[categorical_cols] = features_only[categorical_cols].mode().iloc[0].values
    
    return model, baseline

# Initialize
try:
    model, baseline_template = load_and_train_model()
except FileNotFoundError:
    st.error("Error: 'StealthPhisher2025.csv' not found. Please upload the dataset to your repository.")
    st.stop()

# --- STREAMLIT UI ---
st.title("🛡️ Phishing Anomaly Detector")
st.write("This prototype uses **Unsupervised Machine Learning** to detect suspicious email patterns.")

st.subheader("Simulate Email Features")
col1, col2 = st.columns(2)

with col1:
    url_len = st.number_input("URL Length", min_value=0, value=30)
    js_count = st.number_input("JS File Count", min_value=0, value=0)

with col2:
    complexity = st.slider("Character Complexity", 0.0, 1.0, 0.05)
    
if st.button("Analyze Pattern"):
    # Prepare input for prediction
    input_df = baseline_template.copy()
    input_df['LengthOfURL'] = url_len
    input_df['CntFilesJS'] = js_count
    input_df['CharacterComplexity'] = complexity
    
    prediction = model.predict(input_df)
    
    st.divider()
    if prediction[0] == -1:
        st.error("### ⚠️ Result: PHISHING ANOMALY")
        st.write("The model identified this feature combination as a statistical outlier.")
    else:
        st.success("### ✅ Result: LEGITIMATE")
        st.write("This pattern matches the majority of 'normal' emails in the training set.")
