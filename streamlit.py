import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dual-Model Phishing Lab", layout="wide")

@st.cache_resource
def load_and_train_models():
    # 1. Load your 5,000 row dataset
    # Ensure this filename matches exactly what you uploaded to GitHub!
    try:
        df = pd.read_csv('StealthPhisher_mini.csv') 
    except FileNotFoundError:
        # Fallback to original name if you didn't rename it
        df = pd.read_csv('StealthPhisher2025.csv')

    # 2. Select features
    ui_features = ['LengthOfURL', 'CntFilesJS', 'CharacterComplexity']
    features_only = df[ui_features] 
    
    numerical_cols = features_only.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = features_only.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 3. Build Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', RobustScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    X_processed = preprocessor.fit_transform(features_only)

    # 4. Isolation Forest  
    iso_forest = IsolationForest(contamination=0.25, random_state=42)
    iso_forest.fit(X_processed)

    # 5. DBSCAN + KNN Proxy (To allow DBSCAN to handle new inputs)
    dbscan = DBSCAN(eps=10, min_samples=128 )
    clusters = dbscan.fit_predict(X_processed)
    
    knn_proxy = KNeighborsClassifier(n_neighbors=5)
    knn_proxy.fit(X_processed, clusters)
    
    # 6. Create Baseline for UI Input
    baseline = features_only.iloc[[0]].copy()
    
    # Only calculate median if there are numerical columns
    if len(numerical_cols) > 0:
        baseline[numerical_cols] = features_only[numerical_cols].median().values
        
    # Only calculate mode if there are categorical columns
    if len(categorical_cols) > 0:
        baseline[categorical_cols] = features_only[categorical_cols].mode().iloc[0].values
    
    return preprocessor, iso_forest, knn_proxy, baseline

# Initialize models
try:
    preprocessor, iso_model, db_model, baseline_template = load_and_train_models()
except Exception as e:
    st.error(f"⚠️ Deployment Error: {e}")
    st.info("Check if your CSV file and requirements.txt are uploaded to GitHub.")
    st.stop()

# --- STREAMLIT UI ---
st.title("🛡️ Phishing Detection: Isolation Forest vs. DBSCAN")
st.write("With 5,000 training samples, the model now has a stronger baseline for 'Normal' behavior.")

st.sidebar.header("Test Email Features")
url_len = st.sidebar.number_input("URL Length", value=20, help="Standard URLs are usually under 50 characters.")
js_count = st.sidebar.number_input("JS Files", value=0, help="Legitimate emails rarely contain multiple JS attachments.")
complexity = st.sidebar.slider("Complexity", 0.0, 1.0, 0.05, help="Randomness score of the URL.")

if st.button("Run Dual-Model Analysis"):
    # Prepare input
    input_df = baseline_template.copy()
    input_df['LengthOfURL'] = url_len
    input_df['CntFilesJS'] = js_count
    input_df['CharacterComplexity'] = complexity
    
    input_scaled = preprocessor.transform(input_df)
    
    # Predict
    iso_res = iso_model.predict(input_scaled)[0]
    db_res = db_model.predict(input_scaled)[0]
    
    # Format Labels
    iso_label = "✅ LEGITIMATE" if iso_res == 1 else "🚨 PHISHING"
    db_label = "✅ LEGITIMATE (Cluster)" if db_res != -1 else "🚨 PHISHING (Noise)"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Isolation Forest")
        st.metric("Detection Result", iso_label)
        st.write("Analyzes how 'easy' it is to separate this point from the rest.")
        
    with col2:
        st.subheader("DBSCAN (Density)")
        st.metric("Detection Result", db_label)
        st.write("Analyzes if this point lives in a crowded 'normal' neighborhood.")

    st.divider()
    if iso_res == 1 and db_res != -1:
        st.success("### Final Verdict: SAFE")
        st.write("Both models agree this email follows standard patterns.")
    elif iso_res == -1 and db_res == -1:
        st.error("### Final Verdict: MALICIOUS")
        st.write("High confidence phishing attempt! Both algorithms flagged this as an anomaly.")
    else:
        st.warning("### Final Verdict: SUSPICIOUS")
        st.write("The models disagree. This email should be manually reviewed.")