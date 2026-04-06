import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="Phishing Detection", layout="wide")

@st.cache_resource
def load_and_train_model():
    # 1. Load dataset
    try:
        df = pd.read_csv('StealthPhisher_mini.csv') 
    except FileNotFoundError:
        df = pd.read_csv('StealthPhisher2025.csv')

    # 2. Select ONLY the features controlled by the UI
    ui_features = ['LengthOfURL', 'CntFilesJS', 'CharacterComplexity']
    available_features = [f for f in ui_features if f in df.columns]
    features_only = df[available_features]
    
    # 3. Scale the data (No need for Complex ColumnTransformers now!)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features_only)

    # 4. Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    iso_forest.fit(X_scaled)
    
    # 5. Create Baseline for UI Input
    baseline = features_only.iloc[[0]].copy()
    baseline[available_features] = features_only[available_features].median().values
    
    return scaler, iso_forest, baseline

# Initialize model
try:
    scaler, iso_model, baseline_template = load_and_train_model()
except Exception as e:
    st.error(f"⚠️ Deployment Error: {e}")
    st.stop()

# --- STREAMLIT UI ---
st.title("🛡️ Phishing Detection: Isolation Forest")
st.write("Using anomaly detection to find suspicious email patterns.")

st.sidebar.header("Test Email Features")
url_len = st.sidebar.number_input("URL Length", value=20, help="Standard URLs are usually under 50 characters.")
js_count = st.sidebar.number_input("JS Files", value=0, help="Legitimate emails rarely contain multiple JS attachments.")
complexity = st.sidebar.slider("Complexity", 0.0, 1.0, 0.05, help="Randomness score of the URL.")

if st.button("Run Analysis"):
    # Prepare input
    input_df = baseline_template.copy()
    input_df['LengthOfURL'] = url_len
    input_df['CntFilesJS'] = js_count
    input_df['CharacterComplexity'] = complexity
    
    # Scale and Predict
    input_scaled = scaler.transform(input_df)
    iso_res = iso_model.predict(input_scaled)[0]
    
    st.divider()
    
    # Show Results
    if iso_res == 1:
        st.success("### Final Verdict: ✅ LEGITIMATE")
        st.write("The Isolation Forest model considers this email to be within normal parameters.")
    else:
        st.error("### Final Verdict: 🚨 PHISHING (Anomaly Detected)")
        st.write("High confidence phishing attempt! The Isolation Forest flagged these features as highly unusual.")