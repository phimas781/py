import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('data/gwamz_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('data/model.pkl')

# App configuration
st.set_page_config(page_title="Gwamz Analytics", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Dashboard", "Predictions"])

# Main content
if page == "Dashboard":
    st.header("Music Performance Dashboard")
    data = load_data()
    st.dataframe(data.head())
    
elif page == "Predictions":
    st.header("Stream Prediction Tool")
    model = load_model()
    
    # Prediction form
    with st.form("prediction_form"):
        st.write("Enter track details:")
        release_date = st.date_input("Release Date")
        is_explicit = st.checkbox("Explicit Content")
        is_collab = st.checkbox("Collaboration")
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Make prediction (replace with your actual prediction code)
            prediction = model.predict([[1, 2, 3]])  # Example
            st.success(f"Predicted streams: {prediction[0]:,}")
