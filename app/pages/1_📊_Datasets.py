import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# Upload a dataset (CSV only)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Preview of the uploaded dataset:")
    st.write(df)
    
    dataset_name = st.text_input("Dataset name", value=uploaded_file.name.split('.')[0])
    
    if st.button("Detect Feature Types"):
        detected_features = detect_feature_types(Dataset.from_dataframe(df, name=dataset_name, asset_path=f"{dataset_name}.csv"))
        
        # Display detected feature types
        st.write("Detected feature types:")
        for feature in detected_features:
            st.write(f"Feature: {feature.name}, Type: {feature.type}")
