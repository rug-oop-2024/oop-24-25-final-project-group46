import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# Upload a dataset (CSV only)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display the uploaded dataset
    st.write("Preview of the uploaded dataset:")
    st.write(df)
    
    # Get dataset name input from the user
    dataset_name = st.text_input("Dataset name", value=uploaded_file.name.split('.')[0])

# List available datasets in the registry
st.write("Available datasets:")
datasets = automl.registry.list(type="dataset")
for ds in datasets:
    st.write(f"- {ds['name']}")