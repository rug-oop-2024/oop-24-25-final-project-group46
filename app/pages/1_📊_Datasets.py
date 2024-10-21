import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# Upload a dataset (CSV only)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the uploaded dataset
    st.write("Preview of the uploaded dataset:")
    st.write(df)
    
    # Get dataset name input from the user
    dataset_name = st.text_input("Dataset name", value=uploaded_file.name.split('.')[0])
    
    # Save the dataset to AutoML system if a name is provided
    if st.button("Save Dataset"):
        if dataset_name:
            # Create a Dataset object using the from_dataframe method
            dataset = Dataset.from_dataframe(data=df, name=dataset_name, asset_path=f"{dataset_name}.csv", version="1.0.0")
            
            # Save the dataset into the AutoML system's registry
            automl.registry.save(dataset)
            st.success(f"Dataset '{dataset_name}' has been saved successfully!")
        else:
            st.error("Please provide a valid dataset name.")

# List available datasets in the registry
st.write("Available datasets:")
datasets = automl.registry.list(type="dataset")
for ds in datasets:
    st.write(f"- {ds['name']}")