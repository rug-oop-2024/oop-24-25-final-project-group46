import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
import random
import string

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="Dataset")


st.set_page_config(page_title="Dataset Management", page_icon="📊")

st.write("# 📂 Dataset Management")
st.write("Upload a CSV file to create a new dataset and detect feature types.")

# Upload a dataset (CSV only)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    st.write("Preview of the uploaded dataset:")
    st.write(df)

    dataset_name = st.text_input(
        "Dataset name", value=uploaded_file.name.split('.')[0]
    )

    tags = st.text_input("Give the dataset a tag.")

    def _generate_id(prefix: str) -> str:
        """Generates the random id's for metadata"""
        random_part = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=12))
        return f"{prefix}-{random_part}"

    metadata = {
        "experiment_id": _generate_id("exp"),
        "run_id": _generate_id("run")
    }

    if st.button("Save Dataset"):
        dataset = Dataset.from_dataframe(
            df,
            name=dataset_name,
            asset_path=f"{dataset_name}.csv",
            version="1.0.0",
            tags=tags,
            metadata=metadata
        )

        dataset_exists = any(
            (
                d.name == dataset.name and d.version == dataset.version
            )
            for d in datasets
        )

        if dataset_exists:
            st.warning(
                f"Dataset '{dataset_name}' (version {dataset.version}) "
                "has already been saved."
            )

        else:
            automl.registry.register(dataset)
            st.success(f"Dataset '{dataset_name}' is successfully saved!")
            datasets = automl.registry.list(type="dataset")


st.write("# 📂 Saved Datasets")
if datasets:
    dataset_info = [{
        "Name": dataset.name,
        "Type": dataset._type,
        "ID": dataset.id, "version": dataset.version,
        "Tags": dataset.tags,
        "Metadata": dataset.metadata,
        "asset path": dataset.asset_path
    } for dataset in datasets]
    dataset_df = pd.DataFrame(dataset_info)
    st.dataframe(dataset_df)
