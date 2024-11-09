import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> str:    
    """Write the helper text (?)."""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine\
    learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


if not datasets:
    st.warning("No datasets saved, please add them to the registry")
else:
    # Use a select box to let the user choose a dataset
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a Dataset", dataset_names)

    # Retrieve the selected dataset
    selected_dataset = next((dataset for dataset in datasets if dataset.name == selected_dataset_name), None)

    if selected_dataset:
        # Display information about the selected dataset
        st.write("### Selected Dataset Details")
        st.write(f"**Name**: {selected_dataset.name}")
        st.write(f"**Version**: {selected_dataset.version}")
        st.write(f"**Tags**: {', '.join(selected_dataset.tags)}")
        st.write(f"**Metadata**: {selected_dataset.metadata}")
        
        # Check if selected_dataset.data is available and is a DataFrame
        if selected_dataset.data is not None:
            if isinstance(selected_dataset.data, pd.DataFrame):
                st.write("### Dataset Preview")
                st.dataframe(selected_dataset.data.head())  # Display the first few rows
            else:
                st.warning("The dataset is not in a DataFrame format and cannot be displayed as a table.")
        else:
            st.warning("No data available for the selected dataset.")