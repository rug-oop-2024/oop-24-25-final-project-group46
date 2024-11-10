import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.classification import DecisionTreeClassification
from autoop.core.ml.model.classification import KNN
from autoop.core.ml.model.classification import RandomForestClassification

# from autoop.core.ml.model.base_model import Model
from autoop.core.ml.model.regression import SupportVectorRegression
from autoop.core.ml.model.regression import DecisionTreeRegression
from autoop.core.ml.model.regression import MultipleLinearRegression

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import get_metric  # , Metric

import io

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
    # Select box for datasets.
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a Dataset", dataset_names)
    selected_dataset = next(
        (dataset for dataset in datasets if dataset.name == selected_dataset_name), 
        None
    )

    if selected_dataset:
        # Display information about the selected dataset
        st.write("### Selected Dataset Details")
        st.write(f"**Name**: {selected_dataset.name}")
        st.write(f"**Version**: {selected_dataset.version}")
        st.write(f"**Tags**: {', '.join(selected_dataset.tags)}")
        st.write(f"**Metadata**: {selected_dataset.metadata}")
        st.write(f"Type selected_dataset.data: {type(selected_dataset.data)}")


        # Check if selected_dataset.data is available and is a DataFrame
        if selected_dataset.data is not None:
            if not isinstance(selected_dataset.data, pd.DataFrame):
                selected_dataset.data = pd.read_csv(
                    io.BytesIO(selected_dataset.data)
                )
                st.write("### Dataset Preview")
                st.dataframe(selected_dataset.data.head())
        else:
            st.warning("No data available for the selected dataset.")

        detected_features = detect_feature_types(
            Dataset.from_dataframe(
                selected_dataset.data,
                name=selected_dataset.name,
                asset_path=f"{selected_dataset.name}.csv"
            )
        )
        st.write("### Detected Feature Types")
        feature_names = [feature.name for feature in detected_features]
        feature_types = {
            feature.name: feature.type for feature in detected_features
            }
        for feature in detected_features:
            st.write(f"Feature: {feature.name}, Type: {feature.type}")

        # Create the list for the input features.
        input_features = []
        # Loop over each selected feature name, 
        # create a Feature instance, and add it to input_features
        for feature_name in st.multiselect(
            "Select Input Features",
            list(feature_types.keys())
        ):
            feature_type = feature_types[feature_name]
            feature_instance = Feature(name=feature_name, type=feature_type)
            input_features.append(feature_instance)

        # Create the instance for the target feature
        selected_target_feature_name = st.selectbox(
            "Select Target Feature",
            feature_names
        )
        target_feature_type = feature_types[selected_target_feature_name]
        target_feature = Feature(
            name=selected_target_feature_name,
            type=target_feature_type
        )

        # Check the target feature type, prompt user with detected task.
        if target_feature:
            if target_feature_type == "categorical":
                task_type = "Classification"
            elif target_feature_type == "numerical":
                task_type = "Regression"
            else:
                task_type = "Unknown"
        
            st.write(f"**Suggested Task Type**: {task_type}")
        
            # Model selection based on task type
            if task_type == "Classification":
                model_name = st.selectbox(
                    "Select a Classification Model",
                    ["Decision Tree",
                     "K-Nearest Neighbors",
                     "Random Forests"]
                )
                if model_name == "Decision Tree":
                    model = DecisionTreeClassification()
                elif model_name == "K-Nearest Neighbors":
                    model = KNN()
                elif model_name == "Random Forests":
                    model = RandomForestClassification()
            
            elif task_type == "Regression":
                model_name = st.selectbox(
                    "Select a Regression Model",
                    ["Decision Tree Regressor",
                     "Multiple Linear Regression", 
                     "Support Vector Regressor"]
                )
                if model_name == "Decision Tree Regressor":
                    model = DecisionTreeRegression()
                elif model_name == "Multiple Linear Regression":
                    model = MultipleLinearRegression()
                elif model_name == "Support Vector Regressor":
                    model = SupportVectorRegression()
            else:
                st.warning("Unknown task type. Please check your target feature")

            # A slider for the split ratio of the dataset
            split_ratio = st.slider("Select Dataset Split Ratio for Training", 0.1, 0.9, 0.8)
            st.write(f"Training/Test Split Ratio: {split_ratio:.2f}/{1 - split_ratio:.2f}")

            # Define available metrics and their types
            metrics_info = {
                "mean_squared_error": "regression",
                "mean_absolute_error": "regression",
                "root_mean_squared_error": "regression",
                "accuracy": "classification",
                "precision": "classification",
                "recall": "classification",
            }

            # Define a function to display compatible metrics based on task type
            def get_compatible_metrics(task_type):
                if task_type == "Regression":
                    return {name: m_type for name, m_type in metrics_info.items() if m_type == "regression"}
                elif task_type == "Classification":
                    return {name: m_type for name, m_type in metrics_info.items() if m_type == "classification"}
                else:
                    return {}

            compatible_metrics = get_compatible_metrics(task_type)
            st.write("### Available Metrics")
            for metric_name, metric_type in compatible_metrics.items():
                st.write(f"**Metric**: {metric_name}, **Type**: {metric_type}")

            # Select metrics
            selected_metric_names = st.multiselect("Select Metrics for Evaluation", list(compatible_metrics.keys()))
            selected_metrics = []
            for metric_name in selected_metric_names:
                try:
                    metric_instance = get_metric(metric_name)
                    selected_metrics.append(metric_instance)
                except ValueError as e:
                    st.error(f"Error loading metric {metric_name}: {e}")
            st.write("Selected Metrics:")
            for metric in selected_metrics:
                st.write(f"- {metric.__class__.__name__}")

            
            # Display summary of configurations
            st.write("### Summary of Configurations")
            st.markdown(f"""
                - **Selected Dataset**: {selected_dataset_name}
                - **Input Features**: {', '.join([str(feature) for feature in input_features]) if input_features else "None selected"}
                - **Target Feature**: {target_feature}
                - **Task Type**: {task_type}
                - **Selected Model**: {model_name}
                - **Split Ratio**: {split_ratio:.2f} (Training) / {1 - split_ratio:.2f} (Testing)
                - **Metrics**: {', '.join([metric.__class__.__name__ for metric in selected_metrics]) if selected_metrics else "None selected"}
            """)

        # Train the class and report the results of the pipeline
        if st.button("Train Model"):
            # Initialize the Pipeline with the selected configurations
            pipeline = Pipeline(
                metrics=selected_metrics,
                dataset=selected_dataset,
                model=model,
                input_features=input_features,
                target_feature=target_feature,
                split=split_ratio
            )
            
            # Split the data, train, and evaluate
            st.write("### Training the Model...")
            results = pipeline.execute()
            
            # Display the results
            st.write("### Training Results")
            for metric_name, result in results.items():
                st.write(f"{metric_name}: {result}")
            
            st.success("Model training and evaluation complete!")


    
            