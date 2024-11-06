import pickle
from typing import List

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model import Model
from autoop.functional.preprocessing import preprocess_features


import numpy as np


class Pipeline:
    """A class for the entire pipeline of the ML models."""
    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feat: Feature,
        split: float=0.8,
    ) -> None:
        """Create a constructor for the pipeline class."""
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feat
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feat.type == "categorical" and model.type != "classification":
            raise ValueError(
                "Model type must be classification for categorical target feature"
            )
        if target_feat.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature."
            )

    def __str__(self) -> str:
        """Create a method for returning the pipeline."""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> "Model":
        """Make a property for self model."""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Get artifacts generated during pipeline execution to be saved."""
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> str:
        """Create a method for registering an artifcact."""
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Create a method for preprocessing the features."""
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset)
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get input vectors and output vector, sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        """Create a method for splitting data into train and test sets."""
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [
            vector[
                : int(split * len(vector))] for vector in self._input_vectors
        ]
        self._test_X = [
            vector[
                int(split * len(vector)) :] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)) :]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Returns combined vectors."""
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Create a method for training the data."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """Evaluate the model."""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """Create a method for executing the previous methods."""
        self._preprocess_features()
        self._split_data()
        self._train()
       
        train_X = self._compact_vectors(self._train_X)
        train_y = self._train_y
        train_predictions = self._model.predict(train_X)

        train_metrics_results = []
        for metric in self._metrics:
            result = metric.evaluate(train_y, train_predictions)
            train_metrics_results.append((metric, result))

        test_X = self._compact_vectors(self._test_X)
        test_y = self._test_y
        test_predictions = self._model.predict(test_X)

        test_metrics_results = []
        for metric in self._metrics:
            result = metric.evaluate(test_y, test_predictions)
            test_metrics_results.append((metric, result))

        return {
            "train_metrics": train_metrics_results,
            "train_predictions": train_predictions,
            "test_metrics": test_metrics_results,
            "test_predictions": test_predictions,
        }
