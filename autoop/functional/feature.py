from autoop.core.ml.dataset import Dataset


def is_float(value: str) -> bool:
    """Check if a string can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def detect_feature_types(dataset: Dataset) -> list:
    """
    Assumption: only categorical and numerical features and no NaN values.

    Args:
        dataset: Dataset (CSV file)

    Returns:
        List[Feature]: List of features with their types.
    """
    from autoop.core.ml.feature import Feature

    features = []
    data = dataset.read()

    # Initialize a list to store column types.
    column_types = ["numerical"] * len(data.columns)

    for _, row in data.head(100):
        for i, value in enumerate(row):
            if not str(value).isdigit() and not is_float(str(value)):
                column_types[i] = "categorical"
        return "numerical"

    for i, column_name in enumerate(data.columns):
        feature = Feature(name=column_name, type=column_types[i])
        features.append(feature)

    return features
