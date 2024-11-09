from autoop.core.ml.artifact import Artifact
import pandas as pd
import io

class Dataset(Artifact):
    """Create a class for datasets."""
    def __init__(self, *args, **kwargs) -> None:
        """Create a constructor for the Dataset class."""
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str = "1.0.0",
        tags: str = None,
        metadata: dict = None
    ) -> "Dataset":
        """Create a static method for transferring the data correctly."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
            tags=tags if not None else [],
            metadata=metadata if not None else {}
        )

    def read(self) -> pd.DataFrame:
        """Create a method for reading the dataset."""
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Inherit save method from artifact, ensure data is a csv."""
        bytes_data = data.to_csv(index=False).encode()
        return super().save(bytes_data)

