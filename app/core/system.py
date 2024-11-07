from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """Register the artifacts."""
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """Construct the class."""
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """Register the artifact."""
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """List the artifact in the database."""
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """Get the artifact from the database."""
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, id: str) -> None:
        """Delete the artifact from the database."""
        entries = self._database.list("artifacts")
        print("Current stored artifacts:")
        for id, entry in entries:
            print(f"Stored artifact ID: {id}, Entry data: {entry}")
        data = self._database.get("artifacts", id)
        if data is None:
            print(f"Artifact with ID {id} does not exist.")
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", id)


class AutoMLSystem:
    """Class for an automated multiple linear system."""
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """Construct the AutoMLSystem class."""
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """Get the instance from the system."""
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """Register the artifact."""
        return self._registry
