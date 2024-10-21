from pydantic import BaseModel, Field
import base64

class Artifact(BaseModel):
    name: str
    asset_path: str
    data: bytes = Field(..., description="Binary representation of the artifact, e.g., model, dataset, etc.")
    version: str = "1.0.0"
    metadata: dict = Field(default_factory=dict, description="Metadata related to the artifact")
    
    @property
    def id(self) -> str:
        """Generates a unique ID for the artifact based on asset_path and version."""
        encoded_path = base64.urlsafe_b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """
        Reads the binary data of the artifact.
        This method should return the binary data stored in the artifact.
        """
        return self.data

    def save(self, data: bytes):
        """
        Saves the provided data into the artifact's data field.
        This method updates the binary data of the artifact.
        """
        self.data = data
    pass