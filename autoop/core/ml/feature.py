from pydantic import BaseModel, Field



class Feature(BaseModel):
    """Feature class."""
    name: str = Field()
    type: str = Field(None)

    def __str__(self) -> str:
        """Return the feature name plus type."""
        return f"Feature (name={self.name}, type={self.type})"
