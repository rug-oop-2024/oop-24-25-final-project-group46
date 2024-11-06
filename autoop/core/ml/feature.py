from pydantic import BaseModel, Field

# self added


class Feature(BaseModel):
    """Feature class."""

    name: str = Field(default=None)
    type: str = Field(default=None)

    def __str__(self):
        """Return the feature name plus type."""
        return f"Feature (name={self.name}, type={self.type})"
