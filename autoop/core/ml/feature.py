from abc import ABC

# self added


class Feature(ABC):
    """Feature class."""

    def __init__(
            self, 
            name: str, 
            type: str
        ) -> None:
        self.name = name
        self.type = type

    def __str__(self) -> str:
        """Return the feature name plus type."""
        return f"Feature (name={self.name}, type={self.type})"
