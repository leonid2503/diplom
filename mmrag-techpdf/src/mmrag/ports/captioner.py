from abc import ABC, abstractmethod


class Captioner(ABC):
    """Interface for generating text captions from images."""

    @abstractmethod
    def caption(self, image_path: str) -> str:
        """Generate a caption for an image at the given path."""
        pass
