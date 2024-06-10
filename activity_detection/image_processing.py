from abc import ABC, abstractmethod
from PIL import Image
import numpy as np


class ImageProcessingInterface(ABC):
    @abstractmethod
    def process_frame(self, frame):
        pass


class MoondreamImageProcessor:
    def process_frame(self, frame: np.ndarray) -> str:
        """Encode the input frame using the Moondream model.

        Args:
            frame (np.ndarray): The input frame to process

        Returns:
            str: Base64 encoded string of the processed frame
        """
        image = self._preprocess_frame(frame)
        return image

    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> Image:
        image = Image.fromarray(frame)
        image = image.resize((1920, 1080))
        return image
