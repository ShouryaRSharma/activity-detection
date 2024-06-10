from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np


class ImageProcessingInterface(ABC):
    @abstractmethod
    def process_frame(self, frame):
        pass


class MoondreamImageProcessor:
    def __init__(self, model_id: str, revision: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def process_frame(self, frame: np.ndarray) -> str:
        """Encode the input frame using the Moondream model.

        Args:
            frame (np.ndarray): The input frame to process

        Returns:
            str: Base64 encoded string of the processed frame
        """
        image = self._preprocess_frame(frame)
        enc_image = self.model.encode_image(image)
        return enc_image

    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> Image:
        image = Image.fromarray(frame)
        return image
