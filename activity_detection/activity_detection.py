from abc import ABC, abstractmethod

from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ActivityDetectionInterface(ABC):
    @abstractmethod
    def detect_activity(self, image: Image) -> bool:
        pass


class MoondreamActivityDetector(ActivityDetectionInterface):
    def __init__(self, model_id: str, revision: str):
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision
        )

    def detect_activity(self, image: Image) -> bool:
        """Detect if there is a person close to the camera in the input image.

        Args:
            image (Image): The input image to analyze

        Returns:
            bool: True if a person is close to the camera, False otherwise
        """
        prompt = "Is there a person close to the camera in this image? (Only answer 'YES' or 'NO')"
        encoded_image = self.model.encode_image(image)
        answer = self.model.answer_question(encoded_image, prompt, self.tokenizer)
        if answer not in ["YES", "NO"]:
            raise ValueError("Invalid answer from the model")
        return answer == "YES"
