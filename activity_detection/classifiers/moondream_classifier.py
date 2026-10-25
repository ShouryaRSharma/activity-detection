import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from activity_detection.classifiers.types import Prediction
from activity_detection.classifiers.interfaces import (
    ActivityDetectionInterface,
)
from PIL import Image

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class MoondreamActivityDetector(ActivityDetectionInterface):
    def __init__(
        self,
        model_id: str,
        revision: str,
        allow_remote_code: bool = False,
    ):
        super().__init__()
        if allow_remote_code and not _COMMIT_SHA.fullmatch(revision):
            raise ValueError(
                "A full 40-character commit SHA revision is required when "
                "allow_remote_code is enabled"
            )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=allow_remote_code,
            revision=revision,
            dtype=torch.float32,
        ).to(self.device.value)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_fast=True,
            trust_remote_code=allow_remote_code,
        )

    def detect_activity(self, image: Image) -> Prediction:
        """Detect if there is a person close to the camera in the input image.

        Args:
            image (Image): The input image to analyze

        Returns:
            bool: True if a person is close to the camera, False otherwise
        """
        prompt = "Is there a person close to the camera in this image? (Only answer 'YES' or 'NO')"
        encoded_image = self.model.encode_image(image)
        answer: str = self.model.answer_question(encoded_image, prompt, self.tokenizer)
        if answer.lower() not in ["yes", "no"]:
            raise ValueError("Invalid answer from the model")
        return Prediction(detected=answer.lower() == "yes")
