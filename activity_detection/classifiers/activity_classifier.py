from abc import ABC, abstractmethod

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from ultralytics import YOLOWorld

from activity_detection.devices import get_device
from activity_detection.logging_config import setup_logger


class ActivityDetectionInterface(ABC):
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.device = get_device()
        self.logger.info(f"Using device: {self.device.value}")

    @abstractmethod
    def detect_activity(self, image: Image) -> bool:
        pass


class MoondreamActivityDetector(ActivityDetectionInterface):
    def __init__(self, model_id: str, revision: str):
        super().__init__()
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
            torch_dtype=torch.float32,
        ).to(self.device.value)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, use_fast=True
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
        answer: str = self.model.answer_question(encoded_image, prompt, self.tokenizer)
        if answer.lower() not in ["yes", "no"]:
            raise ValueError("Invalid answer from the model")
        return answer.lower() == "yes"


class YOLOActivityDetector(ActivityDetectionInterface):
    def __init__(self):
        super().__init__()
        self.model = YOLOWorld("yolov8m-worldv2.pt").to(self.device.value)
        self.model.set_classes(["person"])
        self._confidence_threshold = 0.5

    def detect_activity(self, image: Image) -> bool:
        """Detect if there is a person close to the camera in the input image.

        Args:
            image (Image): The input image to analyze

        Returns:
            bool: True if a person is close to the camera, False otherwise
        """
        person_detected = False

        results = self.model.predict(image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if (
                    "person" in result.names.values()
                    and box.conf > self._confidence_threshold
                ):
                    person_detected = True
                    break

            if person_detected:
                break

        self.logger.info(
            f"Person detected: {person_detected}"
        ) if person_detected else None
        return person_detected
