from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO, YOLOWorld
from ultralytics.engine.model import Model
from ultralytics.engine.results import Boxes

from activity_detection.classifiers.interfaces import (
    ActivityDetectionInterface,
)
from activity_detection.classifiers.types import Prediction


def validate_model_reference(model_path: str) -> str:
    """Reject remote, traversal, and non-PyTorch model references."""
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("model_path must be a non-empty string")
    if "://" in model_path:
        raise ValueError("Remote model URLs are not allowed")

    path = Path(model_path)
    if ".." in path.parts:
        raise ValueError("model_path must not contain '..'")
    if path.suffix.lower() != ".pt":
        raise ValueError("Only .pt model files are supported")

    if path.is_absolute():
        if not path.is_file():
            raise ValueError("Absolute model paths must point to an existing file")
        return str(path)

    cwd = Path.cwd().resolve()
    resolved = (cwd / path).resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError as exc:
        raise ValueError("Model path escapes the working directory") from exc
    if resolved.exists() and not resolved.is_file():
        raise ValueError("Model path must point to a file")
    return model_path


class BaseYOLODetector(ActivityDetectionInterface, ABC):
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        super().__init__()
        self.model = self.load_model(validate_model_reference(model_path)).to(
            self.device.value
        )
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def load_model(self, model_path: str) -> Model:
        pass

    def detect_activity(self, image: Image) -> Prediction:
        """Detect if there are people in the input image and return their bounding boxes.
        Args:
            image (Image): The input image to analyze
        Returns:
            Prediction: A Prediction object containing a detection flag and a list of bounding boxes
        """
        person_detected = False
        bounding_boxes = []
        results = self.model.predict(image)
        for result in results:
            boxes: Boxes = result.boxes
            for box in boxes:
                if (
                    "person" in result.names.values()
                    and box.conf > self.confidence_threshold
                ):
                    person_detected = True
                    co_ords = (
                        box.xyxy.cpu().numpy().flatten()
                        if isinstance(box.xyxy, torch.Tensor)
                        else box.xyxy.flatten()
                    )
                    bounding_boxes.append(co_ords)
        num_people = len(bounding_boxes)
        self.logger.info(f"Number of people detected: {num_people}")
        return Prediction(
            detected=person_detected, bounding_box=np.array(bounding_boxes)
        )


class YOLOWorldActivityDetector(BaseYOLODetector):
    def __init__(
        self, model_path: str = "yolov8m-worldv2.pt", confidence_threshold: float = 0.5
    ):
        super().__init__(model_path, confidence_threshold)
        self.model.set_classes(["person"])

    def load_model(self, model_path: str) -> Model:
        return YOLOWorld(model_path)


class YOLOActivityDetector(BaseYOLODetector):
    def __init__(
        self, model_path: str = "yolo11m.pt", confidence_threshold: float = 0.5
    ):
        super().__init__(model_path, confidence_threshold)

    def load_model(self, model_path: str) -> Model:
        return YOLO(model_path)
