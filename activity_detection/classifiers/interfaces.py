from abc import ABC, abstractmethod

from PIL import Image

from activity_detection.classifiers.types import Prediction
from activity_detection.devices import get_device
from activity_detection.logging_config import setup_logger


class ActivityDetectionInterface(ABC):
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.device = get_device()
        self.logger.info(f"Using device: {self.device.value}")

    @abstractmethod
    def detect_activity(self, image: Image) -> Prediction:
        pass
