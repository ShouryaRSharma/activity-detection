from abc import ABC, abstractmethod


class ActivityDetectionInterface(ABC):
    @abstractmethod
    def detect_activity(self, detection_results):
        pass


class SuspiciousActivityDetector(ActivityDetectionInterface):
    def detect_activity(self, detection_results):
        # TODO: Implement analyzing detection results to determine suspicious activity
        pass
