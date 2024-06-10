import cv2
from abc import ABC, abstractmethod

import numpy as np


class CameraInputInterface(ABC):
    @abstractmethod
    def start_capture(self):
        pass

    @abstractmethod
    def stop_capture(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass


class IPCamera(CameraInputInterface):
    def __init__(self, camera_url: str):
        self.camera_url = camera_url
        self.capture = None

    def start_capture(self):
        """Start capturing the live stream from the IP camera."""
        self.capture = cv2.VideoCapture(self.camera_url)
        if not self.capture.isOpened():
            raise IOError("Cannot open IP camera stream")
        self.capture.set(cv2.CAP_PROP_FPS, 20)

    def stop_capture(self):
        """Stop capturing the live stream from the IP camera."""
        if self.capture:
            self.capture.release()
            self.capture = None

    def get_frame(self) -> np.ndarray:
        """Retrieve the current frame from the live stream.

        Returns:
            np.ndarray: The current frame from the live stream
        """
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                return frame
            else:
                raise IOError("Failed to retrieve frame from IP camera")
        else:
            raise IOError("IP camera capture is not initialized")
