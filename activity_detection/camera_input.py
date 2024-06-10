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


class CameraBase(CameraInputInterface):
    def __init__(self):
        self.capture = None

    def start_capture(self):
        """Start capturing frames from the camera."""
        if not self.capture.isOpened():
            raise IOError("Cannot open camera")
        self.capture.set(cv2.CAP_PROP_FPS, 20)

    def stop_capture(self):
        """Stop capturing frames from the camera."""
        if self.capture:
            self.capture.release()
            self.capture = None

    def get_frame(self) -> np.ndarray:
        """Retrieve the current frame from the camera.

        Returns:
            np.ndarray: The current frame from the camera
        """
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                return frame
            else:
                raise IOError("Failed to retrieve frame from camera")
        else:
            raise IOError("Camera capture is not initialized")


class IPCamera(CameraBase):
    def __init__(self, camera_url: str):
        super().__init__()
        self.camera_url = camera_url

    def start_capture(self):
        """Start capturing the live stream from the IP camera."""
        self.capture = cv2.VideoCapture(self.camera_url)
        super().start_capture()


class LocalCamera(CameraBase):
    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index

    def start_capture(self):
        """Start capturing frames from the local webcam."""
        self.capture = cv2.VideoCapture(self.camera_index)
        super().start_capture()
