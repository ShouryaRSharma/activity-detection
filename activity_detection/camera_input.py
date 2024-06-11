import time

import cv2
from abc import ABC, abstractmethod

import numpy as np

from activity_detection.logging_config import setup_logger


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
        self.target_fps = 10  # Desired frame rate
        self.original_fps = None
        self.frame_interval = None
        self.last_frame_time = None
        self.logger = setup_logger(self.__class__.__name__)

    def start_capture(self):
        """Start capturing frames from the camera."""
        if not self.capture.isOpened():
            raise IOError("Cannot open camera")

        self.original_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if self.original_fps == 0:
            raise ValueError("Unable to fetch camera FPS")

        self.frame_interval = int(self.original_fps / self.target_fps)
        self.last_frame_time = time.time()
        self.logger.info(f"Camera capture started with target FPS: {self.target_fps}")

    def stop_capture(self):
        """Stop capturing frames from the camera."""
        if self.capture:
            self.capture.release()
            self.capture = None
            self.logger.info("Camera capture stopped")

    def get_frame(self) -> np.ndarray:
        """Retrieve the current frame from the camera.

        Returns:
            np.ndarray: The current frame from the camera
        """
        if self.capture:
            current_time = time.time()
            elapsed_time = current_time - self.last_frame_time

            if elapsed_time < (1 / self.target_fps):
                time.sleep((1 / self.target_fps) - elapsed_time)

            ret, frame = self.capture.read()
            if ret:
                self.last_frame_time = time.time()
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
