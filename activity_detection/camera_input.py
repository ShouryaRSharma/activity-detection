import cv2
from abc import ABC, abstractmethod


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
    def __init__(self, camera_url):
        self.camera_url = camera_url
        self.capture = None

    def start_capture(self):
        self.capture = cv2.VideoCapture(self.camera_url)
        if not self.capture.isOpened():
            raise IOError("Cannot open IP camera stream")

    def stop_capture(self):
        if self.capture:
            self.capture.release()
            self.capture = None

    def get_frame(self):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                return frame
            else:
                raise IOError("Failed to retrieve frame from IP camera")
        else:
            raise IOError("IP camera capture is not initialized")
