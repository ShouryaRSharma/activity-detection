import datetime
from abc import abstractmethod, ABC

import cv2
import numpy as np

from activity_detection.logging_config import setup_logger


class VideoCaptureInterface(ABC):
    @abstractmethod
    def start_video_capture(self):
        pass

    @abstractmethod
    def stop_video_capture(self):
        pass

    @abstractmethod
    def capture_frame(self, frame: np.ndarray):
        pass


class DefaultVideoCapture(VideoCaptureInterface):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.video_writer = None
        self.output_file = None
        self.logger = setup_logger(self.__class__.__name__)

    def start_video_capture(self):
        if not self.video_writer:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"suspicious_activity_{timestamp}.mp4"
            fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
            frame_size = (1920, 1080)
            self.video_writer = cv2.VideoWriter(
                self.output_file, fourcc, 30, frame_size
            )

        if not self.video_writer.isOpened():
            raise ValueError("Failed to open video writer")

    def stop_video_capture(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.logger.info(f"Video saved: {self.output_file}")

    def capture_frame(self, frame: np.ndarray):
        if self.video_writer:
            self.video_writer.write(frame)
