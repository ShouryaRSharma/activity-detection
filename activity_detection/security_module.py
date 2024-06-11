from abc import ABC, abstractmethod
import cv2
import numpy as np
import datetime

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
                self.output_file, fourcc, 10.0, frame_size
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
            self.logger.info("Frame captured...")


class SecurityLoggingInterface(ABC):
    @abstractmethod
    def log_suspicious_activity(self):
        pass

    @abstractmethod
    def log_no_suspicious_activity(self):
        pass


class DefaultSecurityLogging(SecurityLoggingInterface):
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def log_suspicious_activity(self):
        self.logger.info("Suspicious activity detected!")

    def log_no_suspicious_activity(self):
        self.logger.info("No suspicious activity detected.")


class SecurityModule:
    def __init__(
        self,
        video_capture: VideoCaptureInterface,
        security_logging: SecurityLoggingInterface,
    ):
        self.video_capture = video_capture
        self.security_logging = security_logging
        self.suspicious_activity = False

    def process_frame(self, frame: np.ndarray, suspicious_activity: bool):
        if suspicious_activity and not self.suspicious_activity:
            self.video_capture.start_video_capture()
            self.security_logging.log_suspicious_activity()
            self.suspicious_activity = True
        elif not suspicious_activity and self.suspicious_activity:
            self.video_capture.stop_video_capture()
            self.security_logging.log_no_suspicious_activity()
            self.suspicious_activity = False

        if self.suspicious_activity:
            self.video_capture.capture_frame(frame)
