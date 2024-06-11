import numpy as np

from activity_detection.security.security_capture import VideoCaptureInterface
from activity_detection.security.security_logging import SecurityLoggingInterface


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
