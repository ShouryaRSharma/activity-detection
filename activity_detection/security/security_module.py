import numpy as np

from activity_detection.security.security_capture import VideoCaptureInterface
from activity_detection.security.security_logging import SecurityLoggingInterface


class SecurityModule:
    def __init__(
        self,
        video_capture: VideoCaptureInterface,
        security_logging: SecurityLoggingInterface,
        stop_threshold: int = 5,
    ):
        self.video_capture = video_capture
        self.security_logging = security_logging
        self.suspicious_activity = False
        self.no_activity_count = 0
        self.stop_threshold = stop_threshold

    def process_frame(self, frame: np.ndarray, suspicious_activity: bool):
        """Process the frame and log any suspicious activity.

        We start capturing video immediately suspicious activity is detected and
        stop capturing video when no suspicious activity is detected for
        a certain number of frames (stop_threshold). This reduces the risk of
        missing anything, while making sure we mitigate false negatives as well
        and spamming too many videos.

        Args:
            frame (np.ndarray): The frame to process
            suspicious_activity (bool): Whether suspicious activity is detected
        """
        if suspicious_activity:
            if not self.suspicious_activity:
                self.video_capture.start_video_capture()
                self.security_logging.log_suspicious_activity()
                self.suspicious_activity = True
            self.no_activity_count = 0
        else:
            if self.suspicious_activity:
                self.no_activity_count += 1
                if self.no_activity_count >= self.stop_threshold:
                    self.video_capture.stop_video_capture()
                    self.security_logging.log_no_suspicious_activity()
                    self.suspicious_activity = False
                    self.no_activity_count = 0

        if self.suspicious_activity:
            self.video_capture.capture_frame(frame)
