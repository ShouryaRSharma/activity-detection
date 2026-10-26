import numpy as np

from activity_detection.classifiers.types import Prediction
from activity_detection.security.security_module import SecurityModule


class FakeCapture:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.frames = []

    def start_video_capture(self):
        self.started += 1

    def stop_video_capture(self):
        self.stopped += 1

    def capture_frame(self, frame):
        self.frames.append(frame)


class FakeLogging:
    def __init__(self):
        self.suspicious = 0
        self.clear = 0

    def log_suspicious_activity(self):
        self.suspicious += 1

    def log_no_suspicious_activity(self):
        self.clear += 1


def test_security_module_starts_and_stops_capture():
    capture = FakeCapture()
    logging = FakeLogging()
    module = SecurityModule(capture, logging, stop_threshold=2)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    module.process_frame(frame, Prediction(detected=True))
    module.process_frame(frame, Prediction(detected=False))
    module.process_frame(frame, Prediction(detected=False))

    assert capture.started == 1
    assert capture.stopped == 1
    assert logging.suspicious == 1
    assert logging.clear == 1
    assert len(capture.frames) == 2
