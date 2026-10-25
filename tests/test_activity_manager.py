import pytest

from activity_detection.activity_manager import ActivityManager


class FailingCamera:
    def start_capture(self):
        pass

    def stop_capture(self):
        pass

    def get_frame(self):
        raise IOError("camera offline")


class PassthroughProcessor:
    def process_frame(self, frame):
        return frame


class NoopDetector:
    def detect_activity(self, image):
        raise AssertionError("detector should not be reached")


class NoopVideoCapture:
    def stop_video_capture(self):
        pass


class NoopSecurityModule:
    video_capture = NoopVideoCapture()

    def process_frame(self, frame, prediction):
        pass


def test_worker_failure_stops_and_raises():
    manager = ActivityManager(
        FailingCamera(),
        PassthroughProcessor(),
        NoopDetector(),
        NoopSecurityModule(),
    )

    with pytest.raises(RuntimeError, match="worker thread"):
        manager.run_activity_detection()
