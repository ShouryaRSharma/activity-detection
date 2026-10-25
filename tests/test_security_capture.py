from unittest.mock import patch

import numpy as np
import pytest

from activity_detection.security.security_capture import DefaultVideoCapture


class FakeWriter:
    instances = []

    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.size = size
        self.frames = []
        self.released = False
        FakeWriter.instances.append(self)

    @staticmethod
    def fourcc(*args):
        return 0

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def reset_fake_writers():
    FakeWriter.instances.clear()


def test_capture_writes_into_output_dir_with_frame_size(tmp_path):
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    capture = DefaultVideoCapture(str(tmp_path / "clips"))

    with patch(
        "activity_detection.security.security_capture.cv2.VideoWriter",
        FakeWriter,
    ):
        capture.start_video_capture()
        capture.capture_frame(frame)
        capture.stop_video_capture()

    writer = FakeWriter.instances[0]
    assert writer.size == (20, 10)
    assert writer.frames == [frame]
    assert writer.released is True
    assert str(tmp_path / "clips") in writer.path


def test_capture_rotates_after_max_clip_seconds(tmp_path):
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    capture = DefaultVideoCapture(str(tmp_path / "clips"), max_clip_seconds=1.0)

    with patch(
        "activity_detection.security.security_capture.cv2.VideoWriter",
        FakeWriter,
    ), patch(
        "activity_detection.security.security_capture.time.monotonic",
        side_effect=[0.0, 2.0, 2.0],
    ):
        capture.start_video_capture()
        capture.capture_frame(frame)
        capture.capture_frame(frame)
        capture.stop_video_capture()

    assert len(FakeWriter.instances) == 2
    assert FakeWriter.instances[0].released is True
    assert FakeWriter.instances[1].frames == [frame]


def test_stop_without_start_is_safe(tmp_path):
    capture = DefaultVideoCapture(str(tmp_path / "clips"))
    capture.stop_video_capture()
