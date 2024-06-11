import pytest
from activity_detection.security.security_capture import DefaultVideoCapture


@pytest.mark.xfail(reason="Not implemented yet")
def test_start_video_capture():
    video_capture = DefaultVideoCapture()
    video_capture.start_video_capture()


@pytest.mark.xfail(reason="Not implemented yet")
def test_stop_video_capture():
    video_capture = DefaultVideoCapture()
    video_capture.stop_video_capture()


@pytest.mark.xfail(reason="Not implemented yet")
def test_save_video():
    video_capture = DefaultVideoCapture()
    video_data = None
    video_capture.save_video(video_data)
