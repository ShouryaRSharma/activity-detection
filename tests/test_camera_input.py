import pytest
from camera_detection.camera_input import IPCamera


@pytest.mark.xfail(reason="Not implemented yet")
def test_start_capture():
    camera = IPCamera()
    camera.start_capture()


@pytest.mark.xfail(reason="Not implemented yet")
def test_stop_capture():
    camera = IPCamera()
    camera.stop_capture()


@pytest.mark.xfail(reason="Not implemented yet")
def test_get_frame():
    camera = IPCamera()
    frame = camera.get_frame()
    assert frame is not None
