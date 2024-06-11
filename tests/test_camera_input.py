import pytest
from unittest.mock import Mock, patch
from activity_detection.inputs.camera_input import IPCamera, LocalCamera
import numpy as np


@patch("cv2.VideoCapture")
def test_start_capture_success(mock_video_capture):
    mock_capture = Mock()
    mock_capture.isOpened.return_value = True
    mock_video_capture.return_value = mock_capture

    camera = IPCamera("rtsp://example.com/stream")
    camera.start_capture()

    assert camera.capture == mock_capture
    assert camera.capture.isOpened.called


@patch("cv2.VideoCapture")
def test_start_capture_failure(mock_video_capture):
    mock_capture = Mock()
    mock_capture.isOpened.return_value = False
    mock_video_capture.return_value = mock_capture

    camera = IPCamera("rtsp://example.com/stream")
    with pytest.raises(IOError, match="Cannot open camera"):
        camera.start_capture()


@patch("cv2.VideoCapture")
def test_stop_capture(mock_video_capture):
    mock_capture = Mock()
    mock_video_capture.return_value = mock_capture

    camera = IPCamera("rtsp://example.com/stream")
    camera.start_capture()
    camera.stop_capture()

    assert camera.capture is None
    assert mock_capture.release.called


@patch("cv2.VideoCapture")
def test_get_frame_success(mock_video_capture):
    mock_frame = np.array([[1, 2], [3, 4]])
    mock_capture = Mock()
    mock_capture.read.return_value = (True, mock_frame)
    mock_video_capture.return_value = mock_capture

    camera = IPCamera("rtsp://example.com/stream")
    camera.start_capture()
    frame = camera.get_frame()

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert (frame == mock_frame).all()


def test_get_frame_failure():
    camera = IPCamera("rtsp://example.com/stream")
    with pytest.raises(IOError, match="Camera capture is not initialized"):
        camera.get_frame()


@patch("cv2.VideoCapture")
def test_get_frame_read_failure(mock_video_capture):
    mock_capture = Mock()
    mock_capture.read.return_value = (False, None)
    mock_video_capture.return_value = mock_capture

    camera = IPCamera("rtsp://example.com/stream")
    camera.start_capture()
    with pytest.raises(IOError, match="Failed to retrieve frame from camera"):
        camera.get_frame()


@patch("cv2.VideoCapture")
def test_local_camera_start_capture_success(mock_video_capture):
    mock_capture = Mock()
    mock_capture.isOpened.return_value = True
    mock_video_capture.return_value = mock_capture

    camera = LocalCamera()
    camera.start_capture()

    assert camera.capture == mock_capture
    assert camera.capture.isOpened.called


@patch("cv2.VideoCapture")
def test_local_camera_start_capture_failure(mock_video_capture):
    mock_capture = Mock()
    mock_capture.isOpened.return_value = False
    mock_video_capture.return_value = mock_capture

    camera = LocalCamera()
    with pytest.raises(IOError, match="Cannot open camera"):
        camera.start_capture()
